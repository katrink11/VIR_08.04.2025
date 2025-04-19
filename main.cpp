#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>
namespace fs = std::filesystem;

struct SampleData
{
	std::string name;
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

std::vector<cv::DMatch> ratio_test(const std::vector<std::vector<cv::DMatch>> &matches12, double ratio)
{
	std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < matches12.size(); i++)
	{
		if (matches12[i].size() >= 2 && matches12[i][0].distance < ratio * matches12[i][1].distance)
			good_matches.push_back(matches12[i][0]);
	}
	return good_matches;
}

int main()
{
	std::vector<SampleData> samples;

	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	// Проверка существования директории
	std::string path("./cards");
	if (!fs::exists(path))
	{
		std::cout << "Directory " << path << " does not exist!" << std::endl;
		return -1;
	}

	for (auto &p : fs::recursive_directory_iterator(path))
	{
		if (p.path().extension() == ".png")
		{
			auto img_path = p.path().string();
			SampleData data;
			data.name = p.path().stem().string();
			data.image = cv::imread(img_path);
			if (data.image.empty())
			{
				std::cout << "Failed to parse sample: " << img_path << std::endl;
				continue;
			}
			sift->detectAndCompute(data.image, cv::noArray(), data.keypoints, data.descriptors);
			samples.push_back(data);

			std::cout << "Parsed sample: " << img_path
					  << " (keypoints: " << data.keypoints.size() << ")" << std::endl;
		}
	}

	if (samples.empty())
	{
		std::cout << "No samples loaded!" << std::endl;
		return -1;
	}

	cv::Mat target = cv::imread("./target.png");
	if (target.empty())
	{
		std::cout << "Failed to parse target" << std::endl;
		return -1;
	}

	std::vector<cv::KeyPoint> targetKeypoints;
	cv::Mat targetDescriptors;
	sift->detectAndCompute(target, cv::noArray(), targetKeypoints, targetDescriptors);

	std::cout << "Target keypoints: " << targetKeypoints.size() << std::endl;

	cv::Mat imgMatches = target.clone();

	cv::BFMatcher matcher(cv::NORM_L2);

	bool any_matches = false;

	for (auto &sample : samples)
	{
		if (sample.descriptors.empty() || targetDescriptors.empty())
		{
			std::cout << "Skipping " << sample.name << " - empty descriptors" << std::endl;
			continue;
		}

		std::vector<std::vector<cv::DMatch>> matches;
		matcher.knnMatch(sample.descriptors, targetDescriptors, matches, 2);

		std::cout << sample.name << " initial matches: " << matches.size() << std::endl;

		auto good_matches = ratio_test(matches, 0.75);

		std::cout << sample.name << " good matches: " << good_matches.size() << std::endl;

		if (good_matches.size() < 4)
		{
			std::cout << sample.name << " - not enough matches for homography" << std::endl;
			continue;
		}

		std::vector<cv::Point2f> points_sample, points_target;
		for (int i = 0; i < good_matches.size(); i++)
		{
			points_sample.push_back(sample.keypoints[good_matches[i].queryIdx].pt);
			points_target.push_back(targetKeypoints[good_matches[i].trainIdx].pt);
		}

		cv::Mat H = cv::findHomography(points_sample, points_target, cv::RANSAC);

		if (H.empty())
		{
			std::cout << sample.name << " - homography failed" << std::endl;
			continue;
		}

		std::vector<cv::Point2f> corners_sample = {{0, 0}, {static_cast<float>(sample.image.cols), 0}, {static_cast<float>(sample.image.cols), static_cast<float>(sample.image.rows)}, {0, static_cast<float>(sample.image.rows)}};
		std::vector<cv::Point2f> corners_target;

		cv::perspectiveTransform(corners_sample, corners_target, H);

		double area = cv::contourArea(corners_target);
		std::cout << sample.name << " area: " << area << std::endl;
		if (area < 1000)
		{
			std::cout << sample.name << " - area too small" << std::endl;
			continue;
		}

		for (int i = 0; i < 4; i++)
		{
			cv::line(imgMatches, corners_target[i], corners_target[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
		}

		cv::Point2f center(0, 0);
		for (const auto &pt : corners_target)
			center += pt;
		center *= (1.0 / corners_target.size());

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(sample.name, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
		cv::Point2f text_pos = center - cv::Point2f(textSize.width / 2.0, textSize.height / 2.0);
		cv::putText(imgMatches, sample.name, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

		any_matches = true;
	}

	if (!any_matches)
	{
		std::cout << "No matches found for any sample!" << std::endl;
	}

	cv::imshow("Cards", imgMatches);
	cv::waitKey(0);

	return 0;
}
