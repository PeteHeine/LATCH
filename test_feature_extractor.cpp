#include <iostream>
#include "gtest/gtest.h"
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "feature_extractor.h"

// IndependentMethod is a test case - here, we have 2 tests for this 1 test case
TEST(FeatureExtractor, FeatureExtractor_compare) {
	int i = 3;
	//FeatureExtractor fe(1.2, 8, 20, 7, 0, 5);
	std::string name0 = "C:\\data\\unittest\\15_meter_pipe\\Images\\19700101-010004.300.jpg";
	cv::Mat image0 = cv::imread(name0, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image_show = cv::imread(name0, CV_LOAD_IMAGE_UNCHANGED);
	if (!image0.data)                              // Check for invalid input
	{
		FAIL() << "Could not open or find the image";
	}
	
	std::vector<FeatureExtractor> orb_fe;
	orb_fe.emplace_back(1500, 1.2, 8, 20, 10, 0, 10);//R
	orb_fe.emplace_back(1500, 1.2, 8, 20, 10, 1, 10);//G
	orb_fe.emplace_back(1500, 1.2, 8, 20, 10, 2, 10);//B
	orb_fe.emplace_back(1500, 1.2, 8, 20, 10, 3, 10);//Black
	std::vector<cv::Scalar> colors; //BGR
	colors.emplace_back(0, 0, 255);
	colors.emplace_back(0, 255, 0);
	colors.emplace_back(255, 0, 0);
	colors.emplace_back(255, 255, 255);

	Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image0_eig;
	cv2eigen(image0, image0_eig);
	for (int ind=0;ind< orb_fe.size();ind++)
	{
		FeatureExtractor& fe = orb_fe[ind];
		const cv::Scalar& color = colors[ind];
		Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>  descriptors = fe.detect_and_compute(image0_eig, false);
		Eigen::MatrixXd  poi_info = fe.get_keypoints().transpose();
		int thickness = -1;
		int lineType = 8;
		for (int i = 0; i < poi_info.rows(); i++)
		{
			cv::circle(image_show, cv::Point(poi_info(i, 0), poi_info(i, 1)), 9.0/(ind+1), color, -1, 8);
		}
		std::cout << "Features detected: "<<poi_info.rows() << std::endl;
	}
	
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Display window", image_show);                   // Show our image inside it.
	cv::waitKey(0);
	
}