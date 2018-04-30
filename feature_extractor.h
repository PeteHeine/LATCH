#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H
#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
void basic_detect_interest_points(cv::Mat image, int n_features,std::vector<cv::KeyPoint> &keypoints);
//void basic_detect_interest_points(cv::Mat image, int n_features);
class FeatureExtractor {
    
public: 
	ORB_EXTRACTOR::ORBextractor* orb_slam;
    cv::Ptr<cv::ORB> orb;
    std::vector<cv::KeyPoint> _keypoints;
    cv::Mat _cv_image;

	FeatureExtractor(int n_keypoints, float scale_factor = 1.2f, int n_levels = 8, int threshold_fast = 20);

    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> compute_descriptor(bool use_latch,int print = -1);
    void remove_outside( std::vector<cv::KeyPoint>& keypoints,const int width, const int height, const int border_value);
    Eigen::MatrixXd get_keypoints();
    Eigen::MatrixXd detect_interest_points(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image, bool use_orb_slam);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> detect_and_compute(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image, bool use_orb_slam, bool use_latch);
    //void FeatureExtractor::detect_and_compute(cv::Mat &image,std::vector<cv::KeyPoint> &keypoints,bool use_orb_slam, bool use_latch);
};

#endif