#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H
#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Eigen>
void basic_detect_interest_points(cv::Mat image, int n_features,std::vector<cv::KeyPoint> &keypoints);
//void basic_detect_interest_points(cv::Mat image, int n_features);
class FeatureExtractor {
    
public: 
	ORB_EXTRACTOR::ORBextractor* _orb_slam;
    cv::Ptr<cv::ORB> _orbHigh;
	cv::Ptr<cv::ORB> _orbLow;
    std::vector<cv::KeyPoint> _keypoints;
    cv::Mat _cv_image;
    cv::Mat _cv_mask;
    int _interest_point_detector_type;
    int _n_keypoints;
	static const int NUM_GRID_CELLS;
	static const int EDGE_THRESHOLD;

	FeatureExtractor(int n_keypoints, 
                     float scale_factor = 1.2f, 
                     int n_levels = 8, 
                     int threshold_high = 20, 
                     int threshold_low = 7,
                     int interest_point_detector_type = 0,
                     int keypoint_search_multiplier = 5);
    
    
    

    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> compute_descriptor(bool use_latch,int print = -1);
    void remove_outside( std::vector<cv::KeyPoint>& keypoints,const int width, const int height, const int border_value);
    Eigen::MatrixXd get_keypoints();
    Eigen::MatrixXd detect_interest_points(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image);
    void set_mask(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask);
    void internal_adaptive_non_maximum_suppression();
	void inititialze_POI_on_grid();
    //std::vector<int> adaptive_non_maximum_suppression(Eigen::MatrixXd pointLocation, int numRetPoints, float tolerance, int cols, int rows);
    //void adaptive_non_maximum_suppression(Eigen::MatrixXd pointLocation, int numInPoints, int numRetPoints, float tolerance, int cols, int rows);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> detect_and_compute(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image, bool use_latch);
    //void FeatureExtractor::detect_and_compute(cv::Mat &image,std::vector<cv::KeyPoint> &keypoints,bool use_orb_slam, bool use_latch);
};

#endif