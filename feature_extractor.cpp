/*******************************************************************
*   main.cpp
*   LATCH
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Sep 12, 2016
*******************************************************************/
//
// Fastest implementation of the fully scale-
// and rotation-invariant LATCH 512-bit binary
// feature descriptor as described in the 2015
// paper by Levi and Hassner:
//
// "LATCH: Learned Arrangements of Three Patch Codes"
// http://arxiv.org/abs/1501.03719
//
// See also the ECCV 2016 Descriptor Workshop paper, of which I am a coauthor:
//
// "The CUDA LATCH Binary Descriptor"
// http://arxiv.org/abs/1609.03986
//
// And the original LATCH project's website:
// http://www.openu.ac.il/home/hassner/projects/LATCH/
//
// See my GitHub for the CUDA version, which is extremely fast.
//
// My implementation uses multithreading, SSE2/3/4/4.1, AVX, AVX2, and 
// many many careful optimizations to implement the
// algorithm as described in the paper, but at great speed.
// This implementation outperforms the reference implementation by 800%
// single-threaded or 3200% multi-threaded (!) while exactly matching
// the reference implementation's output and capabilities.
//
// If you do not have AVX2, uncomment the '#define NO_AVX_PLEASE' in LATCH.h to route the code
// through SSE isntructions only. NOTE THAT THIS IS ABOUT 50% SLOWER.
// A processor with full AVX2 support is highly recommended.
//
// All functionality is contained in the file LATCH.h. This file
// is simply a sample test harness with example usage and
// performance testing.
//

#include <bitset>
#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/LU>
#include <opencv2/core/eigen.hpp>
//#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/videoio.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/stitching/detail/matchers.hpp>
//#include <opencv2/videoio.hpp>
#include "ORBextractor.h"


#include <vector>

#include "latch.h"
#include "feature_extractor.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;
using namespace ORB_EXTRACTOR;

FeatureExtractor::FeatureExtractor(int n_keypoints, float scale_factor, int n_levels, int threshold_high, int threshold_low, int interest_point_detector_type, int keypoint_search_multiplier){
	//int edge_treshold = 36; 
	int edge_treshold =  31;
	_interest_point_detector_type = interest_point_detector_type;
	_n_keypoints = n_keypoints;
	if(_interest_point_detector_type == 0){
		orb = cv::ORB::create(n_keypoints, scale_factor, n_levels, edge_treshold, 0, 2, cv::ORB::HARRIS_SCORE, 31, threshold_high);
	}
	else if(_interest_point_detector_type == 1){
		orb = cv::ORB::create(n_keypoints*keypoint_search_multiplier, scale_factor, n_levels, edge_treshold, 0, 2, cv::ORB::HARRIS_SCORE, 31, threshold_low);
	}
	else if(_interest_point_detector_type == 2){
		orb_slam = new ORBextractor(n_keypoints, scale_factor, n_levels,threshold_high, threshold_low);
	}
	//orb->setMaxFeatures(n_keypoints);
}

Eigen::MatrixXd FeatureExtractor::detect_interest_points(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> eigen_image)
{	
	//cv::Mat cv_image(image.rows(),image.cols(),CV_8UC1);
	cv::eigen2cv(eigen_image,_cv_image);	
	
	if((_interest_point_detector_type == 0) || (_interest_point_detector_type == 1)){
		FeatureExtractor::orb->detect(_cv_image, _keypoints);

		if (_interest_point_detector_type == 1){
			internal_adaptive_non_maximum_suppression();
		}
	}
	else if (_interest_point_detector_type == 2) {
		(*FeatureExtractor::orb_slam)(_cv_image, _keypoints);
	}

	// Remove points close to the border 
	//remove_outside(_keypoints,_cv_image.cols,_cv_image.rows,36);
	
	// x,y,scale, angle, response
	Eigen::MatrixXd interest_points(5,_keypoints.size());

	for(unsigned int i_kp = 0; i_kp < _keypoints.size(); i_kp++){
		interest_points(0,i_kp) = _keypoints[i_kp].pt.x;
		interest_points(1,i_kp) = _keypoints[i_kp].pt.y;
		interest_points(2,i_kp) = _keypoints[i_kp].size;
		interest_points(3,i_kp) = _keypoints[i_kp].angle;
		interest_points(4,i_kp) = _keypoints[i_kp].response;
	}
	return interest_points;
}

Eigen::MatrixXd FeatureExtractor::get_keypoints()
{	
	// x,y,scale, angle, response
	Eigen::MatrixXd interest_points(5,_keypoints.size());

	for(unsigned int i_kp = 0; i_kp < _keypoints.size(); i_kp++){
		interest_points(0,i_kp) = _keypoints[i_kp].pt.x;
		interest_points(1,i_kp) = _keypoints[i_kp].pt.y;
		interest_points(2,i_kp) = _keypoints[i_kp].size;
		interest_points(3,i_kp) = _keypoints[i_kp].angle;
		interest_points(4,i_kp) = _keypoints[i_kp].response;
	}
	return interest_points;
}
void FeatureExtractor::remove_outside( std::vector<cv::KeyPoint>& keypoints,const int width, const int height, const int border_value) {
	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [width, height,border_value](const cv::KeyPoint& kp) {return kp.pt.x <= border_value || kp.pt.y <= border_value || kp.pt.x >= width - border_value || kp.pt.y >= height - border_value; }), keypoints.end());
}
Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> FeatureExtractor::compute_descriptor(bool use_latch,int print_kp_idx)
{	
	const int n_found_kp = (int)(_keypoints.size());
	constexpr bool multithread = true;
	Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> eigen_desc;
	if(use_latch){

		// Using cv::Mat. Smart when using opencv. ///////////////////////////////////////////
		/*if(false){
			cv::Mat cv_desc = cv::Mat::zeros(n_found_kp,64,CV_8UC1);

			// A pointer is created to point to the data of the cv::Mat. 
			uint64_t* p_desc = (uint64_t*)(cv_desc.data);

			// Calculate LATCH descriptors 
			LATCH<multithread>(_cv_image.data, _cv_image.cols, _cv_image.rows, static_cast<int>(_cv_image.step), _keypoints, p_desc);

			cv::cv2eigen(cv_desc,eigen_desc);
			/////////////////////////////////////////////////////////////////////////////////
		}*/
		
		// Using eigen. Smart when data is being return using pybind11, that uses eigen.  /////////////////
		eigen_desc = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(64,n_found_kp);
		uint64_t* p_desc = (uint64_t*)(eigen_desc.data());

		// Calculate LATCH descriptors 
		LATCH<multithread>(_cv_image.data, _cv_image.cols, _cv_image.rows, static_cast<int>(_cv_image.step), _keypoints, p_desc);
		/////////////////////////////////////////////////////////////////////////////////

		if(print_kp_idx>-1){
			// LATCH ORIGINAL //////////////////////////////////////////////
			std::cout << "Descriptor (LATCH original): " << print_kp_idx << "/" << _keypoints.size()-1 << std::endl;
			int shift = print_kp_idx*8;
			for (int i = 0; i < 8; ++i) {
				std::cout << std::bitset<64>(p_desc[i+shift]) << std::endl;
			}
			std::cout << std::endl;

			// LATCH CLASS (FLIPPED) /////////////////////////////////////////////////	
			std::cout << "LATCH (FLIPPED)" << std::endl;
			for (int i = 0; i < 8; ++i) {
				for (int ii = 0; ii < 8; ++ii) {
					std::cout << std::bitset<8>(eigen_desc(i*8+7-ii,print_kp_idx));
				}
				std::cout << std::endl; 
			}
			std::cout << std::endl;
		}

		std::cout << "n_keypoints: " << n_found_kp << " == " << _keypoints.size() << std::endl;
		// ASSERTION: The LATCH function will remove keypoints close to the border. This must be avoided
		// as this will create a mismatch between the index of descriptors and keypoints. 
		assert(!(n_found_kp==_keypoints.size()) && "The LATCH function have remove keypoints close to the border. This must be avoided as this will create a mismatch between the index of descriptors and keypoints." );

	}
	else {
		cv::Mat cv_desc;
		orb->compute(_cv_image, _keypoints,cv_desc);
		cv::cv2eigen(cv_desc,eigen_desc);
	}

	
	
	return eigen_desc;
}

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> FeatureExtractor::detect_and_compute(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> eigen_image, bool use_latch)
{	
	cv::eigen2cv(eigen_image,_cv_image);
	Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> eigen_desc;

	if (use_latch == false) {
		if (_interest_point_detector_type==0){
			cv::Mat cv_desc;
			FeatureExtractor::orb->detectAndCompute(_cv_image, cv::noArray(), _keypoints,cv_desc);
			
			// Return for no interest point.
			if(cv_desc.rows == 0){	
				_keypoints.clear();
				return eigen_desc;
			}
			cv::cv2eigen(cv_desc,eigen_desc);
		}
		else if (_interest_point_detector_type==1){
			FeatureExtractor::orb->detect(_cv_image, _keypoints);
			internal_adaptive_non_maximum_suppression();
			eigen_desc = compute_descriptor(false);
		}
		else if (_interest_point_detector_type==2){
			cv::Mat cv_desc;
			(*FeatureExtractor::orb_slam)(_cv_image, cv::noArray(),_keypoints,cv_desc);

			// Return for no interest point.
			if(cv_desc.rows == 0){
				_keypoints.clear();
				return eigen_desc;
			}
			cv::cv2eigen(cv_desc,eigen_desc);
		}
	}
	// Latch-based (CURRENTLY NOT WORKING)
	else {
		if((_interest_point_detector_type == 0) || (_interest_point_detector_type == 1)){
			FeatureExtractor::orb->detect(_cv_image, _keypoints);

			if (_interest_point_detector_type == 1){
				internal_adaptive_non_maximum_suppression();
			}
		}
		else if (_interest_point_detector_type == 2) {
			(*FeatureExtractor::orb_slam)(_cv_image, _keypoints);
		}
		// Bug fixing
		remove_outside(_keypoints,_cv_image.cols,_cv_image.rows,36);

		eigen_desc = compute_descriptor(true);
	}
	
	return eigen_desc;
}

void FeatureExtractor::internal_adaptive_non_maximum_suppression(){
//void FeatureExtractor::adaptive_non_maximum_suppression(Eigen::MatrixXd pointLocation, int numInPoints, int numRetPoints, float tolerance, int cols, int rows){
	//std::cout << "Function called successfully " << std::endl;
	//std::cout << "STEP0 " << std::endl;
    // several temp expression variables to simplify solution equation
	//std::cout << "pointLocation.size " << pointLocation.rows <<  std::endl;
	//std::vector<cv::KeyPoint> keypoints, int numRetPoints, float tolerance, int cols, int rows
	
	// Sort points according to response. 
	std::sort(_keypoints.begin(), _keypoints.end(), [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response; });

	float tolerance = 0.1f;
	int cols = _cv_image.cols;
	int rows = _cv_image.rows;
	int numInPoints = _keypoints.size();
	int numRetPoints = _n_keypoints;
    int exp1 = rows + cols + 2*numRetPoints;
    long long exp2 = ((long long) 4*cols + (long long)4*numRetPoints + (long long)4*rows*numRetPoints + (long long)rows*rows + (long long) cols*cols - (long long)2*rows*cols + (long long)4*rows*cols*numRetPoints);
    double exp3 = sqrt(double(exp2));
    double exp4 = (2*(numRetPoints - 1));

    double sol1 = -round((exp1+exp3)/exp4); // first solution
    double sol2 = -round((exp1-exp3)/exp4); // second solution

    double high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    double low = floor(sqrt((double)numInPoints/numRetPoints));

    double width;
    int prevWidth = -1;
	//std::cout << "STEP0 " << std::endl;
    std::vector<int> ResultVec;
    bool complete = false;
    unsigned int K = numRetPoints; 
	unsigned int Kmin = round(K-(K*tolerance)); 
	unsigned int Kmax = round(K+(K*tolerance));
    
    std::vector<int> result; 
	result.reserve(numInPoints);
	
	//std::cout <<  exp1 << ", " <<  exp2 << ", " <<  exp3 << ", " << exp4 << ", " << sol1 << ", " << sol2 << ", " << high << ", " << low << ", " << K << ", " << Kmin << ", " << Kmax << std::endl;

    while(!complete){
        width = low+(high-low)/2;
        if (width == prevWidth || low>high) { //needed to reassure the same radius is not repeated again
            ResultVec = result; //return the keypoints from the previous iteration
            break;
        }
        result.clear();
        double c = width/2; //initializing Grid
        int numCellCols = floor(cols/c);
        int numCellRows = floor(rows/c);
		
        std::vector<std::vector<bool> > coveredVec(numCellRows+1,std::vector<bool>(numCellCols+1,false));

        for (unsigned int i=0;i<numInPoints;++i){
			int row = floor(_keypoints[i].pt.y/c); //get position of the cell current point is located at
            int col = floor(_keypoints[i].pt.x/c);

			
            if (coveredVec[row][col]==false){ // if the cell is not covered
                result.push_back(i);
                int rowMin = ((row-floor(width/c))>=0)? (row-floor(width/c)) : 0; //get range which current radius is covering
                int rowMax = ((row+floor(width/c))<=numCellRows)? (row+floor(width/c)) : numCellRows;
                int colMin = ((col-floor(width/c))>=0)? (col-floor(width/c)) : 0;
                int colMax = ((col+floor(width/c))<=numCellCols)? (col+floor(width/c)) : numCellCols;
                for (int rowToCov=rowMin; rowToCov<=rowMax; ++rowToCov){
                    for (int colToCov=colMin ; colToCov<=colMax; ++colToCov){
                        if (!coveredVec[rowToCov][colToCov]) coveredVec[rowToCov][colToCov] = true; //cover cells within the square bounding box with width w
                    }
                }
            }
        }
        if (result.size()>=Kmin && result.size()<=Kmax){ //solution found
            ResultVec = result;
            complete = true;
			//std::cout << "STEP END0 " << std::endl;
        }
        else if (result.size()<Kmin) high = width-1; //update binary search range
        else low = width+1;
        prevWidth = width;
    }

	// 
	std::vector<cv::KeyPoint> tmpKeypoints;
	//std::cout << "STEP END1 " << std::endl;
	for(std::vector<int>::iterator it = ResultVec.begin(); it != ResultVec.end(); ++it) {
     	tmpKeypoints.push_back(_keypoints[*it]);
	}
	_keypoints = tmpKeypoints;
}



int main(int argc, const char * argv[]) {
	// ------------- Configuration ------------
	//constexpr int warmups = 30;
	constexpr int runs = 100;
	constexpr int numkps = 2000;
	constexpr bool multithread = true;
	constexpr char name[] = "../pipe.jpg";

	int n_runs = 10;
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------
	
	
	int edge_treshold = 36; // THIS VALUE IS IMPORTANT !!!
	cv::Ptr<cv::ORB> orb = cv::ORB::create(numkps, 1.2f, 8, edge_treshold, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); //
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptors;
	orb->setMaxFeatures(numkps);

	// ------------- Interst Point detector (opencv) ------------
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	for (int i = 0; i < n_runs; ++i){
		orb->detect(image, keypoints);
	}
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < n_runs; ++i){
		orb->compute(image, keypoints,descriptors);
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//std::cout << "Interst Point detector (opencv): " << static_cast<double>((t1 - t0).count()) / (double)(n_runs) * 1e-6 <<  "ms" << std::endl;
	//std::cout << "Descriptor (opencv): " << static_cast<double>((t2 - t1).count()) / (double)(n_runs) * 1e-6 <<  "ms" << std::endl;
	std::cout << "Interst Point detector(" << static_cast<double>((t1 - t0).count()) / (double)(n_runs) * 1e-6 
			  << ") and descriptor (" << static_cast<double>((t2 - t1).count()) / (double)(n_runs) * 1e-6 
			  << ") (seperated - opencv): " << static_cast<double>((t2 - t0).count()) / (double)(n_runs) * 1e-6 <<  "ms" << std::endl;
	// -----------------------------------------------------------
	
	// ------------- Interst Point detector and descriptor (opencv) ------------
	t0 = high_resolution_clock::now();
	for (int i = 0; i < n_runs; ++i){
		orb->detectAndCompute(image, cv::noArray(), keypoints1,descriptors);
	}
	t1 = high_resolution_clock::now();
	std::cout << "Interst Point detector and descriptor (jointed - opencv): " << static_cast<double>((t1 - t0).count()) / (double)(n_runs) * 1e-6 <<  "ms" << std::endl;
	// -----------------------------------------------------------


	// ------------- LATCH ------------
	uint64_t* desc = new uint64_t[8 * keypoints.size()];
	// For testing... Force values to zero. 
	for (int i = 0; i < 8 * keypoints.size(); ++i) {
		desc[i] = 0;
	}

	high_resolution_clock::time_point start1 = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) LATCH<multithread>(image.data, image.cols, image.rows, static_cast<int>(image.step), keypoints, desc);
	high_resolution_clock::time_point end1 = high_resolution_clock::now();
	// --------------------------------
	
	

	//std::cout << std::endl << "LATCH (warmup) took " << static_cast<double>((end0 - start0).count()) * 1e-3 / (static_cast<double>(warmups) * static_cast<double>(kps.size())) << " us per desc over " << kps.size() << " desc" << (kps.size() == 1 ? "." : "s.") << std::endl << std::endl;
	std::cout << std::endl << "LATCH (after) took " << static_cast<double>((end1 - start1).count()) * 1e-3 / (static_cast<double>(runs) * static_cast<double>(keypoints.size())) << " us per desc over " << keypoints.size() << " desc" << (keypoints.size() == 1 ? "." : "s.") << std::endl << std::endl;

	// TESTING ///////////////////
	int print_desc_idx = 0; // keypoints_class.size()
	std::cout << "argc:" << argc << std::endl;
	if (argc > 1) {
		print_desc_idx = std::atoi(argv[1]);
	}
	std::cout << "Select keypoint: " << argv[1] << std::endl;

	// ------------- FeatureExtractor Class ------------
	FeatureExtractor feature_extractor = FeatureExtractor(numkps);
	

	Eigen::Matrix<uint8_t,Eigen::Dynamic,Eigen::Dynamic> eig_image;
	cv::cv2eigen(image,eig_image);
	t0 = high_resolution_clock::now();
	for (int i = 0; i < n_runs; ++i){
		feature_extractor.detect_interest_points(eig_image);
	}
	t1 = high_resolution_clock::now();

	std::vector<cv::KeyPoint> keypoints_class = feature_extractor._keypoints;

	Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> eigen_desc;
	eigen_desc = feature_extractor.compute_descriptor(true,print_desc_idx);
	

	std::cout << "Interst Point detector (Class) " << static_cast<double>((t1 - t0).count()) / (double)(n_runs) * 1e-6 <<  "ms" << std::endl;
	// -----------------------------------------------------------

	

	// LATCH ORIGINAL //////////////////////////////////////////////
	std::cout << "Descriptor (LATCH original): " << print_desc_idx << "/" << keypoints_class.size()-1 << std::endl;
	int shift = print_desc_idx*8;
	for (int i = 0; i < 8; ++i) {
		std::cout << std::bitset<64>(desc[i+shift]) << std::endl;
	}
	std::cout << std::endl;

	// LATCH CLASS (FLIPPED) /////////////////////////////////////////////////	
	std::cout << "LATCH (FLIPPED)" << std::endl;
	for (int i = 0; i < 8; ++i) {
		for (int ii = 0; ii < 8; ++ii) {
			std::cout << std::bitset<8>(eigen_desc(i*8+7-ii,print_desc_idx));
		}
		std::cout << std::endl; 
	}
	std::cout << std::endl;

	long long total = 0;
	for (size_t i = 0; i < 8 * keypoints_class.size(); ++i) total += desc[i];
	std::cout << "Checksum: " << std::hex << total << std::endl << std::endl;

	
}
