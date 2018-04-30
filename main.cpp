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
#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
//#include <Eigen/Core>

#include "latch.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

// Removes interest points that are to close to the border
/*void remove_outside( std::vector<cv::KeyPoint>& keypoints,const int width, const int height, const int border_value) {
	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [width, height,border_value](const cv::KeyPoint& kp) {return kp.pt.x <= border_value || kp.pt.y <= border_value || kp.pt.x >= width - border_value || kp.pt.y >= height - border_value; }), keypoints.end());
}*/
int main(int argc, char **argv) {
	// ------------- Configuration ------------
	
	constexpr int numkps = 2000;
	constexpr bool multithread = true;
	int threshold_latch = 100; 
	/*constexpr char name0[] = "../test.jpg";
	
	// --------------------------------
	
		// ------------- Image Read ------------
	cv::Mat image = cv::imread(name0, CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat image1 = cv::imread(name1, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}

	cv::Rect roi0(50, 50, image.cols-50, image.rows-50);
	cv::Rect roi1(0, 0, image.cols, image.rows);

	cv::Mat image0 = image(roi0); 
	cv::Mat image1 = image(roi1); */

	constexpr char name0[] = "../000000.jpg";
	constexpr char name1[] = "../000005.jpg";

	cv::Mat image0 = cv::imread(name0, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image1 = cv::imread(name1, CV_LOAD_IMAGE_GRAYSCALE);
	
	if (!image0.data || !image1.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}

	
	
	// --------------------------------


	// ------------- Detection ----------------------------
	std::cout << std::endl << "Detecting..." << std::endl;
	// Setting this to 36 is very important !!!
	int edge_treshold = 36; 
	cv::Ptr<cv::ORB> orb = cv::ORB::create(numkps, 1.2f, 8, edge_treshold, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
	
	std::vector<cv::KeyPoint> keypoints0;
	std::vector<cv::KeyPoint> keypoints1;
	orb->detect(image0, keypoints0);
	orb->detect(image1, keypoints1);
	// -------------------------------
	

	// ------------- LATCH feature descriptor  ------------
	// IMPORTANT LINES to later draw the correspondes correctly. The opencv draw 
	// function requires vector<cv::KeyPoint> and Latch uses a vector<KeyPoint> (defined in LATCH.h).
	// What I failed to notice for many hours of debugging is that the LATCH function will automatically remove 
	// keypoints close to the border (36 pixels). This causes problem.
	// This can be fixed by simply removing keypoints close to the border. 
	//remove_outside(keypoints0,image0.cols, image0.rows,36);
	//remove_outside(keypoints1,image1.cols, image1.rows,36); 

	// IMPORTANT! Convertion between OpenCV [n_keypoints x 64](uint8_t) and the latch descriptor [n_keypoints x 8](uint64_t) format
	// The opencv bruteforce matching requires the data to be in a cv::Mat format.
	// A cv::Mat with dimenstions [n_keypoints x 64] of type char/uint8_t is initialized. (A single row correspond to a keypoints with a (64x8=) 512 bits descriptor).
	cv::Mat cv_desc0_latch = cv::Mat::zeros(keypoints0.size(),64,CV_8UC1);
	cv::Mat cv_desc1_latch = cv::Mat::zeros(keypoints1.size(),64,CV_8UC1);
	
	
	// A pointer is created to point to the data of the cv::Mat. 
	uint64_t* desc0_latch = (uint64_t*)(cv_desc0_latch.data);
	uint64_t* desc1_latch = (uint64_t*)(cv_desc1_latch.data);

	
	//std::cout << "Size: " << kps0_latch.size() << ", " << keypoints0.size() << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	LATCH<multithread>(image0.data, image0.cols, image0.rows, static_cast<int>(image0.step), keypoints0, desc0_latch);
	LATCH<multithread>(image1.data, image1.cols, image1.rows, static_cast<int>(image1.step), keypoints1, desc1_latch);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	std::cout 	<< std::endl << "LATCH took " 
				<< static_cast<double>((end - start).count()) * 1e-3 / (static_cast<double>(keypoints0.size()+keypoints1.size())) 
				<< " us per desc over " << keypoints0.size()+keypoints1.size() << " desc" << (keypoints0.size()+keypoints1.size() == 1 ? "." : "s.") << std::endl << std::endl;

	std::cout 	<< std::endl << "LATCH took :  " << static_cast<double>((end - start).count()*1e-6/2) << "ms (per image)" << std::endl;
	// ------------- Matching ----------------------------
	cv::BFMatcher matcher(cv::NORM_HAMMING,true);
	std::vector< cv::DMatch > matches_latch;
	matcher.match( cv_desc0_latch, cv_desc1_latch, matches_latch);
	
	std::vector< cv::DMatch > good_matches_latch;
	
	for(unsigned int i = 0; i < matches_latch.size(); i++ ) { 
		if( matches_latch[i].distance < threshold_latch) {
			good_matches_latch.push_back( matches_latch[i]); 
			//std::cout << "Sample LATCH, queryIdx: " << matches_latch[i].queryIdx << ", trainIdx: " << matches_latch[i].trainIdx << ", Distance: "  << matches_latch[i].distance << std::endl;
		}
	}
	

	// TESTING ///////////////////
	// To validate that keypoints are identical before and after the latch-function.
	// I used many hours debugging, to notice that the latch-functions removes keypoints close than 36 pixels 
	// to the border. 
	/*for(unsigned int i = 0; i < keypoints0.size(); i++ ) {
		std::cout << "KP Latc(before) : " << keypoints0[i].pt << std::endl;
		std::cout << "KP Latc(before) : [" << kps0_latch[i].x << ", " << kps0_latch[i].y << "]" << std::endl << std::endl;
	}*/

	int print_keypoint_idx = 0; // keypoints_class.size()
	std::cout << "argc:" << argc << std::endl;
	if (argc > 1) {
		std::cout << "Select keypoint0 " << argv[1] << "/" << keypoints0.size()-1 << std::endl;
		std::cout << "Select keypoint1 " << argv[1] << "/" << keypoints1.size()-1 << std::endl;
		print_keypoint_idx = std::atoi(argv[1]);
	}

	// LATCH ORIGINAL //////////////////////////////////////////////
	std::cout << "Descriptor (LATCH original): " << print_keypoint_idx << "/" << keypoints0.size()-1 << std::endl;
	int shift = print_keypoint_idx*8;
	for (int i = 0; i < 8; ++i) {
		std::cout << std::bitset<64>(desc0_latch[i+shift]) << std::endl;
	}
	std::cout << std::endl;
	
	// LATCH CLASS ( FLIPPED!!!) /////////////////////////////////////////////////	
	std::cout << "Descriptor (LATCH cv_mat FLIPPED!!!): " << print_keypoint_idx << "/" << keypoints0.size()-1 << std::endl;
	for (int i = 0; i < 8; ++i) {
		for (int ii = 0; ii < 8; ++ii) {
			std::cout << std::bitset<8>(cv_desc0_latch.at<unsigned char>(print_keypoint_idx,i*8+7-ii));
			//std::cout << std::bitset<8>(cv_desc0_latch.at<uint8_t>(i*8+7-ii,print_keypoint_idx));
		}
		std::cout << std::endl; 
	}
	std::cout << std::endl;


	cv::Mat img_matches_latch;
	cv::drawMatches( image0, keypoints0, image1, keypoints1,
			good_matches_latch, img_matches_latch, cv::Scalar::all(-1), cv::Scalar::all(-1),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Show detected matches
	cv::namedWindow( "Good Matches (LATCH)", CV_WINDOW_NORMAL );
	cv::imshow( "Good Matches (LATCH)", img_matches_latch );

	std::cout << "LATCH: Descriptors size: " << cv_desc0_latch.size() << ", Good matches: " << good_matches_latch.size() << "(treshold " << threshold_latch << ")"  << std::endl;


	cv::waitKey(0);

}
