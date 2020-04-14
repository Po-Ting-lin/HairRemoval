#include<iostream>
#include<opencv2/opencv.hpp>
#include<Dense>
#include<string>
#include <ctime>
#include<omp.h>
#include"HairRemoval.h"

using namespace cv;
using namespace std;


void displayImage(const cv::Mat& image, const char* name, bool mag) {
	cv::Mat Out;
	if (mag) {
		cv::resize(image, Out, cv::Size(), 5, 5);
	}
	else {
		image.copyTo(Out);
	}
	namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::imshow(name, Out);
	cv::waitKey(0);
}


int main() {
	Mat image, output;
	image = imread(R"(C:\Users\brian.AMO\Desktop\test.png)");

	if (!image.data) {
		cout << "Error: the image wasn't correctly loaded." << endl;
		return -1;
	}

	//displayImage(image, "image", false);

	HairRemoval obj = HairRemoval();
	///*******************************************************/
	auto start = std::chrono::system_clock::now();
	obj.process(image, output);
	auto end = std::chrono::system_clock::now();

	// time
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout << "time consume: " << elapsed_seconds.count() << endl;
	/*******************************************************/
	
	displayImage(output, "output", false);

	
	//cvtColor(image, cieImage, COLOR_RGB2Lab);

	//// save 
	//vector<int> compression_params;
	//compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(9);

	//Mat show;
	//image.copyTo(show);
	//show.setTo((255, 255, 255), afterGa == 255);
	////displayImage(show, "after dilate", false);
	////imwrite("overlay.png", show, compression_params);
	//displayImage(image, "after inpainting", false);
	////imwrite("afterimage.png", image, compression_params);

	return 0;
}

