#pragma once
#include<opencv2/opencv.hpp>
#include<algorithm>
#include<iostream>
#include<omp.h>

#define CV_8U_DYNAMICRANGE 256
#define EPSILON 1e-8

class HairRemoval
{
private:
	// Parameters of Hair Detection
	int numberOfFilter;
	double alpha;
	double beta;
	double hairWidth;

	// Parameters of Clean Isolated Component
	int minArea;
	double ratioBBox;

	// Parameters of inpainting
	int radiusOfInpaint;
	
	// method
	cv::Mat GaborFilter(double alpha, double beta, double hairW, double theta);
	cv::Mat Gabor(cv::Mat& src);
	void splitThreeChannel(cv::Mat src, cv::Mat& c1, cv::Mat& c2, cv::Mat& c3);
	void grayLevelCoOccurrenceMatrix(cv::Mat& src, cv::Mat& dst);
	int entropyThesholding(cv::Mat& glcm);
	void cleanIsolatedComponent(cv::Mat& src);
	void inpaintHair(cv::Mat& src, cv::Mat& dst, cv::Mat& mask);
	

public:
	HairRemoval() {
		numberOfFilter = 15;
		alpha = 1.4;
		beta = 0.5;
		hairWidth = 2.0;
		minArea = 20;
		ratioBBox = 4.0;
		radiusOfInpaint = 5;
	}
	bool process(cv::Mat& src, cv::Mat& dst);
};

