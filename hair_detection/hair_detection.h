#pragma once
#include<opencv2/opencv.hpp>
#include<algorithm>
#include<iostream>
#include<omp.h>
#include "parameters.h"


#define CV_8U_DYNAMICRANGE 256
#define EPSILON 1e-8

cv::Mat GaborFilter(float theta, HairDetectionParameters para);
void Gabor(cv::Mat& src, cv::Mat& dst, HairDetectionParameters para);
void grayLevelCoOccurrenceMatrix(cv::Mat& src, cv::Mat& dst);
int entropyThesholding(cv::Mat& glcm);
void cleanIsolatedComponent(cv::Mat& src, HairDetectionParameters para);
void inpaintHair(cv::Mat& src, cv::Mat& dst, cv::Mat& mask, HairDetectionParameters para);
void cvtL(cv::Mat& src, cv::Mat& dst);
bool hairDetection(cv::Mat& src, cv::Mat& dst, bool isGPU);

