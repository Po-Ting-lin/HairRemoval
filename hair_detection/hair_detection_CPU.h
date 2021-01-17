#pragma once
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "utils.h"

cv::Mat getGaborFilter(float theta, HairDetectionInfo para);
void gaborFiltering(cv::Mat& src, cv::Mat& dst, HairDetectionInfo para);
void getGrayLevelCoOccurrenceMatrix(cv::Mat& src, cv::Mat& dst);
int entropyThesholding(cv::Mat& glcm);
void cleanIsolatedComponent(cv::Mat& src, HairDetectionInfo para);
void inpaintHair(cv::Mat& src, cv::Mat& dst, cv::Mat& mask, HairDetectionInfo para);
void extractLChannel(cv::Mat& src, cv::Mat& dst);
void getHairMaskCPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo para);