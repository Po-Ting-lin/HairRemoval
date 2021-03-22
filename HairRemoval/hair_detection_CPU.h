#pragma once
#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include "parameters.h"
#include "utils.h"

cv::Mat getGaborFilter(float theta, HairDetectionInfo info);
void gaborFiltering(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info);
void cleanIsolatedComponent(cv::Mat& src, HairDetectionInfo info);
void extractLChannel(cv::Mat& src, cv::Mat& dst);
void getHairMaskCPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info);
void hairDetecting(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info);