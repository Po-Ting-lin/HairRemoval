#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include "hair_detection_GPU.cuh"
#include "fft_convolution.cuh"
#include "entropy_thresholding.cuh"
#include "entropy_thresholding.h"
#include "hair_detection_CPU.h"

bool hairDetection(cv::Mat& src, cv::Mat& dst, bool isGPU);