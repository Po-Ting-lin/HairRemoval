#pragma once
#include <iostream>
#include <stdio.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include "fft_convolution.cuh"
#include "utils.h"

__global__ void extractLChannelWithInstrinicFunction(
	uchar* src,
	float* dst,
	int nx,
	int ny,
	int nz
);

void getHairMaskGPU(
	cv::Mat& src,
	cv::Mat& dst,
	HairDetectionInfo para
);