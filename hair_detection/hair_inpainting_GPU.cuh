#pragma once
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include "utils.h"
#include "parameters.h"
#include "cuda_error.cuh"

class HairInpaintInfo {
public:
	int Width;
	int Height;
	int Channels;
	int Rescale;
	int NumberOfC1Elements;
	int NumberOfC3Elements;

	HairInpaintInfo(int width, int height, int channels, int rescale) {
		Width = width / rescale;
		Height = height / rescale;
		Channels = channels;
		Rescale = rescale;
		NumberOfC1Elements = width * height / rescale / rescale;
		NumberOfC3Elements = width * height * channels / rescale / rescale;
	}
};

__global__ void PDEInpainting(float* mask, float* src, float* tempSrc, int width, int height, int ch);
__global__ void PDEInpaintingSMEM(float* mask, float* src, float* tempSrc, int width, int height);

void hairInpainting(cv::Mat& src, cv::Mat& mask, uchar* dSrc, cv::Mat& dst, HairInpaintInfo info);
void normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage);