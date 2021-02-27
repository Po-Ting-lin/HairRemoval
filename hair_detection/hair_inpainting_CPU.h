#pragma once
#include <omp.h>
#include <immintrin.h>
#include "utils.h"
#include "parameters.h"
#include "cuda_error.cuh"

void normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage, bool channelSplit);
void mergeChannels(float* srcImage, float* dstImage, HairInpaintInfo info);
void hairInpaintingCPU(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info);
void hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info);