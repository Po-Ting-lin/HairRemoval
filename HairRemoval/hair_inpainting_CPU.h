#pragma once
#include <omp.h>
#include "utils.h"
#include "parameters.h"
#include "cuda_error.cuh"

void normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage, HairInpaintInfo info);
void convertToMatArrayFormat(float* srcImage, uchar* dstImage, HairInpaintInfo info);
void hairInpaintingCPU(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info);
void PDEHeatDiffusionCPU(float* normalized_mask, float* normalized_masked_src, float* dst, int ch, HairInpaintInfo info);
void hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info, bool isGPU);