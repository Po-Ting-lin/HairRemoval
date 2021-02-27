#pragma once
#include "utils.h"
#include "parameters.h"
#include "cuda_error.cuh"
#include "hair_inpainting_CPU.h"


__global__ void PDEHeatDiffusion(float* mask, float* src, float* tempSrc, int width, int height, int ch);
__global__ void PDEHeatDiffusionSMEM(float* mask, float* src, float* tempSrc, int width, int height);

void hairInpaintingGPU(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info);
