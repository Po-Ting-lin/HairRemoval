#pragma once
#include <iostream>
#include <stdio.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include <opencv2/opencv.hpp>
#include "filtering.cuh"
#include "utils.h"

__global__ void extractLChannelWithInstrinicFunction(uchar* src, float* dst, int nx, int ny, int nz);
__global__ void entropyCalculationKernel(float* glcmA, float* glcmC, float* eA, float* eC, int dynamic);

void getHairMask(cv::Mat& src, cv::Mat& dst, HairDetectionParameters para);
int entropyThesholdingGPU(cv::Mat& glcm);

__global__ void PreSumXMatrix(float* src, int nx, int tx, int multiplex);
__global__ void PreSumYMatrix(float* src, int nx, int tx, int multiplex);
__global__ void SumMatirx(float* src, int nx, int tx, float* sum);
__global__ void SumSumAMatrix(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold);
__global__ void SumSumMMatrix(float* sum_matrix, float* d_pA, float* d_mA, int sum_matrix_size, int threshold);
__global__ void ComputeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold);
void TestSumMatrix();
void Test666();
int entropyThesholdingGPU2(cv::Mat& glcm);
int GetClosedWidth(int width);
void GetPAArray(float* d_data, int full_width, float* d_pA);
void GetMAArray(float* d_data, int full_width, float* d_pA, float* d_mA);
void GetEAArray(float* d_data, int full_width, float* d_mA, float* d_eA);
