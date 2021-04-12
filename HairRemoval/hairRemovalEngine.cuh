#pragma once
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "hairRemovalStruct.h"
#include "parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ float d_dt[1];
__constant__ float d_center_w[1];

inline __device__ void mulAndScaleModified(const fComplex& a, const fComplex& b, const float& c, fComplex& d) {
    fComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
    d = t;
}

// hair detecting
__global__ void extractLChannelWithInstrinicFunction(uchar* src, float* dst, int nx, int ny, int nz);
__global__ void padDataClampToBorderKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX);
__global__ void padKernelKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX);
__global__ void modulateAndNormalizeKernel(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int dataSize, float c);
__global__ void cubeReductionKernel(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth);


// entropy thresholding
__global__ void preSumXMatrixKernel(float* src, int nx, int raw_width, int new_width);
__global__ void preSumYMatrixKernel(float* src, int nx, int raw_width, int new_width);
__global__ void sumMatirxKernel(float* src, int nx, int multiple_width, float* d_sum_matrix);
__global__ void sumSumMatrixKernel(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold);
__global__ void multiplyRCKernel(float* d_data_rc, float* d_data, int nx, bool reversed);
__global__ void dividePArrayKernel(float* d_p, float* d_m, int size);
__global__ void computeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold, bool reversed);
__global__ void reversedDataKernel(float* d_data, float* d_reversed_data, int nx);

// hair inpainting
__global__ void pdeHeatDiffusionSMEM(float* mask, float* src, float* tempSrc, int width, int height, int ch);
__global__ void pdeHeatDiffusionSMEM2(float* mask, float* src, float* tempSrc, int width, int height, int ch);
__global__ void pdeHeatDiffusion(float* mask, float* src, float* tempSrc, int width, int height, int ch);
