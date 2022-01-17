#pragma once
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "hairRemovalStruct.h"
#include "parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ float constMinMax[2];

inline __device__ void mulAndScaleModified(const fComplex& a, const fComplex& b, const float& c, fComplex& d) {
    fComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
    d = t;
}

// hair detecting
__global__ void extractLChannelKernel(uchar* src, float* dst, uchar* dst2, int nx, int ny, int nz);
__global__ void padDataClampToBorderKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX);
__global__ void padKernelKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX);
__global__ void modulateAndNormalizeKernel(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int dataSize, float c);
__global__ void cubeReductionKernel(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth);
__global__ void binarizeKernel(uchar* d_Src, int width, int height, int threshold);
__global__ void NaiveDilationKernel(uchar* d_Src, bool* d_Dst, int width, int height);

// hair inpainting
__global__ void makeMaskSrcImageKernel(uchar* src, bool* mask, float* maskedSrc, float max, float min, int size);
__global__ void NotKernel(bool* mask, int size);
__global__ void pdeHeatDiffusionSMEMKernel(bool* mask, float* src, float* dst, int width, int height);
__global__ void pdeHeatDiffusionKernel(bool* mask, float* src, float* tempSrc, int width, int height);
__global__ void make8UDstKernel(float* src, uchar* dst, float maxR, float minR, float maxG, float minG, float maxB, float minB, int width, int height);