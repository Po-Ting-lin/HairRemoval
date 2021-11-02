#pragma once
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "hairRemovalStruct.h"
#include "parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

inline __device__ void mulAndScaleModified(const fComplex& a, const fComplex& b, const float& c, fComplex& d) {
    fComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
    d = t;
}

// hair detecting
__global__ void extractLChannelKernel(uchar* src, float* dst, int nx, int ny, int nz);
__global__ void padDataClampToBorderKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX);
__global__ void padKernelKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX);
__global__ void modulateAndNormalizeKernel(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int dataSize, float c);
__global__ void cubeReductionKernel(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth);

// hair inpainting
#define BlockDim_x 8
#define BlockDim_y 8
#define Step 8
#define PadStep 1

__global__ void pdeHeatDiffusionSMEMKernel(float* mask, float* src, float* dst, int width, int height);
__global__ void pdeHeatDiffusionKernel(float* mask, float* src, float* tempSrc, int width, int height, int ch);
