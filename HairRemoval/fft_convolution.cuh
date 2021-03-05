#pragma once
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include "parameters.h"
#include "utils.h"

typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
    float x;
    float y;
} fComplex;
#endif

inline __device__ void mulAndScale(
    fComplex& a,
    const fComplex& b,
    const float& c
) {
    fComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
    a = t;
}

inline __device__ void mulAndScaleModified(
    const fComplex& a,
    const fComplex& b,
    const float& c,
    fComplex& d
) {
    fComplex t = { c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y) };
    d = t;
}

__global__ void padKernelKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

__global__ void padDataClampToBorderKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

__global__ void modulateAndNormalizeKernel(
    fComplex* d_Dst,
    fComplex* d_DataSrc,
    fComplex* d_KernelSrc,
    int dataSize,
    float c
);

__global__ void cubeReductionKernel(
    float* d_Src,
    uchar* d_Dst, 
    int fftH, 
    int fftW, 
    int dataH, 
    int dataW, 
    int depth
);

void convolutionClampToBorderCPU(
    float* h_Result,
    float* h_Data,
    float* h_Kernel,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

void padKernel(
    float* d_PaddedKernel,
    float* d_Kernel,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

void padDataClampToBorder(
    float* d_PaddedData,
    float* d_Data,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
);

void modulateAndNormalize(
    fComplex* d_Dst,
    fComplex* d_DataSrc,
    fComplex* d_KernelSrc,
    int fftH,
    int fftW,
    int padding
);

void cubeReduction(
    float* d_Src,
    uchar* d_Dst,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int depth
);

float* initGaborFilterCube(HairDetectionInfo para);

