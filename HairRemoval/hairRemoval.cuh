#pragma once
#include <cufft.h>
#include "utils.h"
#include "parameters.h"
#include "hairRemovalStruct.h"
#include "cuda_error.cuh"
#include "timer.cuh"
#include "hairRemovalEngine.cuh"
#ifdef _OPENMP
#include <omp.h>
#endif

class HairRemoval {
public:
	HairRemoval(int width, int height, int channel);
	~HairRemoval();
	void Process(cv::Mat& src, cv::Mat& dst);
private:
	HairDetectionInfo _detectionInfo;
	HairInpaintInfo _inpaintInfo;
	cufftHandle _fftPlanFwd;
	cufftHandle _fftPlanInv;

	void _hairDetection(cv::Mat& src, cv::Mat& dst);
	void _hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst);

	float* _initGaborFilterCube(HairDetectionInfo para);
	void _padDataClampToBorder(float* d_Dst, float* d_Src);
	void _padKernel(float* d_Dst, float* d_Src);
	void _modulateAndNormalize(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int padding);
	void _cubeReduction(float* d_Src, uchar* d_Dst);
	void _cleanIsolatedComponent(cv::Mat& src);

	void _normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, bool* dstMask, float* dstMaskImage);
	void _pdeHeatDiffusion(bool* d_normalized_mask, float* d_normalized_masked_src, float* h_dst);
	void _pdeHeatDiffusionSmem(bool* d_normalized_mask, float* d_normalized_masked_src, float* h_dst);
	void _convertToMatArrayFormat(float* srcImage, uchar* dstImage);
};


