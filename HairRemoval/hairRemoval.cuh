#pragma once
#include <omp.h>
#include <cufft.h>
#include "utils.h"
#include "parameters.h"
#include "hairRemovalStruct.h"
#include "cuda_error.cuh"
#include "timer.cuh"
#include "hairRemovalEngine.cuh"



class HairRemoval {
public:
	HairRemoval(bool isGPU);
	void Process(cv::Mat& src, cv::Mat& dst);
private:
	bool _isGPU;

	void _hairDetection(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info);
	void _hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info);

	float* _initGaborFilterCube(HairDetectionInfo para);
	void _extractLChannel(cv::Mat& src, cv::Mat& dst);
	void _gaborFiltering(cv::Mat& src, cv::Mat& dst, HairDetectionInfo para);
	cv::Mat _getGaborFilter(float theta, HairDetectionInfo para);
	void _hairDetectionGPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info);
	void _hairDetectionCPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info);
	void _padDataClampToBorder(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX);
	void _padKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX);
	void _modulateAndNormalize(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int fftH, int fftW, int padding);
	void _cubeReduction(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth);
	void _cleanIsolatedComponent(cv::Mat& src, HairDetectionInfo para);

	void _normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage, HairInpaintInfo info);
	void _hairInpaintingGPU(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info);
	void _hairInpaintingCPU(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info);
	void _PDEHeatDiffusionCPU(float* normalized_mask, float* normalized_masked_src, float* dst, int ch, HairInpaintInfo info);
	void _convertToMatArrayFormat(float* srcImage, uchar* dstImage, HairInpaintInfo info);
};


