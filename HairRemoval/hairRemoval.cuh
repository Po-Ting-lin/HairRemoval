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
	HairRemoval(int width, int height, int channel, bool isGPU);
	void Process(cv::Mat& src, cv::Mat& dst);
	double GetExceedTime() {return _detectionInfo.ExceedTime;}
private:
	HairDetectionInfo _detectionInfo;
	HairInpaintInfo _inpaintInfo;

	void _hairDetection(cv::Mat& src, cv::Mat& dst);
	void _hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst);

	float* _initGaborFilterCube(HairDetectionInfo para);
	void _extractLChannel(cv::Mat& src, cv::Mat& dst);
	void _gaborFiltering(cv::Mat& src, cv::Mat& dst);
	cv::Mat _getGaborFilter(float theta);
	void _hairDetectionGPU(cv::Mat& src, cv::Mat& dst);
	void _hairDetectionCPU(cv::Mat& src, cv::Mat& dst);
	void _padDataClampToBorder(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX);
	void _padKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX);
	void _modulateAndNormalize(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int fftH, int fftW, int padding);
	void _cubeReduction(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth);
	void _cleanIsolatedComponent(cv::Mat& src);

	void _normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage);
	void _hairInpaintingGPU(float* normalized_mask, float* normalized_masked_src, float* dst);
	void _hairInpaintingCPU(float* normalized_mask, float* normalized_masked_src, float* dst);
	void _pdeHeatDiffusionGPU(float* d_normalized_mask, float* d_normalized_masked_src, float* dst);
	void _pdeHeatDiffusionSmemGPU(float* d_normalized_mask, float* d_normalized_masked_src, float* dst);
	void _pdeHeatDiffusionCPU(float* normalized_mask, float* normalized_masked_src, float* dst, int ch);
	void _convertToMatArrayFormat(float* srcImage, uchar* dstImage);
};


