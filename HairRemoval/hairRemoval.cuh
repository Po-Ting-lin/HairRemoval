#pragma once
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <cufft.h>
#include <vector>
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

	void _hairDetection(cv::Mat& src);
	void _hairInpainting(cv::Mat& src, cv::Mat& dst);

	float* _initGaborFilterCube(HairDetectionInfo para);
	void _padDataClampToBorder(float* d_Dst, float* d_Src);
	void _padKernel(float* d_Dst, float* d_Src);
	void _modulateAndNormalize(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int padding);
	void _cubeReduction(float* d_Src, uchar* d_Dst);
	void _cleanIsolatedComponent(cv::Mat& src);
	void _makeHistogram(uchar* d_src, int* h_histogram);
	int _findOtsuThreshold(int* h_histogram);
	void _binarization(uchar* d_Src, int threshold);
	void _dilation(uchar* d_Src, bool* d_Dst);

	void _normalizeImage(float* dstMaskImage);
	void _pdeHeatDiffusion(float* d_normalized_masked_src, uchar* h_dst);
	void _pdeHeatDiffusionSmem(bool* d_normalized_mask, float* d_normalized_masked_src, float* h_dst);
};


