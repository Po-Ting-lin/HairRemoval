#pragma once
#include <omp.h>
#include "utils.h"
#include "parameters.h"


class EntropyBasedThreshold 
{
public:
	EntropyBasedThreshold(cv::Mat& src, bool isGPU);
	int getThreshold();

private:
	bool _isGPU;
	uchar* _data;
	float* _glcm;
	int _width;
	int _height;

	void _getGrayLevelCoOccurrenceMatrix();
	int _entropyThesholding();
#if ISAVX 
	inline void _loadPixel(__m256& x, __m256& p, int width, int r, int c, int cBoundary);
#endif
};