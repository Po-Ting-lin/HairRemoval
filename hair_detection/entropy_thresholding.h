#pragma once
#include <omp.h>
#include "utils.h"
#include "parameters.h"

class EntropyBasedThreshold 
{
public:
	EntropyBasedThreshold(cv::Mat& src);
	int Process();

private:
	uchar* _data;
	float* _glcm;
	int _width;
	int _height;

	void _getGrayLevelCoOccurrenceMatrix();
	int _entropyThesholding();
};