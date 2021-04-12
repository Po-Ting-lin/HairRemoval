#pragma once
#include "cuda_error.cuh"
#include "hairRemovalEngine.cuh"

struct EntropyThresholdDeviceInfo {
	int fullWidth;
	int targetWidth;
	int multipleWidth;
	int startThreshold;
	int sumMatrixSize;
	dim3* preSumBlock;
	dim3* preSumGrid;
	int preSumSmemSize;
	dim3* sumBlock;
	dim3* sumGrid;
	int sumSmemSize;
	dim3* sumSumBlock;
	dim3* sumSumGrid;
	int sumSumSmemSize;

	EntropyThresholdDeviceInfo(int full_width) :
		targetWidth(0),
		multipleWidth(0),
		fullWidth(full_width),
		startThreshold(TILE_DIM - 1),
		preSumBlock(new dim3(TILE_DIM, TILE_DIM)),
		preSumGrid(nullptr),
		sumBlock(new dim3(TILE_DIM, TILE_DIM)),
		sumGrid(nullptr),
		sumSumBlock(new dim3(iDivUp(full_width, TILE_DIM)* iDivUp(full_width, TILE_DIM))),
		sumSumGrid(new dim3(1)),
		preSumSmemSize(0),
		sumSmemSize(TILE_DIM* TILE_DIM * sizeof(float)),
		sumSumSmemSize(iDivUp(full_width, TILE_DIM)* iDivUp(full_width, TILE_DIM) * sizeof(float)),
		sumMatrixSize(iDivUp(full_width, TILE_DIM)* iDivUp(full_width, TILE_DIM)) { };
};

class EntropyBasedThreshold
{
public:
	EntropyBasedThreshold(bool isGPU);
	int getThreshold(cv::Mat& src);

private:
	bool _isGPU;
	uchar* _data;
	float* _glcm;
	int _width;
	int _height;

	void _getGrayLevelCoOccurrenceMatrix();
	int _entropyThesholdingCPU();
	int _entropyThesholdingGPU();

	void _entropyCPU(float* h_data, float* h_e, int width, bool reversed);
	void _getAreaArray(float* d_data, float* d_pA, cudaStream_t* stream, EntropyThresholdDeviceInfo& info);
	void _sumMatrixStream(float* d_buf, float* d_arr, float* d_sum_matrix, EntropyThresholdDeviceInfo& info, int threshold, cudaStream_t stream);
	void _getMeanArray(float* d_data, float* d_pA, float* d_mA, bool reversed, cudaStream_t* stream, EntropyThresholdDeviceInfo& info);
	void _getEntropyArray(float* d_data, float* d_mA, float* d_eA, bool reversed, cudaStream_t* stream, EntropyThresholdDeviceInfo& info);
	void _reversedData(float* d_data, float* d_reversed_data, int full_width);

#if ISAVX 
	inline void _loadPixel(__m256& x, __m256& p, int width, int r, int c, int cBoundary);
#endif
};

