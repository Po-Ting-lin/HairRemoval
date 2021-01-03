#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "parameters.h"


struct BlockInfo {
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
};

inline int getClosedWidth(int width) {
    int number = (int)log2(width);
    return pow(2, number);
}

__global__ void preSumXMatrixKernel(float* src, int nx, int tx, int multiplex);
__global__ void preSumYMatrixKernel(float* src, int nx, int tx, int multiplex);
__global__ void sumMatirxKernel(float* src, int nx, int tx, float* sum);
__global__ void sumSumMatrixKernel(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold);
__global__ void computeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold, bool reversed);
__global__ void multiplyRCKernel(float* d_data_rc, float* d_data, int nx, bool reversed);
__global__ void dividePArrayKernel(float* d_p, float* d_m, int size);
__global__ void reversedDataKernel(float* d_data, float* d_reversed_data, int nx);

int entropyThesholdingGPU(cv::Mat& glcm);
BlockInfo getBlockInfo(int full_width);
void entropyCPU(float* h_data, float* h_e, int width, bool reversed);
void reversedData(float* d_data, float* d_reversed_data, int full_width);
void getAreaArray(float* d_data, float* d_pA, cudaStream_t* stream, BlockInfo& info);
void getMeanArray(float* d_data, float* d_pA, float* d_mA, bool reversed, cudaStream_t* stream, BlockInfo& info);
void getEntropyArray(float* d_data, float* d_mA, float* d_eA, bool reversed, cudaStream_t* stream, BlockInfo& info);
void sumMatrixStream(float* d_buf, float* d_arr, float* d_sum_matrix, BlockInfo& info, int threshold, cudaStream_t stream);
