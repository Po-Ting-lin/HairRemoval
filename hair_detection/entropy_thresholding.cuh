#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include <opencv2/opencv.hpp>
#include "utils.h"

__global__ void PreSumXMatrixKernel(float* src, int nx, int tx, int multiplex);
__global__ void PreSumYMatrixKernel(float* src, int nx, int tx, int multiplex);
__global__ void SumMatirxKernel(float* src, int nx, int tx, float* sum);
__global__ void SumSumMatrixKernel(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold);
__global__ void ComputeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold, bool reversed);
__global__ void MultiplyRCKernel(float* d_data_rc, float* d_data, int nx, bool reversed);
__global__ void DividePArrayKernel(float* d_p, float* d_m, int size);
__global__ void ReversedDataKernel(float* d_data, float* d_reversed_data, int nx);


inline int GetClosedWidth(int width) {
    int number = (int)log2(width);
    return pow(2, number);
}

int entropyThesholdingGPU(cv::Mat& glcm);
void entropyCPU(float* h_data, float* h_e, int width, bool reversed);
void ReversedData(float* d_data, float* d_reversed_data, int full_width);
void GetPArrayStream(float* d_data, int full_width, float* d_pA, cudaStream_t* stream);
void GetMArrayStream(float* d_data, int full_width, float* d_pA, float* d_mA, bool reversed, cudaStream_t* stream);
void GetEArrayStream(float* d_data, int full_width, float* d_mA, float* d_eA, bool reversed, cudaStream_t* stream);
void SumMatrixStream(float* d_buf, float* d_arr, float* d_sum_matrix, int full_width, int raw_width, int multiple_width, int threshold, cudaStream_t stream);
