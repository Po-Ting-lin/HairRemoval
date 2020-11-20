#include "hair_detection_kernel.cuh"
#include "utils.h"
#include <cuFFT.h>

#define TILE_DIM 32
#define BLOCK_DIM 8
#define EPSILON 1e-8
#define NUM_STREAMS 6
#define E_NUM_STREAMS 15

__global__ void extractLChannelWithInstrinicFunction(uchar* src, float* dst, int nx, int ny, int nz) {
    int x = threadIdx.x + TILE_DIM * blockIdx.x;
    int y = threadIdx.y + TILE_DIM * blockIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_DIM) {
        // take pixel from DRAM
        uchar R = *(src + ((y + i) * nx * nz) + (x * nz) + 0);
        uchar G = *(src + ((y + i) * nx * nz) + (x * nz) + 1);
        uchar B = *(src + ((y + i) * nx * nz) + (x * nz) + 2);

        // RGB to XYZ
        float r = fdividef((float)R, 255.0f);
        float g = fdividef((float)G, 255.0f);
        float b = fdividef((float)B, 255.0f);
        r = ((r > 0.04045f) ? __powf(fdividef(r + 0.055f, 1.055f), 2.4f) : fdividef(r, 12.92f)) * 100.0f;
        g = ((g > 0.04045f) ? __powf(fdividef(g + 0.055f, 1.055f), 2.4f) : fdividef(g, 12.92f)) * 100.0f;
        b = ((b > 0.04045f) ? __powf(fdividef(b + 0.055f, 1.055f), 2.4f) : fdividef(b, 12.92f)) * 100.0f;

        // XYZ to LAB
        float Y = fdividef(0.2126f * r + 0.7152f * g + 0.0722f * b, 100.0f);
        Y = (Y > 0.008856f) ? cbrtf(Y) : fmaf(7.787f, Y, 0.1379f);
        float L = fmaf(116.0f, Y, -16.0f) * 2.55f;

        //printf("r: %d g: %d b: %d --- L: %f\n", R, G, B, L);

        // set pixel to DRAM
        *(dst + (y + i) * nx + x) = L;
    }
}

void getHairMask(cv::Mat& src, cv::Mat& dst, HairDetectionParameters para) {
    auto t1 = std::chrono::system_clock::now();

    // declare 
    float
        * d_PaddedData,
        * d_Kernel,
        * d_PaddedKernel,
        * d_DepthResult;
    uchar
        * d_Result;

    fComplex
        * d_DataSpectrum,
        * d_KernelSpectrum,
        * d_TempSpectrum;

    cufftHandle
        fftPlanFwd,
        fftPlanInv;

    uchar* src_ptr = src.data;
    const int dataH = src.rows;
    const int dataW = src.cols;
    const int depth = para.numberOfFilter;
    const int fftH = snapTransformSize(dataH + para.kernelH - 1);
    const int fftW = snapTransformSize(dataW + para.kernelW - 1);
    const unsigned long src_size = src.cols * src.rows * src.channels();
    const unsigned long src_byte_size = src_size * sizeof(uchar);
    const unsigned long src_c_size = src.cols * src.rows;
    const unsigned long src_c_byte_size = src_c_size * sizeof(float);

    // host data
    cudaHostRegister(src_ptr, src_byte_size, cudaHostRegisterDefault);

    auto t2 = std::chrono::system_clock::now();

    // device data
    uchar* device_src_ptr;
    float* device_src_c_ptr;
    gpuErrorCheck(cudaMalloc((uchar**)&device_src_ptr, src_byte_size));
    gpuErrorCheck(cudaMalloc((float**)&device_src_c_ptr, src_c_byte_size));

    // stream
    int SRC_DATA_PER_STREAM = src_size / NUM_STREAMS;
    int DST_DATA_PER_STREAM = src_c_size / NUM_STREAMS;
    int SRC_BYTES_PER_STREAM = src_byte_size / NUM_STREAMS;
    int DST_BYTES_PER_STREAM = src_c_byte_size / NUM_STREAMS;

    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    int block_x_size = TILE_DIM;
    int block_y_size = BLOCK_DIM;
    int grid_x_size = (src.cols + TILE_DIM - 1) / TILE_DIM;
    int pruned_rows = src.rows / NUM_STREAMS;
    int grid_y_size = (pruned_rows + TILE_DIM - 1) / TILE_DIM;

    dim3 block(block_x_size, block_y_size);
    dim3 grid(grid_x_size, grid_y_size);

    int src_offset = 0;
    int dst_offset = 0;

    for (int i = 0; i < NUM_STREAMS; i++) {
        src_offset = i * SRC_DATA_PER_STREAM;
        dst_offset = i * DST_DATA_PER_STREAM;
        gpuErrorCheck(cudaMemcpyAsync(&device_src_ptr[src_offset], &src_ptr[src_offset], SRC_BYTES_PER_STREAM, cudaMemcpyHostToDevice, stream[i]));
        extractLChannelWithInstrinicFunction << < grid, block, 0, stream[i] >> > (&device_src_ptr[src_offset], &device_src_c_ptr[dst_offset], src.cols, pruned_rows, src.channels());
    }

    auto t3 = std::chrono::system_clock::now();

    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    cudaHostUnregister(src_ptr);
    gpuErrorCheck(cudaFree(device_src_ptr));

    auto t4 = std::chrono::system_clock::now();

    // init data
    
    float* h_kernels = GaborFilterCube(para);

    auto t5 = std::chrono::system_clock::now();

    // allocation
    gpuErrorCheck(cudaMalloc((void**)&d_Kernel, para.kernelH * para.kernelW * para.numberOfFilter * sizeof(float)));

    gpuErrorCheck(cudaMalloc((void**)&d_PaddedData, fftH * fftW * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_DepthResult, fftH * fftW * para.numberOfFilter * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_Result, dataH * dataW * sizeof(uchar)));

    gpuErrorCheck(cudaMalloc((void**)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    gpuErrorCheck(cudaMalloc((void**)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    gpuErrorCheck(cudaMalloc((void**)&d_TempSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    // H to D
    gpuErrorCheck(cudaMemcpy(d_Kernel, h_kernels, para.kernelH * para.kernelW * para.numberOfFilter * sizeof(float), cudaMemcpyHostToDevice));

    auto t6 = std::chrono::system_clock::now();

    // init value
    padDataClampToBorder(d_PaddedData, device_src_c_ptr, fftH, fftW, dataH, dataW, para.kernelH, para.kernelW, para.kernelY, para.kernelX);


    auto t7 = std::chrono::system_clock::now();

    // make a FFT plan
    gpuErrorCheck(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    gpuErrorCheck(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

    // FFT data
    gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    gpuErrorCheck(cudaDeviceSynchronize());

    auto t8 = std::chrono::system_clock::now();

    for (int i = 0; i < para.numberOfFilter; i++) {
        int kernel_offset = i * para.kernelH * para.kernelW;
        int data_offset = i * fftH * fftW;

        padKernel(d_PaddedKernel, &(d_Kernel[kernel_offset]), fftH, fftW, para.kernelH, para.kernelW, para.kernelY, para.kernelX);

        // FFT kernel
        gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
        gpuErrorCheck(cudaDeviceSynchronize());

        // mul
        modulateAndNormalize(d_TempSpectrum, d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
        gpuErrorCheck(cufftExecC2R(fftPlanInv, (cufftComplex*)d_TempSpectrum, (cufftReal*)(&d_DepthResult[data_offset])));
        gpuErrorCheck(cudaDeviceSynchronize());
    }

    auto t9 = std::chrono::system_clock::now();

    // debug // 
    //float* h_single;
    //h_single = (float*)malloc(fftH * fftW * sizeof(float));

    //for (int i = 0; i < para.numberOfFilter; i++) {
    //    int offs = i * fftH * fftW;
    //    gpuErrorCheck(cudaMemcpy(h_single, &d_DepthResult[offs], fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    //    displayImage(h_single, fftW, fftH, true);
    //}
    ///////////

    CubeReduction(d_DepthResult, d_Result, fftH, fftW, dataH, dataW, depth);

    auto t10 = std::chrono::system_clock::now();

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaMemcpy(dst.data, d_Result, dataH * dataW * sizeof(uchar), cudaMemcpyDeviceToHost));

    auto t11 = std::chrono::system_clock::now();

    // free
    gpuErrorCheck(cufftDestroy(fftPlanInv));
    gpuErrorCheck(cufftDestroy(fftPlanFwd));

    gpuErrorCheck(cudaFree(d_DataSpectrum));
    gpuErrorCheck(cudaFree(d_KernelSpectrum));
    gpuErrorCheck(cudaFree(d_PaddedData));
    gpuErrorCheck(cudaFree(d_PaddedKernel));
    gpuErrorCheck(cudaFree(d_TempSpectrum));
    gpuErrorCheck(cudaFree(device_src_c_ptr));
    gpuErrorCheck(cudaFree(d_Kernel));
    gpuErrorCheck(cudaFree(d_DepthResult));

    gpuErrorCheck(cudaDeviceReset());

    auto t12 = std::chrono::system_clock::now();

    //printTime(t1, t2, "source registering");
    //printTime(t2, t3, "c channel extracting");
    //printTime(t3, t4, "source unregistering");
    //printTime(t4, t5, "get gabor filter");
    //printTime(t5, t6, "cudaMalloc");
    //printTime(t6, t7, "padDataClampToBorder");
    //printTime(t7, t8, "source FFT");
    //printTime(t8, t9, "kernel FFT and mul");
    //printTime(t9, t10, "CubeReduction");
    //printTime(t10, t11, "D to H result");
    //printTime(t11, t12, "free");
}

int GetClosedWidth(int width){
    int number = (int)log2(width);
    return pow(2, number);
}

__global__ void PreSumXMatrixKernel(float* src, int nx, int raw_width, int new_width) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int diff = raw_width - new_width;
    if (x < raw_width && y < raw_width) {
        if ((x < new_width) && (x >= new_width - diff)) {
            src[y * nx + x] += src[y * nx + x + diff];
        }
    }
}

__global__ void PreSumYMatrixKernel(float* src, int nx, int raw_width, int new_width) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int diff = raw_width - new_width;
    if (x < raw_width && y < raw_width) {
        if ((x < new_width && y < new_width) && (y >= new_width - diff)) {
            src[y * nx + x] += src[(y + diff) * nx + x];
        }
    }
}

__global__ void SumMatirxKernel(float* src, int nx, int multiple_width, float* sum) {
    __shared__ float smem[TILE_DIM * TILE_DIM];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < multiple_width && y < multiple_width) {
        if (multiple_width < blockDim.x) {
            if (threadIdx.x >= multiple_width || threadIdx.y >= multiple_width) {
                smem[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }
            else {
                smem[threadIdx.y * blockDim.x + threadIdx.x] = src[y * nx + x];
            }
        }
        else {
            smem[threadIdx.y * blockDim.x + threadIdx.x] = src[y * nx + x];
        }
        
        __syncthreads();

        for (int offx = blockDim.x / 2; offx > 0; offx /= 2) {
            if (threadIdx.x < offx) {
                smem[threadIdx.y * blockDim.x + threadIdx.x] += smem[threadIdx.y * blockDim.x + threadIdx.x + offx];

                __syncthreads();

                if (threadIdx.y < offx) {
                    smem[threadIdx.y * blockDim.x + threadIdx.x] += smem[(threadIdx.y + offx) * blockDim.x + threadIdx.x];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sum[blockIdx.y * gridDim.x + blockIdx.x] = smem[threadIdx.y * blockDim.x + threadIdx.x];
            //printf("x: %d, y: %d -- %f\n", blockIdx.x, blockIdx.y, sum[blockIdx.y * gridDim.x + blockIdx.x]);
        }
    }
}

__global__ void SumSumMatrixKernel(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold) {
    __shared__ float smem[2 * TILE_DIM];
    int tid = threadIdx.x;

    // put the data in that block from DRAM to shared memory
    if (tid < sum_matrix_size) {
        smem[tid] = sum_matrix[tid];
    }
    else {
        smem[tid] = 0.0f;
    }
    __syncthreads(); // important

    // unrolling warp
    if (tid < 32) {
        volatile float* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32]; 
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        d_pA[threshold] = (float)(smem[0]);
    }
}

__global__ void DividePArrayKernel(float* d_p, float* d_m, int size) {
    int tid = threadIdx.x;
    if (tid < size) {
        d_m[tid] = __fdividef(d_m[tid], d_p[tid]);
    }
}

__global__ void MultiplyRCKernel(float* d_data_rc, float* d_data, int nx, bool reversed) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int cx = reversed ? nx - x - 1 : x;
    int cy = reversed ? nx - y - 1 : y;

    if (x < nx && y < nx) {
        d_data_rc[y * nx + x] = d_data[y * nx + x] * cx * cy;
    }
}

__global__ void ComputeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold, bool reversed) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int cx = reversed ? nx - x - 1 : x;
    int cy = reversed ? nx - y - 1 : y;

    if (x < nx && y < nx) {
        float meanA = d_mA[threshold];
        float p = d_data[y * nx + x];
        float value = p * cx * cy * log2f(((float)cx * cy + EPSILON) / (meanA + EPSILON));
        value += meanA * p * log2f(meanA / ((float)cx + EPSILON) / ((float)cy + EPSILON) + EPSILON);
        d_data_computed[y * nx + x] = value;
    }
}

__global__ void ReversedDataKernel(float* d_data, float* d_reversed_data, int nx) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int rx = nx - x - 1;
    int ry = nx - y - 1;

    if (x < nx && y < nx) {
        d_reversed_data[ry * nx + rx] = d_data[y * nx + x];
    }
}

int entropyThesholdingGPU(cv::Mat& glcm) {
    int dynamic_range = 256;
    float
        * h_data,
        * h_reversed_data,
        * h_eA,
        * h_eC,
        * h_AC;

    float
        * d_data,
        * d_reversed_data,
        * d_pA,
        * d_mA,
        * d_eA,
        * d_pC,
        * d_mC,
        * d_eC;

    auto t1 = std::chrono::system_clock::now();

    // host 
    h_data = (float*)glcm.data;
    h_reversed_data = (float*)malloc(dynamic_range * dynamic_range * sizeof(float*));
    h_eA = (float*)malloc(dynamic_range * sizeof(float*));
    h_eC = (float*)malloc(dynamic_range * sizeof(float*));
    h_AC = (float*)malloc(dynamic_range * sizeof(float*));

    for (int i = 0, int j = dynamic_range * dynamic_range - 1; i < dynamic_range * dynamic_range; i++, j--) {
        h_reversed_data[j] = h_data[i];
    }

    auto t2 = std::chrono::system_clock::now();

    // device
    gpuErrorCheck(cudaMalloc((void**)&d_data, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_reversed_data, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_data, h_data, dynamic_range * dynamic_range * sizeof(float), cudaMemcpyHostToDevice));

    auto t3 = std::chrono::system_clock::now();

    cudaStream_t stream[E_NUM_STREAMS];
    for (int i = 0; i < E_NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    entropyCPU(h_data, h_eA, dynamic_range, false);
    entropyCPU(h_reversed_data, h_eC, dynamic_range, true);

    auto t35 = std::chrono::system_clock::now();

    bool reversed = false;
    GetPArrayStream(d_data, dynamic_range, d_pA, stream);
    GetMArrayStream(d_data, dynamic_range, d_pA, d_mA, reversed, stream);
    GetEArrayStream(d_data, dynamic_range, d_mA, d_eA, reversed, stream);

    reversed = true;
    ReversedData(d_data, d_reversed_data, dynamic_range);
    GetPArrayStream(d_reversed_data, dynamic_range, d_pC, stream);
    GetMArrayStream(d_reversed_data, dynamic_range, d_pC, d_mC, reversed, stream);
    GetEArrayStream(d_reversed_data, dynamic_range, d_mC, d_eC, reversed, stream);
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < E_NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    auto t4 = std::chrono::system_clock::now();

    gpuErrorCheck(cudaMemcpy(h_eA, d_eA, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(h_eC, d_eC, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));

    auto t5 = std::chrono::system_clock::now();

    float min_value = FLT_MAX;
    int min_t = -1;
    for (int i = 0, int j = dynamic_range - 1; i < dynamic_range; i++, j--) {
        h_AC[i] = h_eA[i] + h_eC[j];
        //printf("i: %d, A:%f, C: %f, AC: %f\n", i, h_eA[i], h_eC[j], h_AC[i]);
        if (h_AC[i] < min_value) {
            min_t = i;
            min_value = h_AC[i];
        }
    }

    auto t6 = std::chrono::system_clock::now();

    //printTime(t1, t2, "make reverse data");
    //printTime(t2, t3, "H to D");
    //printTime(t3, t35, "CPU entropy");
    //printTime(t35, t4, "kerenl");
    //printTime(t4, t5, "D to H");
    //printTime(t5, t6, "conbine");

    printf("min threshold : %d\n", min_t);

    gpuErrorCheck(cudaFree(d_data));
    gpuErrorCheck(cudaFree(d_reversed_data));
    gpuErrorCheck(cudaFree(d_pA));
    gpuErrorCheck(cudaFree(d_mA));
    gpuErrorCheck(cudaFree(d_eA));
    gpuErrorCheck(cudaFree(d_pC));
    gpuErrorCheck(cudaFree(d_mC));
    gpuErrorCheck(cudaFree(d_eC));
    free(h_reversed_data);
    free(h_eA);
    free(h_eC);
    free(h_AC);
    gpuErrorCheck(cudaDeviceReset());
    return min_t;
}

void ReversedData(float* d_data, float* d_reversed_data, int full_width) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
    ReversedDataKernel << <grid, block >> > (d_data, d_reversed_data, full_width);
    gpuErrorCheck(cudaDeviceSynchronize());
}

void GetPArrayStream(float* d_data, int full_width, float* d_pA, cudaStream_t* stream) {
    float* d_buf, * d_sum_matrix;
    int sum_matirx_size = 2 * TILE_DIM;
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * full_width * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * full_width * sizeof(float)));

    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int idx = i % E_NUM_STREAMS;
        int buf_offset = i * full_width * full_width;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = GetClosedWidth(raw_width);
        gpuErrorCheck(cudaMemcpyAsync(&d_buf[buf_offset], d_data, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice, stream[idx]));
        SumMatrixStream(&d_buf[buf_offset], d_pA, &d_sum_matrix[sum_matrix_offset], full_width, raw_width, multiple_width, i, stream[idx]);
    }

    gpuErrorCheck(cudaStreamSynchronize(*stream));
    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void GetMArrayStream(float* d_data, int full_width, float* d_pA, float* d_mA, bool reversed, cudaStream_t* stream) {
    float * d_buf, * d_data_rc, * d_sum_matrix;
    int sum_matirx_size = 2 * TILE_DIM;
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_data_rc, full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * full_width * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * full_width * sizeof(float)));

    // r * c * element
    MultiplyRCKernel << <rc_grid, rc_block >> > (d_data_rc, d_data, full_width, reversed);
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int idx = i % E_NUM_STREAMS;
        int buf_offset = i * full_width * full_width;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = GetClosedWidth(raw_width);
        gpuErrorCheck(cudaMemcpyAsync(&d_buf[buf_offset], d_data_rc, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice, stream[idx]));
        SumMatrixStream(&d_buf[buf_offset], d_mA, &d_sum_matrix[sum_matrix_offset], full_width, raw_width, multiple_width, i, stream[idx]);
    }
    gpuErrorCheck(cudaStreamSynchronize(*stream));

    // divide area
    DividePArrayKernel << <1, full_width >> > (d_pA, d_mA, full_width);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_data_rc));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void GetEArrayStream(float* d_data, int full_width, float* d_mA, float* d_eA, bool reversed, cudaStream_t* stream) {
    float * d_buf, * d_sum_matrix;
    int sum_matirx_size = 2 * TILE_DIM;
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));

    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * full_width * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * full_width * sizeof(float)));

    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int idx = i % E_NUM_STREAMS;
        int buf_offset = i * full_width * full_width;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = GetClosedWidth(raw_width);
        ComputeEntropyMatrixKernel << <rc_grid, rc_block, 0, stream[idx] >> > (&d_buf[buf_offset], d_data, full_width, d_mA, i, reversed);
        SumMatrixStream(&d_buf[buf_offset], d_eA, &d_sum_matrix[sum_matrix_offset], full_width, raw_width, multiple_width, i, stream[idx]);
    }
    gpuErrorCheck(cudaStreamSynchronize(*stream));

    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void SumMatrixStream(float* d_buf, float* d_arr, float* d_sum_matrix, int full_width, int raw_width, int multiple_width, int threshold, cudaStream_t stream) {
    dim3 pre_sum_block(TILE_DIM, TILE_DIM);
    dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
    dim3 sum_block(TILE_DIM, TILE_DIM);
    dim3 sum_grid(iDivUp(multiple_width, TILE_DIM), iDivUp(multiple_width, TILE_DIM));

    if (raw_width != multiple_width) {
        PreSumXMatrixKernel << <pre_sum_grid, pre_sum_block, 0, stream >> > (d_buf, full_width, raw_width, multiple_width);
        PreSumYMatrixKernel << <pre_sum_grid, pre_sum_block, 0, stream >> > (d_buf, full_width, raw_width, multiple_width);
    }
    SumMatirxKernel << <sum_grid, sum_block, TILE_DIM * TILE_DIM * sizeof(float), stream >> > (d_buf, full_width, multiple_width, d_sum_matrix);
    SumSumMatrixKernel << <1, 2 * TILE_DIM, 2 * TILE_DIM * sizeof(float), stream >> > (d_sum_matrix, d_arr, sum_grid.x * sum_grid.x, threshold);
}

void SumMatrix(float* d_buf, float* d_arr, int full_width, int raw_width, int multiple_width, int threshold) {
    dim3 pre_sum_block(TILE_DIM, TILE_DIM);
    dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
    dim3 sum_block(TILE_DIM, TILE_DIM);
    dim3 sum_grid(iDivUp(multiple_width, TILE_DIM), iDivUp(multiple_width, TILE_DIM));

    float* d_sum_matrix;
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_grid.x * sum_grid.x * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_grid.x * sum_grid.x * sizeof(float)));

    if (raw_width != multiple_width) {
        PreSumXMatrixKernel << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
        gpuErrorCheck(cudaDeviceSynchronize());
        PreSumYMatrixKernel << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
        gpuErrorCheck(cudaDeviceSynchronize());
    }
    SumMatirxKernel << <sum_grid, sum_block >> > (d_buf, full_width, multiple_width, d_sum_matrix);
    gpuErrorCheck(cudaDeviceSynchronize());

    SumSumMatrixKernel << <1, 2 * TILE_DIM >> > (d_sum_matrix, d_arr, sum_grid.x * sum_grid.x, threshold);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void entropyCPU(float* h_data, float* h_e, int width, bool reversed) {
#pragma omp parallel for
    for (int threshold = 0; threshold < TILE_DIM - 1; threshold++) {
        const int cols = width;
        float p = 0.0f;
        float mean = 0.0f;
        float entropy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;

        // pA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                p += h_data[r * cols + c];
            }
        }

        // meanA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                cx = reversed ? width - c - 1 : c;
                cy = reversed ? width - r - 1 : r;
                mean += cx * cy * h_data[r * cols + c];
            }
        }
        mean /= p;

        // entropyA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                cx = reversed ? width - c - 1 : c;
                cy = reversed ? width - r - 1 : r;
                entropy += cy * cx * h_data[r * cols + c] * log2((cy * cx + EPSILON) / (mean + EPSILON));
                entropy += mean * h_data[r * cols + c] * log2(mean / (cy + EPSILON) / (cx + EPSILON) + EPSILON);
            }
        }
        h_e[threshold] = entropy;
    }
}