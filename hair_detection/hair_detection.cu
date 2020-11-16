#include "hair_detection_kernel.cuh"
#include "utils.h"
#include <cuFFT.h>

#define TILE_DIM 32
#define BLOCK_DIM 8
#define EPSILON 1e-6

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

// very inefficient, frequently load from DRAM 
__global__ void entropyCalculationKernel(float* glcmA, float* glcmC, float* eA, float* eC, int dynamic) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < dynamic) {
        float pA = 0.0f;
        float meanA = 0.0f;
        float entropyA = 0.0f;
        int tA = gid;
        // pA
        for (int r = 0; r < tA + 1; r++) {
            for (int c = 0; c < tA + 1; c++) {
                pA += *(glcmA + r * dynamic + c);
            }
        }

        // meanA
        for (int r = 0; r < tA + 1; r++) {
            for (int c = 0; c < tA + 1; c++) {
                meanA += ((float)r) * ((float)c) * (*(glcmA + r * dynamic + c));
            }
        }
        meanA /= pA;

        // entropyA
        for (int r = 0; r < tA + 1; r++) {
            for (int c = 0; c < tA + 1; c++) {
                float raw = (*(glcmA + r * dynamic + c));
                entropyA += ((float)r) * ((float)c) * raw * log2f((((float)r) * ((float)c) + EPSILON) / (meanA + EPSILON));
                entropyA += meanA * raw * log2f(meanA / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON);
            }
        }

        eA[tA] = entropyA;
    }
    else {
        float pC = 0.0;
        float meanC = 0.0;
        float entropyC = 0.0f;
        int tC = gid - dynamic;
        // pC
        for (int r = tC + 1; r < dynamic; r++) {
            for (int c = tC + 1; c < dynamic; c++) {
                pC += *(glcmC + r * dynamic + c);
            }
        }

        // meanC
        for (int r = tC + 1; r < dynamic; r++) {
            for (int c = tC + 1; c < dynamic; c++) {
                meanC += ((float)r) * ((float)c) * (*(glcmC + r * dynamic + c));
            }
        }
        meanC /= pC;

        // entropyC
        for (int r = tC + 1; r < dynamic; r++) {
            for (int c = tC + 1; c < dynamic; c++) {
                float raw = (*(glcmC + r * dynamic + c));
                entropyC += ((float)r) * ((float)c) * raw * log2f((((float)r) * ((float)c) + EPSILON) / (meanC + EPSILON));
                entropyC += meanC * raw * log2f(meanC / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON);
            }
        }

        eC[tC] = entropyC;
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
    const int NUM_STREAMS = 6;
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



int entropyThesholdingGPU(cv::Mat& glcm) {
    int dynamic_range = 256;
    float
        * h_eA,
        * h_eC;
    float
        * d_glcmA,
        * d_glcmC,
        * d_eA,
        * d_eC;
    float* src_ptr = (float*)glcm.data;

    h_eA = (float*)malloc(dynamic_range * sizeof(float*));
    h_eC = (float*)malloc(dynamic_range * sizeof(float*));
    gpuErrorCheck(cudaMalloc((void**)&d_glcmA, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_glcmC, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_glcmA, src_ptr, dynamic_range * dynamic_range * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_glcmC, d_glcmA, dynamic_range * dynamic_range * sizeof(float), cudaMemcpyDeviceToDevice));

    dim3 block(TILE_DIM);
    dim3 grid(iDivUp(dynamic_range*2, TILE_DIM)); // 512 threads

    entropyCalculationKernel << <grid, block >> > (d_glcmA, d_glcmC, d_eA, d_eC, dynamic_range);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaMemcpy(h_eA, d_eA, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(h_eC, d_eC, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));

    int bestT = 0;
    float minLCM = FLT_MAX;

    for (int t = 0; t < dynamic_range; t++) {
        if (minLCM > h_eA[t] + h_eC[t]) {
            bestT = t;
            minLCM = h_eA[t] + h_eC[t];
        }
    }

    gpuErrorCheck(cudaFree(d_glcmA));
    gpuErrorCheck(cudaFree(d_glcmC));
    gpuErrorCheck(cudaFree(d_eA));
    gpuErrorCheck(cudaFree(d_eC));
    free(h_eA);
    free(h_eC);
    gpuErrorCheck(cudaDeviceReset());

    return bestT;
}

void TestSumMatrix() {
    float
        * h_data,
        * h_test,
        * h_sum_matrix;
    float
        * d_data,
        * d_sum_matrix;

    float ref = 0;
    int raw_width = 32;
    int width = GetClosedWidth(raw_width);

    dim3 pre_sum_block(TILE_DIM, TILE_DIM);
    dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
    dim3 sum_block(TILE_DIM, TILE_DIM);
    dim3 sum_grid(width / TILE_DIM, width / TILE_DIM);
    
    h_data = (float*)malloc(raw_width * raw_width * sizeof(float*));
    h_test = (float*)malloc(raw_width * raw_width * sizeof(float*));
    h_sum_matrix = (float*)malloc(sum_grid.x * sum_grid.x * sizeof(float*));

    for (int i = 0; i < raw_width * raw_width; i++) {
        h_data[i] = ((float)i / 100.0f);
        ref += ((float)i / 100.0f);
    }
    //Display2DArray(h_data, raw_width, raw_width);
    //std::cout << std::endl;
    
    gpuErrorCheck(cudaMalloc((void**)&d_data, raw_width * raw_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_grid.x * sum_grid.x * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_data, h_data, raw_width * raw_width * sizeof(float), cudaMemcpyHostToDevice));

    // presum
    if (raw_width != width){
        PreSumXMatrix << <pre_sum_grid, pre_sum_block >> > (d_data, raw_width, raw_width, width);
        gpuErrorCheck(cudaDeviceSynchronize());
        PreSumYMatrix << <pre_sum_grid, pre_sum_block >> > (d_data, raw_width, raw_width, width);
        gpuErrorCheck(cudaDeviceSynchronize());
    }
    SumMatirx << <sum_grid, sum_block >> > (d_data, raw_width, width, d_sum_matrix);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaMemcpy(h_sum_matrix, d_sum_matrix, sum_grid.x * sum_grid.x * sizeof(float), cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (int i = 0; i < sum_grid.x * sum_grid.x; i++) {
        result += h_sum_matrix[i];
    }

    printf("ref: %f, gpu: %f\n", ref, result);

    gpuErrorCheck(cudaFree(d_data));
    gpuErrorCheck(cudaFree(d_sum_matrix));
    free(h_data);
    free(h_sum_matrix);
    gpuErrorCheck(cudaDeviceReset());
    return;
}

int GetClosedWidth(int width){
    int number = (int)log2(width);
    return pow(2, number);
}

__global__ void PreSumXMatrix(float* src, int nx, int raw_width, int new_width) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int diff = raw_width - new_width;
    if (x < raw_width && y < raw_width) {
        if ((x < new_width) && (x >= new_width - diff)) {
            src[y * nx + x] += src[y * nx + x + diff];
        }
    }
}

__global__ void PreSumYMatrix(float* src, int nx, int raw_width, int new_width) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int diff = raw_width - new_width;
    if (x < raw_width && y < raw_width) {
        if ((x < new_width && y < new_width) && (y >= new_width - diff)) {
            src[y * nx + x] += src[(y + diff) * nx + x];
        }
    }
}

__global__ void SumMatirx(float* src, int nx, int tx, float* sum) {
    __shared__ float smem[TILE_DIM * TILE_DIM];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < tx && y < tx) {
        smem[threadIdx.y * blockDim.x + threadIdx.x] = src[y * nx + x];
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

__global__ void SumSumAMatrix(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold) {
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

__global__ void SumSumMMatrix(float* sum_matrix, float *d_pA, float* d_mA, int sum_matrix_size, int threshold) {
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
        //printf("threshold: %d, output: %f\n", threshold, (float)(smem[0]));
        d_mA[threshold] = (float)(smem[0] / d_pA[threshold]);
    }
}

__global__ void MultiplyRC(float* d_data_rc, float* d_data, int nx) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < nx && y < nx) {
        d_data_rc[y * nx + x] = d_data[y * nx + x] * x * y;
    }
}

__global__ void ComputeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < nx && y < nx) {
        float meanA = d_mA[threshold];
        float p = d_data[y * nx + x];
        float value = p * x * y * log2f(((float)x * y + EPSILON) / (meanA + EPSILON));
        value += meanA * p * log2f(meanA / ((float)x + EPSILON) / ((float)y + EPSILON) + EPSILON);
        d_data_computed[y * nx + x] = value;
    }
}

void Test666() {
    float
        * h_data,
        * h_pA,
        * h_mA,
        * h_eA;

    float
        * d_data,
        * d_pA,
        * d_mA,
        * d_eA;

    int raw_width = 256;

    // init h_data
    h_data = (float*)malloc(raw_width * raw_width * sizeof(float*));
    h_pA = (float*)malloc(raw_width * sizeof(float*));
    h_mA = (float*)malloc(raw_width * sizeof(float*));
    h_eA = (float*)malloc(raw_width * sizeof(float*));
    for (int i = 0; i < raw_width * raw_width; i++) {
        h_data[i] = getRand() / 1000.0f;
    }
    //Display2DArray(h_data, raw_width, raw_width);
    //std::cout << std::endl;
    gpuErrorCheck(cudaMalloc((void**)&d_data, raw_width * raw_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pA, raw_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mA, raw_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eA, raw_width * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_data, h_data, raw_width * raw_width * sizeof(float), cudaMemcpyHostToDevice));

    GetPAArray(d_data, 256, d_pA);
    //gpuErrorCheck(cudaMemcpy(h_pA, d_pA, raw_width * sizeof(float), cudaMemcpyDeviceToHost));

    GetMAArray(d_data, 256, d_pA, d_mA);
    //gpuErrorCheck(cudaMemcpy(h_mA, d_mA, raw_width * sizeof(float), cudaMemcpyDeviceToHost));

    GetEAArray(d_data, 256, d_mA, d_eA);
    //gpuErrorCheck(cudaMemcpy(h_eA, d_eA, raw_width * sizeof(float), cudaMemcpyDeviceToHost));

    //// check pA
    //float* pA_ref;
    //pA_ref = (float*)malloc(raw_width * sizeof(float*));
    //for (int t = 0; t < raw_width; t++) {
    //    float pa_sum = 0.0f;
    //    for (int x = 0; x < t + 1; x++) {
    //        for (int y = 0; y < t + 1; y++) {
    //            pa_sum += h_data[y * raw_width + x];
    //        }
    //    }
    //    pA_ref[t] = pa_sum;
    //}
    //for (int i = 0; i < raw_width; i++) {
    //    //printf("ref: %f, gpu: %f\n", pA_ref[i], h_pA[i]);
    //}
    //
    //// check mA
    //float* mA_ref;
    //mA_ref = (float*)malloc(raw_width * sizeof(float*));
    //for (int t = 0; t < raw_width; t++) {
    //    float ma_sum = 0.0f;
    //    for (int x = 0; x < t + 1; x++) {
    //        for (int y = 0; y < t + 1; y++) {
    //            ma_sum += h_data[y * raw_width + x] * x * y;
    //        }
    //    }
    //    if (pA_ref[t] != 0.0f) {
    //        mA_ref[t] = ma_sum / pA_ref[t];
    //    }
    //    else {
    //        mA_ref[t] = 0.0f;
    //    }
    //}
    //for (int i = 0; i < raw_width; i++) {
    //    //printf("i: %d, ref: %f, gpu: %f\n", i, mA_ref[i], h_mA[i]);
    //}

    //// check eA
    //float* eA_ref;
    //eA_ref = (float*)malloc(raw_width * sizeof(float*));
    //for (int t = 0; t < raw_width; t++) {
    //    float ea_sum = 0.0f;
    //    float meanA = mA_ref[t];
    //    for (int x = 0; x < t + 1; x++) {
    //        for (int y = 0; y < t + 1; y++) {
    //            float p = h_data[y * raw_width + x];
    //            ea_sum += ((float)x) * ((float)y) * p * log2((((float)x) * ((float)y) + EPSILON) / (meanA + EPSILON));
    //            ea_sum += meanA * p * log2(meanA / (((float)x) + EPSILON) / (((float)y) + EPSILON) + EPSILON);
    //        }
    //    }
    //    eA_ref[t] = ea_sum;
    //}
    //for (int i = 0; i < raw_width; i++) {
    //    //printf("i: %d, ref: %f, gpu: %f\n", i, eA_ref[i], h_eA[i]);
    //}

    gpuErrorCheck(cudaFree(d_data));
    free(h_data);
    gpuErrorCheck(cudaDeviceReset());
    return;
}

int entropyThesholdingGPU2(cv::Mat& glcm) {
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

    h_data = (float*)glcm.data;
    h_reversed_data = (float*)malloc(dynamic_range * dynamic_range * sizeof(float*));


    h_eA = (float*)malloc(dynamic_range * sizeof(float*));
    h_eC = (float*)malloc(dynamic_range * sizeof(float*));
    h_AC = (float*)malloc(dynamic_range * sizeof(float*));

    int j = dynamic_range * dynamic_range - 1;
    for (int i = 0; i < dynamic_range * dynamic_range; i++, j--) {
        h_reversed_data[j] = h_data[i];
    }

    gpuErrorCheck(cudaMalloc((void**)&d_data, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_reversed_data, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_data, h_data, dynamic_range * dynamic_range * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_reversed_data, h_reversed_data, dynamic_range * dynamic_range * sizeof(float), cudaMemcpyHostToDevice));

    GetPAArray(d_data, 256, d_pA);
    GetMAArray(d_data, 256, d_pA, d_mA);
    GetEAArray(d_data, 256, d_mA, d_eA);

    GetPAArray(d_reversed_data, 256, d_pC);
    GetMAArray(d_reversed_data, 256, d_pC, d_mC);
    GetEAArray(d_reversed_data, 256, d_mC, d_eC);

    gpuErrorCheck(cudaMemcpy(h_eA, d_eA, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(h_eC, d_eC, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));

    int jj = dynamic_range - 1;
    for (int i = 0; i < dynamic_range; i++, jj--) {
        h_AC[i] = h_eA[i] + h_eC[jj];
        printf("i: %d, A:%f, C: %f, AC: %f\n", i, h_eA[i], h_eC[jj], h_AC[i]);
    }

    gpuErrorCheck(cudaFree(d_data));
    gpuErrorCheck(cudaFree(d_pA));
    gpuErrorCheck(cudaFree(d_mA));
    gpuErrorCheck(cudaFree(d_eA));
    free(h_eA);
    free(h_eC);
    free(h_AC);
    gpuErrorCheck(cudaDeviceReset());
    return 0;
}

void GetPAArray(float* d_data, int full_width, float* d_pA) {
    float* d_buf;
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * sizeof(float)));

    // from 32 to 255
    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int multiple_width = GetClosedWidth(raw_width);

        // refresh d_buf
        gpuErrorCheck(cudaMemcpy(d_buf, d_data, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice));

        dim3 pre_sum_block(TILE_DIM, TILE_DIM);
        dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
        dim3 sum_block(TILE_DIM, TILE_DIM);
        dim3 sum_grid(multiple_width / TILE_DIM, multiple_width / TILE_DIM);

        float* d_sum_matrix;
        gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_grid.x * sum_grid.x * sizeof(float)));
        gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_grid.x * sum_grid.x * sizeof(float)));

        if (raw_width != multiple_width) {
            PreSumXMatrix << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
            gpuErrorCheck(cudaDeviceSynchronize());
            PreSumYMatrix << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
            gpuErrorCheck(cudaDeviceSynchronize());
        }
        SumMatirx << <sum_grid, sum_block >> > (d_buf, full_width, multiple_width, d_sum_matrix);
        gpuErrorCheck(cudaDeviceSynchronize());

        SumSumAMatrix << <1, 2 * TILE_DIM >> > (d_sum_matrix, d_pA, sum_grid.x * sum_grid.x, i); // launch 64 threads to init smem
        gpuErrorCheck(cudaDeviceSynchronize());

        // free
        gpuErrorCheck(cudaFree(d_sum_matrix));
    }

    gpuErrorCheck(cudaFree(d_buf));
}

void GetMAArray(float* d_data, int full_width, float* d_pA, float* d_mA) {
    float
        * d_buf,
        * d_data_rc;
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_data_rc, full_width * full_width * sizeof(float)));
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
    MultiplyRC << <rc_grid, rc_block >> > (d_data_rc, d_data, full_width);
    gpuErrorCheck(cudaDeviceSynchronize());

    // from 32 to 255
    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int multiple_width = GetClosedWidth(raw_width);

        // refresh d_buf
        gpuErrorCheck(cudaMemcpy(d_buf, d_data_rc, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice));

        dim3 pre_sum_block(TILE_DIM, TILE_DIM);
        dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
        dim3 sum_block(TILE_DIM, TILE_DIM);
        dim3 sum_grid(multiple_width / TILE_DIM, multiple_width / TILE_DIM);

        float* d_sum_matrix;
        gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_grid.x * sum_grid.x * sizeof(float)));
        gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_grid.x * sum_grid.x * sizeof(float)));

        if (raw_width != multiple_width) {
            PreSumXMatrix << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
            gpuErrorCheck(cudaDeviceSynchronize());
            PreSumYMatrix << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
            gpuErrorCheck(cudaDeviceSynchronize());
        }
        SumMatirx << <sum_grid, sum_block >> > (d_buf, full_width, multiple_width, d_sum_matrix);
        gpuErrorCheck(cudaDeviceSynchronize());
        SumSumMMatrix << <1, 2 * TILE_DIM >> > (d_sum_matrix, d_pA, d_mA, sum_grid.x * sum_grid.x, i);
        gpuErrorCheck(cudaDeviceSynchronize());

        // free
        gpuErrorCheck(cudaFree(d_sum_matrix));
    }

    gpuErrorCheck(cudaFree(d_buf));
}

void GetEAArray(float* d_data, int full_width, float* d_mA, float* d_eA) {
    float
        * d_buf,
        * d_data_computed;
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_data_computed, full_width * full_width * sizeof(float)));

    // from 32 to 255
    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int multiple_width = GetClosedWidth(raw_width);

        // 
        dim3 rc_block(TILE_DIM, TILE_DIM);
        dim3 rc_grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
        ComputeEntropyMatrixKernel << <rc_grid, rc_block >> > (d_data_computed, d_data, full_width, d_mA, i);
        gpuErrorCheck(cudaDeviceSynchronize());
        //

        // refresh d_buf
        gpuErrorCheck(cudaMemcpy(d_buf, d_data_computed, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice));

        dim3 pre_sum_block(TILE_DIM, TILE_DIM);
        dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
        dim3 sum_block(TILE_DIM, TILE_DIM);
        dim3 sum_grid(multiple_width / TILE_DIM, multiple_width / TILE_DIM);

        float* d_sum_matrix;
        gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_grid.x * sum_grid.x * sizeof(float)));
        gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_grid.x * sum_grid.x * sizeof(float)));

        if (raw_width != multiple_width) {
            PreSumXMatrix << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
            gpuErrorCheck(cudaDeviceSynchronize());
            PreSumYMatrix << <pre_sum_grid, pre_sum_block >> > (d_buf, full_width, raw_width, multiple_width);
            gpuErrorCheck(cudaDeviceSynchronize());
        }
        SumMatirx << <sum_grid, sum_block >> > (d_buf, full_width, multiple_width, d_sum_matrix);
        gpuErrorCheck(cudaDeviceSynchronize());

        SumSumAMatrix << <1, 2 * TILE_DIM >> > (d_sum_matrix, d_eA, sum_grid.x * sum_grid.x, i);
        gpuErrorCheck(cudaDeviceSynchronize());

        // free
        gpuErrorCheck(cudaFree(d_sum_matrix));
    }

    gpuErrorCheck(cudaFree(d_buf));
}