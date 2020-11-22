#include "hair_detection_kernel.cuh"
#include "utils.h"
#include <cuFFT.h>

#define TILE_DIM 32
#define BLOCK_DIM 8
#define EPSILON 1e-8
#define NUM_STREAMS 6
#define TIMER false
#define DEBUG false

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

        // set pixel to DRAM
        *(dst + (y + i) * nx + x) = L;
    }
}

void getHairMask(cv::Mat& src, cv::Mat& dst, HairDetectionParameters para) {

#if TIMER
    auto t1 = std::chrono::system_clock::now();
#endif

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

#if TIMER
    auto t2 = std::chrono::system_clock::now();
#endif

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

#if TIMER
    auto t3 = std::chrono::system_clock::now();
#endif

    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    cudaHostUnregister(src_ptr);
    gpuErrorCheck(cudaFree(device_src_ptr));

#if TIMER
    auto t4 = std::chrono::system_clock::now();
#endif

    // init data
    float* h_kernels = gaborFilterCube(para);

#if TIMER
    auto t5 = std::chrono::system_clock::now();
#endif

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

#if TIMER
    auto t6 = std::chrono::system_clock::now();
#endif

    // init value
    padDataClampToBorder(d_PaddedData, device_src_c_ptr, fftH, fftW, dataH, dataW, para.kernelH, para.kernelW, para.kernelY, para.kernelX);

#if TIMER
    auto t7 = std::chrono::system_clock::now();
#endif

    // make a FFT plan
    gpuErrorCheck(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    gpuErrorCheck(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

    // FFT data
    gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    gpuErrorCheck(cudaDeviceSynchronize());

#if TIMER
    auto t8 = std::chrono::system_clock::now();
#endif

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

#if TIMER
    auto t9 = std::chrono::system_clock::now();
#endif

#if DEBUG 
    float* h_single;
    h_single = (float*)malloc(fftH * fftW * sizeof(float));
    for (int i = 0; i < para.numberOfFilter; i++) {
        int offs = i * fftH * fftW;
        gpuErrorCheck(cudaMemcpy(h_single, &d_DepthResult[offs], fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
        displayImage(h_single, fftW, fftH, true);
    }
#endif

    cubeReduction(d_DepthResult, d_Result, fftH, fftW, dataH, dataW, depth);

#if TIMER
    auto t10 = std::chrono::system_clock::now();
#endif

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaMemcpy(dst.data, d_Result, dataH * dataW * sizeof(uchar), cudaMemcpyDeviceToHost));

#if TIMER
    auto t11 = std::chrono::system_clock::now();
#endif

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

#if TIMER
    auto t12 = std::chrono::system_clock::now();

    printTime(t1, t2, "source registering");
    printTime(t2, t3, "c channel extracting");
    printTime(t3, t4, "source unregistering");
    printTime(t4, t5, "get gabor filter");
    printTime(t5, t6, "cudaMalloc");
    printTime(t6, t7, "padDataClampToBorder");
    printTime(t7, t8, "source FFT");
    printTime(t8, t9, "kernel FFT and mul");
    printTime(t9, t10, "CubeReduction");
    printTime(t10, t11, "D to H result");
    printTime(t11, t12, "free");
#endif
}

