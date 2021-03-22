#include <cuFFT.h>
#include "hair_detection_GPU.cuh"
#include "utils.h"
#include "parameters.h"


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

void getHairMaskGPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info) {

#if L3_TIMER
    auto t1 = getTime();
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
    const int depth = info.NumberOfFilter;
    const int fftH = snapTransformSize(info.Height + info.KernelH - 1);
    const int fftW = snapTransformSize(info.Width + info.KernelW - 1);
    const unsigned long src_size = src.cols * src.rows * src.channels();
    const unsigned long src_byte_size = src_size * sizeof(uchar);
    const unsigned long src_c_size = src.cols * src.rows;
    const unsigned long src_c_byte_size = src_c_size * sizeof(float);

    // host data
    cudaHostRegister(src_ptr, src_byte_size, cudaHostRegisterDefault);

#if L3_TIMER
    auto t2 = getTime();
#endif

    // device data
    uchar* device_src_ptr;
    float* device_src_c_ptr;
    gpuErrorCheck(cudaMalloc((uchar**)&device_src_ptr, src_byte_size));
    gpuErrorCheck(cudaMalloc((float**)&device_src_c_ptr, src_c_byte_size));

    // stream
    int SRC_DATA_PER_STREAM = src_size / D_NUM_STREAMS;
    int DST_DATA_PER_STREAM = src_c_size / D_NUM_STREAMS;
    int SRC_BYTES_PER_STREAM = src_byte_size / D_NUM_STREAMS;
    int DST_BYTES_PER_STREAM = src_c_byte_size / D_NUM_STREAMS;

    cudaStream_t stream[D_NUM_STREAMS];
    for (int i = 0; i < D_NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    int block_x_size = TILE_DIM;
    int block_y_size = BLOCK_DIM;
    int grid_x_size = (src.cols + TILE_DIM - 1) / TILE_DIM;
    int pruned_rows = src.rows / D_NUM_STREAMS;
    int grid_y_size = (pruned_rows + TILE_DIM - 1) / TILE_DIM;

    dim3 block(block_x_size, block_y_size);
    dim3 grid(grid_x_size, grid_y_size);

    int src_offset = 0;
    int dst_offset = 0;

    for (int i = 0; i < D_NUM_STREAMS; i++) {
        src_offset = i * SRC_DATA_PER_STREAM;
        dst_offset = i * DST_DATA_PER_STREAM;
        gpuErrorCheck(cudaMemcpyAsync(&device_src_ptr[src_offset], &src_ptr[src_offset], SRC_BYTES_PER_STREAM, cudaMemcpyHostToDevice, stream[i]));
        extractLChannelWithInstrinicFunction << < grid, block, 0, stream[i] >> > (&device_src_ptr[src_offset], &device_src_c_ptr[dst_offset], src.cols, pruned_rows, src.channels());
    }

#if L3_TIMER
    auto t3 = getTime();
#endif

    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < D_NUM_STREAMS; i++) {
        gpuErrorCheck(cudaStreamDestroy(stream[i]));
    }

    gpuErrorCheck(cudaHostUnregister(src_ptr));
    gpuErrorCheck(cudaFree(device_src_ptr));

#if L3_TIMER
    auto t4 = getTime();
#endif

    // init data
    float* h_kernels = initGaborFilterCube(info);

#if L3_TIMER
    auto t5 = getTime();
#endif

    // allocation
    gpuErrorCheck(cudaMalloc((void**)&d_Kernel, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_PaddedData, fftH * fftW * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_DepthResult, fftH * fftW * info.NumberOfFilter * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_Result, info.Height * info.Width * sizeof(uchar)));
    gpuErrorCheck(cudaMalloc((void**)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    gpuErrorCheck(cudaMalloc((void**)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    gpuErrorCheck(cudaMalloc((void**)&d_TempSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    // H to D
    gpuErrorCheck(cudaMemcpy(d_Kernel, h_kernels, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float), cudaMemcpyHostToDevice));

#if L3_TIMER
    auto t6 = getTime();
#endif

    // init value
    padDataClampToBorder(d_PaddedData, device_src_c_ptr, fftH, fftW, info.Height, info.Width, info.KernelH, info.KernelW, info.KernelY, info.KernelX);

#if L3_TIMER
    auto t7 = getTime();
#endif

    // make a FFT plan
    gpuErrorCheck(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    gpuErrorCheck(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

    // FFT data
    gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    gpuErrorCheck(cudaDeviceSynchronize());

#if L3_TIMER
    auto t8 = getTime();
#endif

    for (int i = 0; i < info.NumberOfFilter; i++) {
        int kernel_offset = i * info.KernelH * info.KernelW;
        int data_offset = i * fftH * fftW;

        padKernel(d_PaddedKernel, &(d_Kernel[kernel_offset]), fftH, fftW, info.KernelH, info.KernelW, info.KernelY, info.KernelX);

        // FFT kernel
        gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
        gpuErrorCheck(cudaDeviceSynchronize());

        // mul
        modulateAndNormalize(d_TempSpectrum, d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
        gpuErrorCheck(cufftExecC2R(fftPlanInv, (cufftComplex*)d_TempSpectrum, (cufftReal*)(&d_DepthResult[data_offset])));
        gpuErrorCheck(cudaDeviceSynchronize());
    }

#if L3_TIMER
    auto t9 = getTime();
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

    cubeReduction(d_DepthResult, d_Result, fftH, fftW, info.Height, info.Width, depth);

#if L3_TIMER
    auto t10 = getTime();
#endif

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaMemcpy(dst.data, d_Result, info.Height * info.Width * sizeof(uchar), cudaMemcpyDeviceToHost));

#if L3_TIMER
    auto t11 = getTime();
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
    //gpuErrorCheck(cudaDeviceReset()); not yet

#if L3_TIMER
    auto t12 = getTime();

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

