#include "hairRemoval.cuh"


void HairRemoval::Process(cv::Mat& src, cv::Mat& dst) {
#if L2_TIMER
    auto t1 = getTime();
#endif
    _hairDetection(src);
#if L2_TIMER
    auto t2 = getTime();
#endif
    _hairInpainting(src, dst);
#if L2_TIMER
    auto t3 = getTime();
    printTime(t1, t2, "main -- detection");
    printTime(t2, t3, "main -- inpainting");
#endif
}

void HairRemoval::_hairDetection(cv::Mat& src) {
    HairDetectionInfo info = _detectionInfo;
    uchar* src_ptr = src.data;
    const int fftH = info.FFTH;
    const int fftW = info.FFTW;
    const unsigned long src_size = info.Width * info.Height * info.Channels;
    const unsigned long src_byte_size = src_size * sizeof(uchar);
    const unsigned long src_c_size = info.Width * info.Height;
    const unsigned long src_c_byte_size = src_c_size * sizeof(float);
    float* d_PaddedData;
    float* d_Kernel;
    float* d_PaddedKernel;
    float* d_DepthResult;
    float* d_src_c_ptr;
    uchar* d_Result;
    uchar* d_src_ptr;
    fComplex* d_DataSpectrum;
    fComplex* d_KernelSpectrum;
    fComplex* d_TempSpectrum;
    int* h_histogram = new int[DYNAMICRANGE];
    Check(cudaMalloc((uchar**)&d_src_ptr, src_byte_size));
    Check(cudaMalloc((uchar**)&(_detectionInfo.SplitSrc), src_byte_size));
    Check(cudaMalloc((float**)&d_src_c_ptr, src_c_byte_size));
    Check(cudaMalloc((void**)&d_Kernel, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float)));
    Check(cudaMalloc((void**)&d_PaddedData, fftH * fftW * sizeof(float)));
    Check(cudaMalloc((void**)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    Check(cudaMalloc((void**)&d_DepthResult, fftH * fftW * info.NumberOfFilter * sizeof(float)));
    Check(cudaMalloc((void**)&d_Result, info.Height * info.Width * sizeof(uchar)));
    Check(cudaMalloc((void**)&(_detectionInfo.Mask), info.Height * info.Width * sizeof(bool)));
    Check(cudaMalloc((void**)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    Check(cudaMalloc((void**)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    Check(cudaMalloc((void**)&d_TempSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    // init filter
    float* h_kernels = _initGaborFilterCube(info);

    // H to D
    Check(cudaMemcpy(d_Kernel, h_kernels, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(d_src_ptr, src_ptr, src_byte_size, cudaMemcpyHostToDevice));
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y / DETECT_UNROLL_Y);
    dim3 grid(iDivUp(info.Width, DETECT_TILE_X), iDivUp(info.Height, DETECT_TILE_Y));

    // only extract L channel
    extractLChannelKernel << < grid, block >> > (d_src_ptr, d_src_c_ptr, _detectionInfo.SplitSrc, info.Width, info.Height, info.Channels);
    Check(cudaDeviceSynchronize());
    _padDataClampToBorder(d_PaddedData, d_src_c_ptr);

    // FFT data
    Check(cufftExecR2C(_fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    Check(cudaDeviceSynchronize());
    for (int i = 0; i < info.NumberOfFilter; i++) {
        int kernel_offset = i * info.KernelH * info.KernelW;
        int data_offset = i * fftH * fftW;
        _padKernel(d_PaddedKernel, &(d_Kernel[kernel_offset]));

        // FFT kernel
        Check(cufftExecR2C(_fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
        Check(cudaDeviceSynchronize());

        // mul
        _modulateAndNormalize(d_TempSpectrum, d_DataSpectrum, d_KernelSpectrum, 1);
        Check(cufftExecC2R(_fftPlanInv, (cufftComplex*)d_TempSpectrum, (cufftReal*)(&d_DepthResult[data_offset])));
        Check(cudaDeviceSynchronize());
    }
    _cubeReduction(d_DepthResult, d_Result);

    // histogram
    _makeHistogram(d_Result, h_histogram);
    _binarization(d_Result, _findOtsuThreshold(h_histogram));
    _dilation(d_Result, _detectionInfo.Mask);

#if PEEK_MASK
    bool* peek_mask = new bool[info.Width * info.Height];
    Check(cudaMemcpy(peek_mask, _detectionInfo.Mask, info.Width * info.Height * sizeof(bool), cudaMemcpyDeviceToHost));
    displayImage(peek_mask, info.Width, info.Height, "peek mask");
    delete[] peek_mask;
#endif

    // free
    delete[] h_histogram;
    Check(cudaFree(d_Result));
    Check(cudaFree(d_src_ptr));
    Check(cudaFree(d_DataSpectrum));
    Check(cudaFree(d_KernelSpectrum));
    Check(cudaFree(d_PaddedData));
    Check(cudaFree(d_PaddedKernel));
    Check(cudaFree(d_TempSpectrum));
    Check(cudaFree(d_src_c_ptr));
    Check(cudaFree(d_Kernel));
    Check(cudaFree(d_DepthResult));
}

void HairRemoval::_hairInpainting(cv::Mat& src, cv::Mat& dst) {
    HairInpaintInfo info = _inpaintInfo;
    float* d_masked_src;
    Check(cudaMalloc((void**)&d_masked_src, info.NumberOfC3Elements * sizeof(float)));
    _normalizeImage(d_masked_src);
    _pdeHeatDiffusion(d_masked_src, dst.data);
    Check(cudaFree(d_masked_src));
}
