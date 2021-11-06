#include "hairRemoval.cuh"

HairRemoval::HairRemoval(int width, int height, int channel) {
    _detectionInfo = HairDetectionInfo(width, height, channel);
    _inpaintInfo = HairInpaintInfo(width, height, channel);
    // temporarily move to here, because it's slow in CUDA 11)
    gpuErrorCheck(cufftPlan2d(&_fftPlanFwd, _detectionInfo.FFTH, _detectionInfo.FFTW, CUFFT_R2C));
    gpuErrorCheck(cufftPlan2d(&_fftPlanInv, _detectionInfo.FFTH, _detectionInfo.FFTW, CUFFT_C2R));
}

HairRemoval::~HairRemoval() {
    gpuErrorCheck(cufftDestroy(_fftPlanInv));
    gpuErrorCheck(cufftDestroy(_fftPlanFwd));
}

void HairRemoval::Process(cv::Mat& src, cv::Mat& dst) {
    cv::Mat mask(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
#if L2_TIMER
    auto t1 = getTime();
#endif
	_hairDetection(src, mask);
#if L2_TIMER
    auto t2 = getTime();
#endif
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
#if L2_TIMER
    auto t3 = getTime();
#endif
    //_cleanIsolatedComponent(mask, hair_detection_info);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);
#if L2_TIMER
    auto t4 = getTime();
#endif
	_hairInpainting(src, mask, dst);
#if L2_TIMER
    auto t5 = getTime();
    printTime(t1, t2, "main -- detection");
    printTime(t2, t3, "main -- entropyThesholding");
    printTime(t3, t4, "main -- cleanIsolatedComponent & morphology");
    printTime(t4, t5, "main -- inpainting");
#endif

#if PEEK_MASK
    displayImage(mask, "mask", false);
#endif
    gpuErrorCheck(cudaDeviceReset());
}

void HairRemoval::_hairDetection(cv::Mat& src, cv::Mat& dst) {
    HairDetectionInfo info = _detectionInfo;
    uchar* src_ptr = src.data;
    const int fftH = info.FFTH;
    const int fftW = info.FFTW;
    const unsigned long src_size = src.cols * src.rows * src.channels();
    const unsigned long src_byte_size = src_size * sizeof(uchar);
    const unsigned long src_c_size = src.cols * src.rows;
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

    // allocate
    gpuErrorCheck(cudaMalloc((uchar**)&d_src_ptr, src_byte_size));
    gpuErrorCheck(cudaMalloc((float**)&d_src_c_ptr, src_c_byte_size));
    gpuErrorCheck(cudaMalloc((void**)&d_Kernel, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_PaddedData, fftH * fftW * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_DepthResult, fftH * fftW * info.NumberOfFilter * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_Result, info.Height * info.Width * sizeof(uchar)));
    gpuErrorCheck(cudaMalloc((void**)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    gpuErrorCheck(cudaMalloc((void**)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    gpuErrorCheck(cudaMalloc((void**)&d_TempSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    // init data
    float* h_kernels = _initGaborFilterCube(info);

    // H to D
    gpuErrorCheck(cudaMemcpy(d_Kernel, h_kernels, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_src_ptr, src_ptr, src_byte_size, cudaMemcpyHostToDevice));
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y / DETECT_UNROLL_Y);
    dim3 grid(iDivUp(src.cols, DETECT_TILE_X), iDivUp(src.rows, DETECT_TILE_Y));
    extractLChannelKernel << < grid, block >> > (d_src_ptr, d_src_c_ptr, src.cols, src.rows, src.channels());
    gpuErrorCheck(cudaDeviceSynchronize());

    _padDataClampToBorder(d_PaddedData, d_src_c_ptr);

    // FFT data
    gpuErrorCheck(cufftExecR2C(_fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < info.NumberOfFilter; i++) {
        int kernel_offset = i * info.KernelH * info.KernelW;
        int data_offset = i * fftH * fftW;
        _padKernel(d_PaddedKernel, &(d_Kernel[kernel_offset]));

        // FFT kernel
        gpuErrorCheck(cufftExecR2C(_fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
        gpuErrorCheck(cudaDeviceSynchronize());

        // mul
        _modulateAndNormalize(d_TempSpectrum, d_DataSpectrum, d_KernelSpectrum, 1);
        gpuErrorCheck(cufftExecC2R(_fftPlanInv, (cufftComplex*)d_TempSpectrum, (cufftReal*)(&d_DepthResult[data_offset])));
        gpuErrorCheck(cudaDeviceSynchronize());
    }
    _cubeReduction(d_DepthResult, d_Result);
    gpuErrorCheck(cudaMemcpy(dst.data, d_Result, info.Height * info.Width * sizeof(uchar), cudaMemcpyDeviceToHost));

    // free
    gpuErrorCheck(cudaFree(d_src_ptr));
    gpuErrorCheck(cudaFree(d_DataSpectrum));
    gpuErrorCheck(cudaFree(d_KernelSpectrum));
    gpuErrorCheck(cudaFree(d_PaddedData));
    gpuErrorCheck(cudaFree(d_PaddedKernel));
    gpuErrorCheck(cudaFree(d_TempSpectrum));
    gpuErrorCheck(cudaFree(d_src_c_ptr));
    gpuErrorCheck(cudaFree(d_Kernel));
    gpuErrorCheck(cudaFree(d_DepthResult));
}

void HairRemoval::_hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst) {
    HairInpaintInfo info = _inpaintInfo;
    bool* normalized_mask = (bool*)malloc(info.NumberOfC1Elements * sizeof(bool));
    float* raw_dst = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    float* normalized_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    float* normalized_masked_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    uchar* h_dst_RGB_array = (uchar*)malloc(info.NumberOfC3Elements * sizeof(uchar));
    _normalizeImage(src, mask, normalized_src, normalized_mask, normalized_masked_src);
    _pdeHeatDiffusion(normalized_mask, normalized_masked_src, raw_dst);
    //_pdeHeatDiffusionSmemGPU(d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp);
    _convertToMatArrayFormat(raw_dst, h_dst_RGB_array);
    cv::Mat dst_mat(info.Height, info.Width, CV_8UC3, h_dst_RGB_array);
    dst = dst_mat;

    free(normalized_src);
    free(normalized_mask);
    free(normalized_masked_src);
}

float* HairRemoval::_initGaborFilterCube(HairDetectionInfo para) {
    float* output = new float[para.KernelW * para.KernelH * para.NumberOfFilter];
    float* output_ptr = output;
    for (int curNum = 0; curNum < para.NumberOfFilter; curNum++) {
        float theta = (float)CV_PI / para.NumberOfFilter * curNum;
        for (int y = -para.KernelRadius; y < para.KernelRadius + 1; y++) {
            for (int x = -para.KernelRadius; x < para.KernelRadius + 1; x++, output_ptr++) {
                float xx = x;
                float yy = y;
                float xp = xx * cos(theta) + yy * sin(theta);
                float yp = yy * cos(theta) - xx * sin(theta);
                *output_ptr = exp((float)(-CV_PI) * (xp * xp / para.SigmaX / para.SigmaX + yp * yp / para.SigmaY / para.SigmaY)) * cos((float)CV_2PI * para.Beta / para.HairWidth * xp + (float)CV_PI);
            }
        }
    }
    return output;
}

void HairRemoval::_padDataClampToBorder(float* d_Dst, float* d_Src) {
    assert(d_Src != d_Dst);
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y);
    dim3 grid(iDivUp(_detectionInfo.FFTW, DETECT_TILE_X), iDivUp(_detectionInfo.FFTH, DETECT_TILE_Y));
    padDataClampToBorderKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        _detectionInfo.FFTH,
        _detectionInfo.FFTW,
        _detectionInfo.Height,
        _detectionInfo.Width,
        _detectionInfo.KernelH,
        _detectionInfo.KernelW,
        _detectionInfo.KernelY,
        _detectionInfo.KernelX
        );
    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

void HairRemoval::_padKernel(float* d_Dst, float* d_Src) {
    assert(d_Src != d_Dst);
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y);
    dim3 grid(iDivUp(_detectionInfo.KernelW, DETECT_TILE_X), iDivUp(_detectionInfo.KernelH, DETECT_TILE_Y));
    padKernelKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        _detectionInfo.FFTH,
        _detectionInfo.FFTW,
        _detectionInfo.KernelH,
        _detectionInfo.KernelW,
        _detectionInfo.KernelY,
        _detectionInfo.KernelX
        );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
    cudaDeviceSynchronize();
}

void HairRemoval::_modulateAndNormalize(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int padding) {
    assert(fftW % 2 == 0);
    const int dataSize = _detectionInfo.FFTH * (_detectionInfo.FFTW / 2 + padding);
    modulateAndNormalizeKernel << <iDivUp(dataSize, 256), 256 >> > (
        d_Dst,
        d_DataSrc,
        d_KernelSrc,
        dataSize,
        1.0f / (float)(_detectionInfo.FFTW * _detectionInfo.FFTH)
        );
    getLastCudaError("modulateAndNormalize() execution failed\n");
}

void HairRemoval::_cubeReduction(float* d_Src, uchar* d_Dst) {
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y);
    dim3 grid(iDivUp(_detectionInfo.Width, DETECT_TILE_X), iDivUp(_detectionInfo.Height, DETECT_TILE_Y));
    cubeReductionKernel << <grid, block >> > (
        d_Src,
        d_Dst,
        _detectionInfo.FFTH,
        _detectionInfo.FFTW,
        _detectionInfo.Height,
        _detectionInfo.Width,
        _detectionInfo.NumberOfFilter
        );
    getLastCudaError("CubeReductionKernel<<<>>> execution failed\n");
}

void HairRemoval::_cleanIsolatedComponent(cv::Mat& src) {
    cv::Mat labels, labels_uint8, stats, centroids;
    HairDetectionInfo info = _detectionInfo;
    std::vector<int> label_to_stay = std::vector<int>();

    int components = cv::connectedComponentsWithStats(src, labels, stats, centroids);
    int* statsPtr = (int*)stats.data;

    for (int i = 1; i < components; i++) {
        statsPtr = (int*)stats.data + i * stats.cols;
        int big_boundary = std::max(*(statsPtr + cv::CC_STAT_WIDTH), *(statsPtr + cv::CC_STAT_HEIGHT));
        int small_boundary = std::min(*(statsPtr + cv::CC_STAT_WIDTH), *(statsPtr + cv::CC_STAT_HEIGHT));
        int area = *(statsPtr + cv::CC_STAT_AREA);
        double ratio = (double)big_boundary / (double)small_boundary;

        if ((area > info.MinArea)) {
            label_to_stay.push_back(i);
        }
    }

    cv::Mat dst(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
    cv::Mat look_up_table(cv::Size(1, DYNAMICRANGE), CV_8U, cv::Scalar(0));
    uchar* lutPtr = look_up_table.data;

    for (int i = 0; i < label_to_stay.size(); i++) {
        *(lutPtr + label_to_stay[i]) = DYNAMICRANGE - 1;
    }

    labels.convertTo(labels_uint8, CV_8U);
    cv::LUT(labels_uint8, look_up_table, dst);
    src = dst;
}

void HairRemoval::_normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, bool* dstMask, float* dstMaskImage) {
    HairInpaintInfo info = _inpaintInfo;
    const int width = srcImage.cols;
    const int height = srcImage.rows;
    uchar* src_image_ptr = srcImage.data;
    uchar* src_mask_ptr = srcMask.data;
#pragma omp parallel for
    for (int i = 0; i < height * width; i++) {
        dstMask[i] = src_mask_ptr[i] == 0;
    }
    int pixel = 0;
    int index = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            index = y * (width * 3) + (x * 3);
            for (int k = 0; k < 3; k++) {
                pixel = src_image_ptr[index + k];
                if (pixel > info.MaxRgb[k]) info.MaxRgb[k] = pixel;
                if (pixel < info.MinRgb[k]) info.MinRgb[k] = pixel;
            }
        }
    }
    int range_list[] = { info.MaxRgb[0] - info.MinRgb[0], info.MaxRgb[1] - info.MinRgb[1], info.MaxRgb[2] - info.MinRgb[2] };
    for (int k = 0; k < 3; k++) {
        int channel_offset = k * width * height;
#pragma omp parallel for collapse (2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int maskI = y * width + x;
                int srcI = y * (width * 3) + (x * 3) + k;
                int dstI = channel_offset + maskI;
                float value = ((float)src_image_ptr[srcI] - info.MinRgb[k]) / range_list[k];
                dstImage[dstI] = value;
                dstMaskImage[dstI] = dstMask[maskI] ? value : 1.0f;
            }
        }

        for (int x = 0; x < width; x += width - 1) {
            for (int y = 0; y < height; y++) {
                int maskI = y * width + x;
                int dstI = channel_offset + maskI;
                dstMaskImage[dstI] = dstImage[dstI];
            }
        }
        for (int y = 0; y < height; y += height - 1) {
            for (int x = 0; x < width; x++) {
                int maskI = y * width + x;
                int dstI = channel_offset + maskI;
                dstMaskImage[dstI] = dstImage[dstI];
            }
        }
    }
}

void HairRemoval::_pdeHeatDiffusion(bool* normalized_mask, float* normalized_masked_src, float* h_dst) {
    HairInpaintInfo info = _inpaintInfo;
    bool* d_normalized_mask;
    float* d_normalized_masked_src;
    float* d_normalized_masked_src_temp;
    gpuErrorCheck(cudaMalloc((bool**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(bool)));
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float)));
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float)));

    cudaStream_t* streams = new cudaStream_t[info.Channels];
    for (int i = 0; i < info.Channels; i++) 
        cudaStreamCreate(&streams[i]);
    gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(bool), cudaMemcpyHostToDevice));

    dim3 block(INPAINT_TILE_X, INPAINT_TILE_Y / INPAINT_UNROLL_Y);
    dim3 grid(iDivUp(info.Width, INPAINT_TILE_X), iDivUp(info.Height / INPAINT_UNROLL_Y, INPAINT_TILE_Y / INPAINT_UNROLL_Y));

    for (int k = 0; k < info.Channels; k++) {
        int offset = k * info.Width * info.Height;
        gpuErrorCheck(cudaMemcpyAsync(d_normalized_masked_src + offset, normalized_masked_src + offset, info.NumberOfC1Elements * sizeof(float), cudaMemcpyHostToDevice, streams[k]));
        gpuErrorCheck(cudaMemcpyAsync(d_normalized_masked_src_temp + offset, d_normalized_masked_src + offset, info.NumberOfC1Elements * sizeof(float), cudaMemcpyDeviceToDevice, streams[k]));

        for (int i = 0; i < info.Iters; i++) {
            pdeHeatDiffusionKernel << <grid, block, 0, streams[k]>> > (d_normalized_mask, d_normalized_masked_src + offset, d_normalized_masked_src_temp + offset, info.Width, info.Height);
        }
    }

    gpuErrorCheck(cudaMemcpy(h_dst, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < info.Channels; i++) 
        cudaStreamDestroy(streams[i]);
    gpuErrorCheck(cudaFree(d_normalized_mask));
    gpuErrorCheck(cudaFree(d_normalized_masked_src));
    gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
}

void HairRemoval::_pdeHeatDiffusionSmem(bool* normalized_mask, float* normalized_masked_src, float* h_dst) {
    HairInpaintInfo info = _inpaintInfo;
    bool* d_normalized_mask;
    float* d_normalized_masked_src;
    float* d_normalized_masked_src_temp;
    gpuErrorCheck(cudaMalloc((bool**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(bool)));
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float)));
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float)));

    gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToDevice));


    assert(info.Width / INPAINT_SMEM_TILE_X * STEP == 0);
    assert(info.Height / INPAINT_SMEM_TILE_Y * STEP == 0);
    dim3 block(INPAINT_SMEM_TILE_X, INPAINT_SMEM_TILE_Y);
    dim3 grid(iDivUp(info.Width, INPAINT_SMEM_TILE_X * STEP), iDivUp(info.Height, INPAINT_SMEM_TILE_Y * STEP));

    for (int k = 0; k < info.Channels; k++) {
        float* src = d_normalized_masked_src + k * info.Width * info.Height;
        float* dst = d_normalized_masked_src_temp + k * info.Width * info.Height;
        for (int i = 0; i < info.Iters; i++) {
            pdeHeatDiffusionSMEMKernel << <grid, block >> > (d_normalized_mask, src, dst, info.Width, info.Height);
        }
    }

    gpuErrorCheck(cudaMemcpy(h_dst, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaFree(d_normalized_mask));
    gpuErrorCheck(cudaFree(d_normalized_masked_src));
    gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
}

void HairRemoval::_convertToMatArrayFormat(float* srcImage, uchar* dstImage) {
    HairInpaintInfo info = _inpaintInfo;
    for (int k = 0; k < info.Channels; k++) {
        int channel_offset = k * info.Width * info.Height;
        int range = info.MaxRgb[k] - info.MinRgb[k];
        int offset = info.MinRgb[k];
#pragma omp parallel for collapse (2)
        for (int y = 0; y < info.Height; y++) {
            for (int x = 0; x < info.Width; x++) {
                int dstI = y * (info.Width * info.Channels) + (x * info.Channels) + k;
                int srcI = channel_offset + y * info.Width + x;
                dstImage[dstI] = (uchar)(range * srcImage[srcI] + offset);
            }
        }
    }
}


