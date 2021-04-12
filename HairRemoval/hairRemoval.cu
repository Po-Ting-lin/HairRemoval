#include "hairRemoval.cuh"
#include "entropyThreshold.cuh"

HairRemoval::HairRemoval(bool isGPU) {
    _isGPU = isGPU;
}

void HairRemoval::Process(cv::Mat& src, cv::Mat& dst) {
    cv::Mat mask(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
    HairDetectionInfo hair_detection_info(src.cols, src.rows, src.channels(), _isGPU);
    HairInpaintInfo hair_inpainting_info(src.cols, src.rows, src.channels(), _isGPU);
    EntropyBasedThreshold thresholding(_isGPU);

#if L2_TIMER
    auto t1 = getTime();
#endif
	_hairDetection(src, mask, hair_detection_info);
#if L2_TIMER
    auto t2 = getTime();
#endif
    cv::threshold(mask, mask, thresholding.getThreshold(mask), DYNAMICRANGE - 1, 0);
#if L2_TIMER
    auto t3 = getTime();
#endif
    //_cleanIsolatedComponent(mask, hair_detection_info);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 2);
#if L2_TIMER
    auto t4 = getTime();
#endif
	_hairInpainting(src, mask, dst, hair_inpainting_info);
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

void HairRemoval::_hairDetection(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info) {
    if (info.IsGPU) _hairDetectionGPU(src, dst, info);
    else _hairDetectionCPU(src, dst, info);
}

void HairRemoval::_hairDetectionGPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info) {
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
    cufftHandle fftPlanFwd;
    cufftHandle fftPlanInv;

    uchar* src_ptr = src.data;
    const int depth = info.NumberOfFilter;
    const int fftH = snapTransformSize(info.Height + info.KernelH - 1);
    const int fftW = snapTransformSize(info.Width + info.KernelW - 1);
    const unsigned long src_size = src.cols * src.rows * src.channels();
    const unsigned long src_byte_size = src_size * sizeof(uchar);
    const unsigned long src_c_size = src.cols * src.rows;
    const unsigned long src_c_byte_size = src_c_size * sizeof(float);
    int src_size_per_stream = src_size / D_NUM_STREAMS;
    int dst_size_per_stream = src_c_size / D_NUM_STREAMS;
    int src_bytes_per_stream = src_byte_size / D_NUM_STREAMS;
    int dst_bytes_per_stream = src_c_byte_size / D_NUM_STREAMS;
    int block_x_size = TILE_DIM;
    int block_y_size = BLOCK_DIM;
    int grid_x_size = (src.cols + TILE_DIM - 1) / TILE_DIM;
    int pruned_rows = src.rows / D_NUM_STREAMS;
    int grid_y_size = (pruned_rows + TILE_DIM - 1) / TILE_DIM;

    dim3 block(block_x_size, block_y_size);
    dim3 grid(grid_x_size, grid_y_size);

    // make a FFT plan
    gpuErrorCheck(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    gpuErrorCheck(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

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
    cudaHostRegister(src_ptr, src_byte_size, cudaHostRegisterDefault);
    gpuErrorCheck(cudaMemcpy(d_Kernel, h_kernels, info.KernelH * info.KernelW * info.NumberOfFilter * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream[D_NUM_STREAMS];
    for (int i = 0; i < D_NUM_STREAMS; i++) {cudaStreamCreate(&stream[i]);}
    for (int i = 0; i < D_NUM_STREAMS; i++) {
        int src_offset = i * src_size_per_stream;
        int dst_offset = i * dst_size_per_stream;
        gpuErrorCheck(cudaMemcpyAsync(&d_src_ptr[src_offset], &src_ptr[src_offset], src_bytes_per_stream, cudaMemcpyHostToDevice, stream[i]));
        extractLChannelWithInstrinicFunction << < grid, block, 0, stream[i] >> > (&d_src_ptr[src_offset], &d_src_c_ptr[dst_offset], src.cols, pruned_rows, src.channels());
    }
    gpuErrorCheck(cudaDeviceSynchronize());
    for (int i = 0; i < D_NUM_STREAMS; i++) {gpuErrorCheck(cudaStreamDestroy(stream[i]));}

    gpuErrorCheck(cudaHostUnregister(src_ptr));
    gpuErrorCheck(cudaFree(d_src_ptr));

    _padDataClampToBorder(d_PaddedData, d_src_c_ptr, fftH, fftW, info.Height, info.Width, info.KernelH, info.KernelW, info.KernelY, info.KernelX);

    // FFT data
    gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedData, (cufftComplex*)d_DataSpectrum));
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < info.NumberOfFilter; i++) {
        int kernel_offset = i * info.KernelH * info.KernelW;
        int data_offset = i * fftH * fftW;

        _padKernel(d_PaddedKernel, &(d_Kernel[kernel_offset]), fftH, fftW, info.KernelH, info.KernelW, info.KernelY, info.KernelX);

        // FFT kernel
        gpuErrorCheck(cufftExecR2C(fftPlanFwd, (cufftReal*)d_PaddedKernel, (cufftComplex*)d_KernelSpectrum));
        gpuErrorCheck(cudaDeviceSynchronize());

        // mul
        _modulateAndNormalize(d_TempSpectrum, d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
        gpuErrorCheck(cufftExecC2R(fftPlanInv, (cufftComplex*)d_TempSpectrum, (cufftReal*)(&d_DepthResult[data_offset])));
        gpuErrorCheck(cudaDeviceSynchronize());
    }
    _cubeReduction(d_DepthResult, d_Result, fftH, fftW, info.Height, info.Width, depth);
    gpuErrorCheck(cudaDeviceSynchronize());

    // D to H
    gpuErrorCheck(cudaMemcpy(dst.data, d_Result, info.Height * info.Width * sizeof(uchar), cudaMemcpyDeviceToHost));

    // free
    gpuErrorCheck(cufftDestroy(fftPlanInv));
    gpuErrorCheck(cufftDestroy(fftPlanFwd));
    gpuErrorCheck(cudaFree(d_DataSpectrum));
    gpuErrorCheck(cudaFree(d_KernelSpectrum));
    gpuErrorCheck(cudaFree(d_PaddedData));
    gpuErrorCheck(cudaFree(d_PaddedKernel));
    gpuErrorCheck(cudaFree(d_TempSpectrum));
    gpuErrorCheck(cudaFree(d_src_c_ptr));
    gpuErrorCheck(cudaFree(d_Kernel));
    gpuErrorCheck(cudaFree(d_DepthResult));
}

void HairRemoval::_hairDetectionCPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo info) {
    cv::Mat chL(cv::Size(src.cols, src.rows), CV_8U);
    _extractLChannel(src, chL);
    _gaborFiltering(chL, dst, info);
}

void HairRemoval::_gaborFiltering(cv::Mat& src, cv::Mat& dst, HairDetectionInfo para) {
    const int rows = src.rows;
    const int cols = src.cols;
    const int depth = para.NumberOfFilter;
    const int step = src.channels();
    uchar* cube = new uchar[rows * cols * depth];

    // filtering
#pragma omp parallel for 
    for (int curNum = 0; curNum < depth; curNum++) {
        double theta = CV_PI / depth * curNum;
        cv::Mat kernel, tmp;
        kernel = _getGaborFilter(theta, para);

        filter2D(src, tmp, CV_8U, kernel); // tmp.type() == CV_8U

        // put AfterFilter into a cube
        int count = 0;
        int idx_for_cube = 0;

        uchar* tmpPtr = tmp.data;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                idx_for_cube = curNum + depth * count;
                cube[idx_for_cube] = *tmpPtr;

                tmpPtr++;
                count++;
            }
        }
    }

    // max value
#pragma omp parallel for 
    for (int count = 0; count < rows * cols; count++) {
        int rRow = count / cols;
        int rCol = count % cols;
        int output_offset = rRow * cols + rCol;
        uchar* outPtr = dst.data + output_offset;

        uchar* start_element = cube + output_offset * depth;
        for (uchar* p = start_element; p != start_element + depth; p++) {
            if (*p > * outPtr) {
                *outPtr = *p;
            }
        }
    }
}

cv::Mat HairRemoval::_getGaborFilter(float theta, HairDetectionInfo para) {
    cv::Mat output(cv::Size(para.KernelRadius * 2 + 1, para.KernelRadius * 2 + 1), CV_64F, cv::Scalar(0.0));
    double* outPtr = (double*)output.data;
    for (int y = -para.KernelRadius; y < para.KernelRadius + 1; y++) {
        for (int x = -para.KernelRadius; x < para.KernelRadius + 1; x++, outPtr++) {
            double xx = x;
            double yy = y;
            double xp = xx * cos(theta) + yy * sin(theta);
            double yp = yy * cos(theta) - xx * sin(theta);
            *outPtr = exp(-CV_PI * (xp * xp / para.SigmaX / para.SigmaX + yp * yp / para.SigmaY / para.SigmaY)) * cos(CV_2PI * para.Beta / para.HairWidth * xp + CV_PI);
        }
    }
    return output;
}

void HairRemoval::_hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info) {
    cv::resize(src, src, cv::Size(info.Width, info.Height));
    cv::resize(mask, mask, cv::Size(info.Width, info.Height));

    float* normalized_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    float* normalized_mask = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
    float* normalized_masked_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    _normalizeImage(src, mask, normalized_src, normalized_mask, normalized_masked_src, info);
    float* h_dst_array = nullptr;
    uchar* h_dst_RGB_array = (uchar*)malloc(info.NumberOfC3Elements * sizeof(uchar));
    if (info.IsGPU) {
        _hairInpaintingGPU(normalized_mask, normalized_masked_src, h_dst_array, info);
    }
    else {
        _hairInpaintingCPU(normalized_mask, normalized_masked_src, h_dst_array, info);
    }

    _convertToMatArrayFormat(h_dst_array, h_dst_RGB_array, info);
    cv::Mat dst_mat(info.Height, info.Width, CV_8UC3, h_dst_RGB_array);
    cv::resize(dst_mat, dst_mat, cv::Size(info.Width * info.RescaleFactor, info.Height * info.RescaleFactor));
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

void HairRemoval::_extractLChannel(cv::Mat& src, cv::Mat& dst) {
    uchar* src_ptr = src.data;
    uchar* dst_ptr = dst.data;
    int n_channels = src.channels();

#pragma omp parallel for
    for (int x = 0; x < src.cols; x++) {
        for (int y = 0; y < src.rows; y++) {
            uchar R = *(src_ptr + (y * src.step) + (x * n_channels) + 0);
            uchar G = *(src_ptr + (y * src.step) + (x * n_channels) + 1);
            uchar B = *(src_ptr + (y * src.step) + (x * n_channels) + 2);
            float l;
            float a;
            float b;
            RGBtoLab(R, G, B, l, a, b);
            *(dst_ptr + y * src.cols + x) = (uchar)l;
        }
    }
}

void HairRemoval::_padDataClampToBorder(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelW, int kernelH, int kernelY, int kernelX) {
    assert(d_Src != d_Dst);
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(fftW, block.x), iDivUp(fftH, block.y));
    padDataClampToBorderKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
        );
    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

void HairRemoval::_padKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX) {
    assert(d_Src != d_Dst);
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(kernelW, block.x), iDivUp(kernelH, block.y));
    padKernelKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
        );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
    cudaDeviceSynchronize();
}

void HairRemoval::_modulateAndNormalize(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int fftH, int fftW, int padding) {
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);
    modulateAndNormalizeKernel << <iDivUp(dataSize, 256), 256 >> > (
        d_Dst,
        d_DataSrc,
        d_KernelSrc,
        dataSize,
        1.0f / (float)(fftW * fftH)
        );
    getLastCudaError("modulateAndNormalize() execution failed\n");
}

void HairRemoval::_cubeReduction(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth) {
    dim3 block(TILE_DIM, 8);
    dim3 grid(iDivUp(dataW, block.x), iDivUp(dataH, block.y));
    cubeReductionKernel << <grid, block >> > (
        d_Src,
        d_Dst,
        fftH,
        fftW,
        dataH,
        dataW,
        depth
        );
    getLastCudaError("CubeReductionKernel<<<>>> execution failed\n");
}

void HairRemoval::_cleanIsolatedComponent(cv::Mat& src, HairDetectionInfo para) {
    cv::Mat labels, labels_uint8, stats, centroids;
    std::vector<int> label_to_stay = std::vector<int>();

    int components = cv::connectedComponentsWithStats(src, labels, stats, centroids);
    int* statsPtr = (int*)stats.data;

    for (int i = 1; i < components; i++) {
        statsPtr = (int*)stats.data + i * stats.cols;
        int big_boundary = std::max(*(statsPtr + cv::CC_STAT_WIDTH), *(statsPtr + cv::CC_STAT_HEIGHT));
        int small_boundary = std::min(*(statsPtr + cv::CC_STAT_WIDTH), *(statsPtr + cv::CC_STAT_HEIGHT));
        int area = *(statsPtr + cv::CC_STAT_AREA);
        double ratio = (double)big_boundary / (double)small_boundary;

        if ((area > para.MinArea)) {
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

void HairRemoval::_normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage, HairInpaintInfo info) {
    const int width = srcImage.cols;
    const int height = srcImage.rows;
    uchar* src_image_ptr = srcImage.data;
    uchar* src_mask_ptr = srcMask.data;
#pragma omp parallel for
    for (int i = 0; i < height * width; i++) {
        dstMask[i] = src_mask_ptr[i] != 0 ? 0.0f : 1.0f;
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
                dstMaskImage[dstI] = dstMask[maskI] > 0.0f ? value : 1.0f;
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

void HairRemoval::_hairInpaintingGPU(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info) {
    float* d_normalized_mask;
    float* d_normalized_masked_src;
    float* d_normalized_masked_src_temp;
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(float)));
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float)));
    gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid(iDivUp(info.Width, TILE_DIM), iDivUp(info.Height, TILE_DIM), info.Channels);

    dim3 block2(TILE_DIM2, TILE_DIM2, 1);
    dim3 grid2(iDivUp(info.Width, TILE_DIM2), iDivUp(info.Height, TILE_DIM2), info.Channels);

    const float h_const_dt = 0.1f;
    const float h_center_w = 4.0f;
    gpuErrorCheck(cudaMemcpyToSymbol(d_dt, &h_const_dt, 1 * sizeof(float)));
    gpuErrorCheck(cudaMemcpyToSymbol(d_center_w, &h_center_w, 1 * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int i = 0; i < info.Iters; i++) {
        //PDEHeatDiffusion << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.Channels);
        if (i % 2 == 0) {
            pdeHeatDiffusionSMEM << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.Channels);
        }
        else {
            pdeHeatDiffusionSMEM2 << <grid2, block2 >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.Channels);
        }
    }
    gpuErrorCheck(cudaDeviceSynchronize());

    float* h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
    dst = h_result;

    gpuErrorCheck(cudaFree(d_normalized_mask));
    gpuErrorCheck(cudaFree(d_normalized_masked_src));
    gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
}

void HairRemoval::_hairInpaintingCPU(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info) {
    float* img_u = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
    memcpy(img_u, normalized_masked_src, info.NumberOfC3Elements * sizeof(float));
    _PDEHeatDiffusionCPU(normalized_mask, normalized_masked_src, img_u, info.Channels, info);
    dst = img_u;
}

void HairRemoval::_PDEHeatDiffusionCPU(float* normalized_mask, float* normalized_masked_src, float* dst, int ch, HairInpaintInfo info) {
    int x_boundary = info.Width - 1;
    for (int i = 0; i < info.Iters; i++) {
        for (int k = 0; k < ch; k++) {
            int channel_offset = k * info.Width * info.Height;
#pragma omp parallel for
            for (int y = 1; y < info.Height - 1; y++) {
#if ISAVX
                __m256 _pA = SET8F(0.0f);
                __m256 _pC = SET8F(0.0f);
                __m256 _mA = SET8F(0.0f);
                __m256 _mC = SET8F(0.0f);
                __m256 _eA = SET8F(0.0f);
                __m256 _eC = SET8F(0.0f);
                __m256 _dt = SET8F(info.Dt);
                __m256 _cw = SET8F(info.Cw);
                __m256 _x;
                __m256 _c, _u, _d, _l, _r, _mc, _oc;
                __m256i _x_mask;
                for (int x = 1; x < x_boundary; x += 8) {
                    int c1i = y * info.Width + x;
                    int c3i = channel_offset + c1i;
                    int c3ui = channel_offset + (y - 1) * info.Width + x;
                    int c3di = channel_offset + (y + 1) * info.Width + x;
                    int c3li = channel_offset + y * info.Width + (x - 1);
                    int c3ri = channel_offset + y * info.Width + (x + 1);

                    _x = SET8FE(x + 7.0f, x + 6.0f, x + 5.0f, x + 4.0f, x + 3.0f, x + 2.0f, x + 1.0f, x);
                    _x_mask = GETMASK(_x, SET8F(x_boundary));

                    _c = SET8F(0.0f);
                    _u = SET8F(0.0f);
                    _d = SET8F(0.0f);
                    _l = SET8F(0.0f);
                    _r = SET8F(0.0f);
                    _mc = SET8F(0.0f);
                    _oc = SET8F(0.0f);
                    _c = MASKLOAD(&dst[c3i], _x_mask);
                    _u = MASKLOAD(&dst[c3ui], _x_mask);
                    _d = MASKLOAD(&dst[c3di], _x_mask);
                    _l = MASKLOAD(&dst[c3li], _x_mask);
                    _r = MASKLOAD(&dst[c3ri], _x_mask);
                    _mc = MASKLOAD(&normalized_mask[c1i], _x_mask);
                    _oc = MASKLOAD(&normalized_masked_src[c3i], _x_mask);
                    MASKSTORE(&dst[c3i]
                        , _x_mask
                        , SUB8F(ADD8F(_c, MUL8F(_dt, SUB8F(ADD8F(_u, ADD8F(_d, ADD8F(_l, _r))), MUL8F(_cw, _c)))), MUL8F(_dt, MUL8F(_mc, SUB8F(_c, _oc)))));
                }
#else
                for (int x = 1; x < x_boundary; x++) {
                    int c1i = y * info.Width + x;
                    int c3i = channel_offset + c1i;
                    int c3ui = channel_offset + (y - 1) * info.Width + x;
                    int c3di = channel_offset + (y + 1) * info.Width + x;
                    int c3li = channel_offset + y * info.Width + (x - 1);
                    int c3ri = channel_offset + y * info.Width + (x + 1);

                    dst[c3i] = dst[c3i]
                        + info.Dt * (dst[c3ui] + dst[c3di] + dst[c3li] + dst[c3ri] - info.Cw * dst[c3i])
                        - info.Dt * normalized_mask[c1i] * (dst[c3i] - normalized_masked_src[c3i]);
                }
#endif
            }
        }
    }
}

void HairRemoval::_convertToMatArrayFormat(float* srcImage, uchar* dstImage, HairInpaintInfo info) {
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


