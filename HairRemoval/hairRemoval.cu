#include "hairRemoval.cuh"

HairRemoval::HairRemoval(int width, int height, int channel) {
    _detectionInfo = HairDetectionInfo(width, height, channel);
    _inpaintInfo = HairInpaintInfo(width, height, channel);
    // temporarily move to here, because it's slow in CUDA 11)
    Check(cufftPlan2d(&_fftPlanFwd, _detectionInfo.FFTH, _detectionInfo.FFTW, CUFFT_R2C));
    Check(cufftPlan2d(&_fftPlanInv, _detectionInfo.FFTH, _detectionInfo.FFTW, CUFFT_C2R));
}

HairRemoval::~HairRemoval() {
    //Check(cufftDestroy(_fftPlanInv));
    //Check(cufftDestroy(_fftPlanFwd));
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
    Check(cudaDeviceSynchronize());
    getLastCudaError("CubeReductionKernel<<<>>> execution failed\n");
}

void HairRemoval::_binarization(uchar* d_Src, int threshold) {
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y);
    dim3 grid(iDivUp(_detectionInfo.Width, DETECT_TILE_X), iDivUp(_detectionInfo.Height, DETECT_TILE_Y));
    binarizeKernel << <grid, block >> > (
        d_Src,
        _detectionInfo.Width,
        _detectionInfo.Height,
        threshold
        );
    Check(cudaDeviceSynchronize());
    getLastCudaError("CubeReductionKernel<<<>>> execution failed\n");
}

void HairRemoval::_dilation(uchar* d_Src, bool* d_Dst) {
    dim3 block(DETECT_TILE_X, DETECT_TILE_Y);
    dim3 grid(iDivUp(_detectionInfo.Width, DETECT_TILE_X), iDivUp(_detectionInfo.Height, DETECT_TILE_Y));
    NaiveDilationKernel << <grid, block >> > (
        d_Src,
        d_Dst,
        _detectionInfo.Width,
        _detectionInfo.Height
        );
    Check(cudaDeviceSynchronize());
    getLastCudaError("NaiveDilationKernel<<<>>> execution failed\n");
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

void HairRemoval::_makeHistogram(uchar* d_src, int* h_histogram) {
    const int size = _detectionInfo.Height * _detectionInfo.Width;
    thrust::device_vector<uchar> src(d_src, d_src + size); // memcpy D to D
    thrust::device_vector<int> dst(DYNAMICRANGE, 0); // malloc
    // sort data to bring equal elements together
    thrust::sort(src.begin(), src.end());
    // find the end of each bin of values
    thrust::counting_iterator<int> search_begin(0);
    thrust::upper_bound(src.begin(), src.end(), search_begin, search_begin + DYNAMICRANGE, dst.begin());
    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(dst.begin(), dst.end(), dst.begin());
    thrust::copy(dst.begin(), dst.end(), h_histogram); // memcpy D to H
}

int HairRemoval::_findOtsuThreshold(int* h_histogram) {
    float all_u = 0;
    float u1 = 0;
    float w1 = 0;
    float max_variance = 0;
    int threshold = 0;
    float scale = 1.0f / (_detectionInfo.Width * _detectionInfo.Height);
    for (int i = 0; i < DYNAMICRANGE; i++)
        all_u += i * h_histogram[i];
    all_u *= scale;
    for (int i = 0; i < DYNAMICRANGE; i++) {
        float wi, w2, u2, variance;
        wi = h_histogram[i] * scale; // normalize count
        u1 *= w1;
        w1 += wi;
        w2 = 1.0f - w1;
        if (std::min(w1, w2) < FLT_EPSILON || std::max(w1, w2) > 1.0f - FLT_EPSILON) continue;
        u1 = (u1 + i * wi) / w1;
        u2 = (all_u - w1 * u1) / w2;
        variance = w1 * w2 * (u1 - u2) * (u1 - u2);
        if (variance > max_variance)
        {
            max_variance = variance;
            threshold = i;
        }
    }
    return threshold;
}

void HairRemoval::_normalizeImage(float* dstMaskImage) {
    const int size = _inpaintInfo.Width * _inpaintInfo.Height;
    thrust::device_ptr<uchar> d_ptr = thrust::device_pointer_cast(_detectionInfo.SplitSrc);

    dim3 block(NORMALIZED_TILE);
    dim3 grid(iDivUp(size, NORMALIZED_TILE));

    for (int k = 0; k < _inpaintInfo.Channels; k++) {
        const int offset = k * size;
        thrust::pair<thrust::device_vector<uchar>::iterator, thrust::device_vector<uchar>::iterator> tuple;
        tuple = thrust::minmax_element(d_ptr + offset, d_ptr + offset + size);
        uchar min = *(tuple.first);
        uchar max = *(tuple.second);
        _inpaintInfo.MaxRgb[k] = max;
        _inpaintInfo.MinRgb[k] = min;
        uchar* split_src = _detectionInfo.SplitSrc + offset;
        float* dst = dstMaskImage + offset;
        makeMaskSrcImageKernel << <grid, block >> > (split_src, _detectionInfo.Mask, dst, (float)max, (float)min, size);
        Check(cudaDeviceSynchronize());
    }
    NotKernel << <grid, block >> > (_detectionInfo.Mask, size);
    Check(cudaDeviceSynchronize());
}

void HairRemoval::_pdeHeatDiffusion(float* d_masked_src, uchar* h_dst) {
    HairInpaintInfo info = _inpaintInfo;
    float* d_processed_mask;
    uchar* d_raw_dst;
    Check(cudaMalloc((uchar**)&d_raw_dst, info.NumberOfC3Elements * sizeof(uchar)));
    Check(cudaMalloc((float**)&d_processed_mask, info.NumberOfC3Elements * sizeof(float)));

    cudaStream_t* streams = new cudaStream_t[info.Channels];
    for (int i = 0; i < info.Channels; i++) 
        cudaStreamCreate(&streams[i]);

    dim3 block(INPAINT_TILE_X, INPAINT_TILE_Y / INPAINT_UNROLL_Y);
    dim3 grid(iDivUp(info.Width, INPAINT_TILE_X), iDivUp(info.Height / INPAINT_UNROLL_Y, INPAINT_TILE_Y / INPAINT_UNROLL_Y));

    for (int k = 0; k < info.Channels; k++) {
        int offset = k * info.Width * info.Height;
        float* image = d_masked_src + offset;
        float* processed_image = d_processed_mask + offset;
        Check(cudaMemcpyAsync(processed_image, image, info.NumberOfC1Elements * sizeof(float), cudaMemcpyDeviceToDevice, streams[k]));
        for (int i = 0; i < info.Iters / INPAINT_ITER_UNROLL; i++) {
            pdeHeatDiffusionKernel << <grid, block, 0, streams[k]>> > (_detectionInfo.Mask, image, processed_image, info.Width, info.Height);
        }
    }
    Check(cudaDeviceSynchronize());
    const int size = info.Width * info.Height;
    dim3 block2(INPAINT_TILE_X, INPAINT_TILE_Y);
    dim3 grid2(iDivUp(info.Width, INPAINT_TILE_X), iDivUp(info.Height, INPAINT_TILE_Y));
    make8UDstKernel << < grid2, block2>>>(d_processed_mask, d_raw_dst,
        (float)_inpaintInfo.MaxRgb[0], (float)_inpaintInfo.MinRgb[0],
        (float)_inpaintInfo.MaxRgb[1], (float)_inpaintInfo.MinRgb[1],
        (float)_inpaintInfo.MaxRgb[2], (float)_inpaintInfo.MinRgb[2],
        info.Width, info.Height);
    Check(cudaMemcpy(h_dst, d_raw_dst, info.NumberOfC3Elements * sizeof(uchar), cudaMemcpyDeviceToHost));
    for (int i = 0; i < info.Channels; i++) 
        cudaStreamDestroy(streams[i]);
    Check(cudaFree(d_processed_mask));
    Check(cudaFree(d_raw_dst));
}

void HairRemoval::_pdeHeatDiffusionSmem(bool* normalized_mask, float* normalized_masked_src, float* h_dst) {
    HairInpaintInfo info = _inpaintInfo;
    bool* d_normalized_mask;
    float* d_normalized_masked_src;
    float* d_normalized_masked_src_temp;
    Check(cudaMalloc((bool**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(bool)));
    Check(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float)));
    Check(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float)));

    Check(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(bool), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToDevice));


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
    Check(cudaMemcpy(h_dst, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
    Check(cudaFree(d_normalized_mask));
    Check(cudaFree(d_normalized_masked_src));
    Check(cudaFree(d_normalized_masked_src_temp));
}


