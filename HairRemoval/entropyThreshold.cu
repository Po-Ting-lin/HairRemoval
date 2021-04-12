#include "entropyThreshold.cuh"

EntropyBasedThreshold::EntropyBasedThreshold(bool isGPU) {
    _data = nullptr;
    _width = 0;
    _height = 0;
    _glcm = new float[DYNAMICRANGE * DYNAMICRANGE];
    _isGPU = false;
}

int EntropyBasedThreshold::getThreshold(cv::Mat& src) {
    _width = src.cols;
    _height = src.rows;
    _data = (uchar*)src.data;
    _getGrayLevelCoOccurrenceMatrix();
    return _isGPU ? _entropyThesholdingGPU() : _entropyThesholdingCPU();
}

void EntropyBasedThreshold::_getGrayLevelCoOccurrenceMatrix() {
    int sum = 0;
    uchar* srcPtr = _data;
    float* dstPtr = _glcm;
    uchar* curptr = nullptr;
    float* curDstPtr = nullptr;

    for (int r = 0; r < DYNAMICRANGE; r++) {
        for (int c = 0; c < DYNAMICRANGE; c++) {
            dstPtr[r * DYNAMICRANGE + c] = 0.0f;
        }
    }

    for (int r = 0; r < _height; r++) {
        for (int c = 0; c < _width - 1; c++, *curDstPtr += 1.0f, sum += 1) {
            curptr = srcPtr + (r * _width + c);
            curDstPtr = dstPtr + (int)(*curptr) * DYNAMICRANGE + (int)(*(curptr + 1));
        }
    }

    if (sum != 0) {
        for (int r = 0; r < DYNAMICRANGE; r++) {
            for (int c = 0; c < DYNAMICRANGE; c++) {
                dstPtr[r * DYNAMICRANGE + c] /= (float)sum;
            }
        }
    }
}

int EntropyBasedThreshold::_entropyThesholdingCPU() {
    int bestT = 0;
    float minLCM = FLT_MAX;
    float* glcmPtr = _glcm;
    const int rows = DYNAMICRANGE;
    const int cols = DYNAMICRANGE;

#pragma omp parallel for
    for (int threshold = 0; threshold < DYNAMICRANGE; threshold++) {
        float pA = 0.0f;
        float pC = 0.0f;
        float meanA = 0.0f;
        float meanC = 0.0f;
        float entropyA = 0.0f;
        float entropyC = 0.0f;

#if ISAVX
        __m256 _pA = SET8F(0.0f);
        __m256 _pC = SET8F(0.0f);
        __m256 _mA = SET8F(0.0f);
        __m256 _mC = SET8F(0.0f);
        __m256 _eA = SET8F(0.0f);
        __m256 _eC = SET8F(0.0f);
        __m256 _p, _x, _r, _rc;
        __m256 _epsilon;
        __m256 _meanA, _meanC;
        __m256 _f1, _f2;
#endif
        float* curptr = nullptr;

        // pA
        for (int r = 0; r < threshold + 1; r++) {
#if ISAVX
            for (int c = 0; c < threshold + 1; c += 8) {
                _loadPixel(_x, _p, cols, r, c, threshold + 1);
                _pA = ADD8F(_pA, _p);
            }
#else
            for (int c = 0; c < threshold + 1; c++) {
                pA += glcmPtr[r * cols + c];
            }
#endif
        }

        // pC
        for (int r = threshold + 1; r < DYNAMICRANGE; r++) {
#if ISAVX
            for (int c = threshold + 1; c < DYNAMICRANGE; c += 8) {
                _loadPixel(_x, _p, cols, r, c, DYNAMICRANGE);
                _pC = ADD8F(_pC, _p);
            }
#else
            for (int c = threshold + 1; c < DYNAMICRANGE; c++) {
                pC += glcmPtr[r * cols + c];
            }
#endif
        }

#if ISAVX
        pA = sum8f(_pA);
        pC = sum8f(_pC);
#endif

        // meanA
        for (int r = 0; r < threshold + 1; r++) {
#if ISAVX
            for (int c = 0; c < threshold + 1; c += 8) {
                _loadPixel(_x, _p, cols, r, c, threshold + 1);
                _r = SET8F(r);
                _mA = ADD8F(_mA, MUL8F(MUL8F(_r, _x), _p));
            }
#else
            for (int c = 0; c < threshold + 1; c++) {
                meanA += ((float)r) * ((float)c) * glcmPtr[r * cols + c];
            }
#endif
        }

        // meanC
        for (int r = threshold + 1; r < DYNAMICRANGE; r++) {
#if ISAVX
            for (int c = threshold + 1; c < DYNAMICRANGE; c += 8) {
                _loadPixel(_x, _p, cols, r, c, DYNAMICRANGE);
                _r = SET8F(r);
                _mC = ADD8F(_mC, MUL8F(MUL8F(_r, _x), _p));
            }
#else
            for (int c = threshold + 1; c < DYNAMICRANGE; c++) {
                meanC += ((float)r) * ((float)c) * glcmPtr[r * cols + c];
            }
#endif
        }

#if ISAVX
        meanA = sum8f(_mA);
        meanC = sum8f(_mC);
#endif
        meanA /= pA;
        meanC /= pC;

        // entropyA
        for (int r = 0; r < threshold + 1; r++) {
#if ISAVX
            for (int c = 0; c < threshold + 1; c += 8) {
                _loadPixel(_x, _p, cols, r, c, threshold + 1);
                _r = SET8F(r);
                _rc = MUL8F(_r, _x);
                _epsilon = SET8F(EPSILON);
                _meanA = SET8F(meanA);
                _f1 = MUL8F(MUL8F(_rc, _p), LOG28F(DIV8F(ADD8F(_rc, _epsilon), ADD8F(_meanA, _epsilon))));
                _f2 = MUL8F(MUL8F(_meanA, _p), LOG28F(ADD8F(DIV8F(DIV8F(_meanA, ADD8F(_r, _epsilon)), ADD8F(_x, _epsilon)), _epsilon)));
                _eA = ADD8F(_eA, ADD8F(_f1, _f2));
            }
#else
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                entropyA += ((float)r) * ((float)c) * (*curptr) * log2((((float)r) * ((float)c) + EPSILON) / (meanA + EPSILON));
                entropyA += meanA * (*curptr) * log2(meanA / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON);
            }
#endif
        }

        // entropyC
        for (int r = threshold + 1; r < DYNAMICRANGE; r++) {
#if ISAVX
            for (int c = threshold + 1; c < DYNAMICRANGE; c += 8) {
                _loadPixel(_x, _p, cols, r, c, DYNAMICRANGE);
                _r = SET8F(r);
                _rc = MUL8F(_r, _x);
                _epsilon = SET8F(EPSILON);
                _meanC = SET8F(meanC);
                _f1 = MUL8F(MUL8F(_rc, _p), LOG28F(DIV8F(ADD8F(_rc, _epsilon), ADD8F(_meanC, _epsilon))));
                _f2 = MUL8F(MUL8F(_meanC, _p), LOG28F(ADD8F(DIV8F(DIV8F(_meanC, ADD8F(_r, _epsilon)), ADD8F(_x, _epsilon)), _epsilon)));
                _eC = ADD8F(_eC, ADD8F(_f1, _f2));
            }
#else
            for (int c = threshold + 1; c < DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                entropyC += ((float)r) * ((float)c) * (*curptr) * log2((((float)r) * ((float)c) + EPSILON) / (meanC + EPSILON));
                entropyC += meanC * (*curptr) * log2(meanC / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON);
            }
#endif
        }

#if ISAVX
        entropyA = sum8f(_eA);
        entropyC = sum8f(_eC);
#endif

#pragma omp critical
        {
            if (minLCM > entropyA + entropyC) {
                bestT = threshold;
                minLCM = entropyA + entropyC;
            }
        }
    }
    return bestT;
}

int EntropyBasedThreshold::_entropyThesholdingGPU() {
    float* h_data;
    float* h_reversed_data;
    float* h_eA;
    float* h_eC;
    float* h_AC;
    float* d_data;
    float* d_reversed_data;
    float* d_pA;
    float* d_mA;
    float* d_eA;
    float* d_pC;
    float* d_mC;
    float* d_eC;

    int dynamic_range = 256;

    // host 
    h_data = _glcm;
    h_reversed_data = (float*)malloc(dynamic_range * dynamic_range * sizeof(float*));
    h_eA = (float*)malloc(dynamic_range * sizeof(float*));
    h_eC = (float*)malloc(dynamic_range * sizeof(float*));
    h_AC = (float*)malloc(dynamic_range * sizeof(float*));

    for (int i = 0, int j = dynamic_range * dynamic_range - 1; i < dynamic_range * dynamic_range; i++, j--) {
        h_reversed_data[j] = h_data[i];
    }

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

    cudaStream_t stream[E_NUM_STREAMS];
    for (int i = 0; i < E_NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    _entropyCPU(h_data, h_eA, dynamic_range, false);
    _entropyCPU(h_reversed_data, h_eC, dynamic_range, true);

    EntropyThresholdDeviceInfo info(dynamic_range);
    bool reversed = false;
    _getAreaArray(d_data, d_pA, stream, info);
    _getMeanArray(d_data, d_pA, d_mA, reversed, stream, info);
    _getEntropyArray(d_data, d_mA, d_eA, reversed, stream, info);

    reversed = true;
    _reversedData(d_data, d_reversed_data, dynamic_range);
    _getAreaArray(d_reversed_data, d_pC, stream, info);
    _getMeanArray(d_reversed_data, d_pC, d_mC, reversed, stream, info);
    _getEntropyArray(d_reversed_data, d_mC, d_eC, reversed, stream, info);
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < E_NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }


    gpuErrorCheck(cudaMemcpy(h_eA, d_eA, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(h_eC, d_eC, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));

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

void EntropyBasedThreshold::_entropyCPU(float* h_data, float* h_e, int width, bool reversed) {
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

void EntropyBasedThreshold::_getAreaArray(float* d_data, float* d_pA, cudaStream_t* stream, EntropyThresholdDeviceInfo& info) {
    float* d_buf, * d_sum_matrix;
    int sum_matirx_size = info.sumMatrixSize;
    gpuErrorCheck(cudaMalloc((void**)&d_buf, info.fullWidth * info.fullWidth * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * info.fullWidth * sizeof(float)));

    for (int i = info.startThreshold; i < info.fullWidth; i++) {
        int target_width = i + 1;
        int idx = i % E_NUM_STREAMS;
        int buf_offset = i * info.fullWidth * info.fullWidth;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = getClosedWidth(target_width);

        info.targetWidth = target_width;
        info.multipleWidth = multiple_width;
        info.preSumGrid = new dim3(iDivUp(target_width, TILE_DIM), iDivUp(target_width, TILE_DIM));
        info.sumGrid = new dim3(iDivUp(multiple_width, TILE_DIM), iDivUp(multiple_width, TILE_DIM));

        gpuErrorCheck(cudaMemcpyAsync(&d_buf[buf_offset], d_data, info.fullWidth * info.fullWidth * sizeof(float), cudaMemcpyDeviceToDevice, stream[idx]));
        _sumMatrixStream(&d_buf[buf_offset], d_pA, &d_sum_matrix[sum_matrix_offset], info, i, stream[idx]);
    }

    gpuErrorCheck(cudaStreamSynchronize(*stream));
    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void EntropyBasedThreshold::_getMeanArray(float* d_data, float* d_pA, float* d_mA, bool reversed, cudaStream_t* stream, EntropyThresholdDeviceInfo& info) {
    float* d_buf, * d_data_rc, * d_sum_matrix;
    int sum_matirx_size = info.sumMatrixSize;
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(info.fullWidth, TILE_DIM), iDivUp(info.fullWidth, TILE_DIM));
    gpuErrorCheck(cudaMalloc((void**)&d_buf, info.fullWidth * info.fullWidth * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_data_rc, info.fullWidth * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * info.fullWidth * sizeof(float)));

    // r * c * element
    multiplyRCKernel << <rc_grid, rc_block >> > (d_data_rc, d_data, info.fullWidth, reversed);
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = info.startThreshold; i < info.fullWidth; i++) {
        int target_width = i + 1;
        int idx = i % E_NUM_STREAMS;
        int buf_offset = i * info.fullWidth * info.fullWidth;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = getClosedWidth(target_width);

        info.targetWidth = target_width;
        info.multipleWidth = multiple_width;
        info.preSumGrid = new dim3(iDivUp(target_width, TILE_DIM), iDivUp(target_width, TILE_DIM));
        info.sumGrid = new dim3(iDivUp(multiple_width, TILE_DIM), iDivUp(multiple_width, TILE_DIM));

        gpuErrorCheck(cudaMemcpyAsync(&d_buf[buf_offset], d_data_rc, info.fullWidth * info.fullWidth * sizeof(float), cudaMemcpyDeviceToDevice, stream[idx]));
        _sumMatrixStream(&d_buf[buf_offset], d_mA, &d_sum_matrix[sum_matrix_offset], info, i, stream[idx]);
    }
    gpuErrorCheck(cudaStreamSynchronize(*stream));

    // divide area
    dividePArrayKernel << <1, info.fullWidth >> > (d_pA, d_mA, info.fullWidth);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_data_rc));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void EntropyBasedThreshold::_getEntropyArray(float* d_data, float* d_mA, float* d_eA, bool reversed, cudaStream_t* stream, EntropyThresholdDeviceInfo& info) {
    float* d_buf, * d_sum_matrix;
    int sum_matirx_size = info.sumMatrixSize;
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(info.fullWidth, TILE_DIM), iDivUp(info.fullWidth, TILE_DIM));

    gpuErrorCheck(cudaMalloc((void**)&d_buf, info.fullWidth * info.fullWidth * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * info.fullWidth * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * info.fullWidth * sizeof(float)));

    for (int i = info.startThreshold; i < info.fullWidth; i++) {
        int target_width = i + 1;
        int idx = i % E_NUM_STREAMS;
        int buf_offset = i * info.fullWidth * info.fullWidth;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = getClosedWidth(target_width);

        info.targetWidth = target_width;
        info.multipleWidth = multiple_width;
        info.preSumGrid = new dim3(iDivUp(target_width, TILE_DIM), iDivUp(target_width, TILE_DIM));
        info.sumGrid = new dim3(iDivUp(multiple_width, TILE_DIM), iDivUp(multiple_width, TILE_DIM));

        computeEntropyMatrixKernel << <rc_grid, rc_block, 0, stream[idx] >> > (&d_buf[buf_offset], d_data, info.fullWidth, d_mA, i, reversed);
        _sumMatrixStream(&d_buf[buf_offset], d_eA, &d_sum_matrix[sum_matrix_offset], info, i, stream[idx]);
    }

    gpuErrorCheck(cudaStreamSynchronize(*stream));
    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void EntropyBasedThreshold::_sumMatrixStream(float* d_buf, float* d_arr, float* d_sum_matrix, EntropyThresholdDeviceInfo& info, int threshold, cudaStream_t stream) {
    if (info.targetWidth != info.multipleWidth) {
        preSumXMatrixKernel << <*info.preSumGrid, *info.preSumBlock, info.preSumSmemSize, stream >> > (d_buf, info.fullWidth, info.targetWidth, info.multipleWidth);
        preSumYMatrixKernel << <*info.preSumGrid, *info.preSumBlock, info.preSumSmemSize, stream >> > (d_buf, info.fullWidth, info.targetWidth, info.multipleWidth);
    }
    sumMatirxKernel << <*info.sumGrid, *info.sumBlock, info.sumSmemSize, stream >> > (d_buf, info.fullWidth, info.multipleWidth, d_sum_matrix);
    sumSumMatrixKernel << <*info.sumSumGrid, *info.sumSumBlock, info.sumSumSmemSize, stream >> > (d_sum_matrix, d_arr, info.sumMatrixSize, threshold);
}

void EntropyBasedThreshold::_reversedData(float* d_data, float* d_reversed_data, int full_width) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
    reversedDataKernel << <grid, block >> > (d_data, d_reversed_data, full_width);
    gpuErrorCheck(cudaDeviceSynchronize());
}

#if ISAVX
inline void EntropyBasedThreshold::_loadPixel(__m256& x, __m256& p, int width, int r, int c, int cBoundary) {
    x = SET8FE(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
    __m256i _x_mask = GETMASK(x, SET8F(cBoundary));
    p = SET8F(0.0f);
    p = MASKLOAD(&_glcm[r * width + c], _x_mask);
}
#endif
