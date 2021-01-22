#include "entropy_thresholding.h"

EntropyBasedThreshold::EntropyBasedThreshold(cv::Mat& src) {
    _data = (uchar*)src.data;
    _glcm = new float[DYNAMICRANGE * DYNAMICRANGE];
    _width = src.cols;
    _height = src.rows;
}

int EntropyBasedThreshold::Process() {
    _getGrayLevelCoOccurrenceMatrix();
    return _entropyThesholding();
}

/* Gray Level Co-Occurrence Matrix (degree 0) */
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

int EntropyBasedThreshold::_entropyThesholding() {
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
        __m256 _pA = _mm256_set1_ps(0.0f);
        __m256 _pC = _mm256_set1_ps(0.0f);
        __m256 _mA = _mm256_set1_ps(0.0f);
        __m256 _mC = _mm256_set1_ps(0.0f);
        __m256 _eA = _mm256_set1_ps(0.0f);
        __m256 _eC = _mm256_set1_ps(0.0f);
        __m256 _p, _x, _r, _x2;
        __m256 _rcp;
        __m256 _rc;
        __m256 _epsilon;
        __m256i _x_mask;
        __m256 _meanA, _meanC;
        __m256 _f1, _f2;
        __m256 _meanAp, _meanCp;
#endif
        float* curptr = nullptr;

        // pA
        for (int r = 0; r < threshold + 1; r++) {
#if ISAVX
            for (int c = 0; c < threshold + 1; c += 8) {
                int i = r * cols + c;
                _x = _mm256_set_ps(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
                _x2 = _mm256_set1_ps(threshold + 1);
                _x_mask = _mm256_cvtps_epi32(_mm256_cmp_ps(_x, _x2, _CMP_LT_OS));
                _p = _mm256_set1_ps(0.0f);
                _p = _mm256_maskload_ps(&glcmPtr[i], _x_mask);
                _pA = _mm256_add_ps(_pA, _p);
            }
#else
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                pA += (*curptr);
            }
#endif
        }

        // pC
        for (int r = threshold + 1; r < DYNAMICRANGE; r++) {
#if ISAVX
            for (int c = threshold + 1; c < DYNAMICRANGE; c += 8) {
                int i = r * cols + c;
                _x = _mm256_set_ps(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
                _x2 = _mm256_set1_ps(DYNAMICRANGE);
                _x_mask = _mm256_cvtps_epi32(_mm256_cmp_ps(_x, _x2, _CMP_LT_OS));
                _p = _mm256_set1_ps(0.0f);
                _p = _mm256_maskload_ps(&glcmPtr[i], _x_mask);
                _pC = _mm256_add_ps(_pC, _p);
            }
#else
            for (int c = threshold + 1; c < DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                pC += (*curptr);
            }
#endif
        }

#if ISAVX
        pA = sum8(_pA);
        pC = sum8(_pC);
#endif

        // meanA
        for (int r = 0; r < threshold + 1; r++) {
#if ISAVX
            for (int c = 0; c < threshold + 1; c += 8) {
                int i = r * cols + c;
                _x = _mm256_set_ps(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
                _x2 = _mm256_set1_ps(threshold + 1);
                _x_mask = _mm256_cvtps_epi32(_mm256_cmp_ps(_x, _x2, _CMP_LT_OS));
                _p = _mm256_set1_ps(0.0f);
                _p = _mm256_maskload_ps(&glcmPtr[i], _x_mask);

                _r = _mm256_set1_ps(r);
                _rcp = _mm256_mul_ps(_mm256_mul_ps(_r, _x), _p);
                _mA = _mm256_add_ps(_mA, _rcp);
            }
#else
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                meanA += ((float)r) * ((float)c) * (*curptr);
            }
#endif
        }

        // meanC
        for (int r = threshold + 1; r < DYNAMICRANGE; r++) {
#if ISAVX
            for (int c = threshold + 1; c < DYNAMICRANGE; c += 8) {
                int i = r * cols + c;
                _x = _mm256_set_ps(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
                _x2 = _mm256_set1_ps(DYNAMICRANGE);
                _x_mask = _mm256_cvtps_epi32(_mm256_cmp_ps(_x, _x2, _CMP_LT_OS));
                _p = _mm256_set1_ps(0.0f);
                _p = _mm256_maskload_ps(&glcmPtr[i], _x_mask);

                _r = _mm256_set1_ps(r);
                _rcp = _mm256_mul_ps(_mm256_mul_ps(_r, _x), _p);
                _mC = _mm256_add_ps(_mC, _rcp);
            }
#else
            for (int c = threshold + 1; c < DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                meanC += ((float)r) * ((float)c) * (*curptr);
            }
#endif
        }

#if ISAVX
        meanA = sum8(_mA);
        meanC = sum8(_mC);
#endif
        meanA /= pA;
        meanC /= pC;

        // entropyA
        for (int r = 0; r < threshold + 1; r++) {
#if ISAVX
            for (int c = 0; c < threshold + 1; c += 8) {
                int i = r * cols + c;
                _x = _mm256_set_ps(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
                _x2 = _mm256_set1_ps(threshold + 1);
                _x_mask = _mm256_cvtps_epi32(_mm256_cmp_ps(_x, _x2, _CMP_LT_OS));
                _p = _mm256_set1_ps(0.0f);
                _p = _mm256_maskload_ps(&glcmPtr[i], _x_mask);

                _r = _mm256_set1_ps(r);
                _rc = _mm256_mul_ps(_r, _x);
                _rcp = _mm256_mul_ps(_rc, _p);
                _epsilon = _mm256_set1_ps(EPSILON);

                _meanA = _mm256_set1_ps(meanA);
                _f1 = _mm256_mul_ps(_rcp, _mm256_log2_ps(_mm256_div_ps(_mm256_add_ps(_rc, _epsilon), _mm256_add_ps(_meanA, _epsilon))));
                _meanAp = _mm256_mul_ps(_meanA, _p);
                _f2 = _mm256_mul_ps(_meanAp, _mm256_log2_ps(_mm256_add_ps(_mm256_div_ps(_mm256_div_ps(_meanA, _mm256_add_ps(_r, _epsilon)), _mm256_add_ps(_x, _epsilon)), _epsilon)));
                _eA = _mm256_add_ps(_eA, _mm256_add_ps(_f1, _f2));
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
                int i = r * cols + c;
                _x = _mm256_set_ps(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
                _x2 = _mm256_set1_ps(DYNAMICRANGE);
                _x_mask = _mm256_cvtps_epi32(_mm256_cmp_ps(_x, _x2, _CMP_LT_OS));
                _p = _mm256_set1_ps(0.0f);
                _p = _mm256_maskload_ps(&glcmPtr[i], _x_mask);

                _r = _mm256_set1_ps(r);
                _rc = _mm256_mul_ps(_r, _x);
                _rcp = _mm256_mul_ps(_rc, _p);
                _epsilon = _mm256_set1_ps(EPSILON);
                _meanC = _mm256_set1_ps(meanC);
                _f1 = _mm256_mul_ps(_rcp, _mm256_log2_ps(_mm256_div_ps(_mm256_add_ps(_rc, _epsilon), _mm256_add_ps(_meanC, _epsilon))));
                _meanCp = _mm256_mul_ps(_meanC, _p);
                _f2 = _mm256_mul_ps(_meanCp, _mm256_log2_ps(_mm256_add_ps(_mm256_div_ps(_mm256_div_ps(_meanC, _mm256_add_ps(_r, _epsilon)), _mm256_add_ps(_x, _epsilon)), _epsilon)));
                _eC = _mm256_add_ps(_eC, _mm256_add_ps(_f1, _f2));
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
        entropyA = sum8(_eA);
        entropyC = sum8(_eC);
#endif

#pragma omp critical
        {
            if (minLCM > entropyA + entropyC) {
                bestT = threshold;
                minLCM = entropyA + entropyC;
            }
        }
    }

    std::cout << bestT << std::endl;
    return bestT;
}
