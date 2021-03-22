#include "entropy_thresholding.h"
#include "entropy_thresholding.cuh"

EntropyBasedThreshold::EntropyBasedThreshold(cv::Mat& src, bool isGPU) {
    _data = (uchar*)src.data;
    _glcm = new float[DYNAMICRANGE * DYNAMICRANGE];
    _width = src.cols;
    _height = src.rows;
    _isGPU = isGPU;
}

int EntropyBasedThreshold::getThreshold() {
    _getGrayLevelCoOccurrenceMatrix();
    return _isGPU ? entropyThesholdingGPU(_glcm) : _entropyThesholding();
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
                _f2 = MUL8F(MUL8F(_meanC, _p) , LOG28F(ADD8F(DIV8F(DIV8F(_meanC, ADD8F(_r, _epsilon)), ADD8F(_x, _epsilon)), _epsilon)));
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

#if ISAVX
inline void EntropyBasedThreshold::_loadPixel(__m256& x, __m256& p, int width, int r, int c, int cBoundary) {
    x = SET8FE(c + 7.0f, c + 6.0f, c + 5.0f, c + 4.0f, c + 3.0f, c + 2.0f, c + 1.0f, c);
    __m256i _x_mask = GETMASK(x, SET8F(cBoundary));
    p = SET8F(0.0f);
    p = MASKLOAD(&_glcm[r * width + c], _x_mask);
}
#endif
