#include "hair_detection_CPU.h"

cv::Mat getGaborFilter(float theta, HairDetectionInfo para) {
    cv::Mat output(cv::Size(para.kernelRadius * 2 + 1, para.kernelRadius * 2 + 1), CV_64F, cv::Scalar(0.0));
    double* outPtr = (double*)output.data;
    for (int y = -para.kernelRadius; y < para.kernelRadius + 1; y++) {
        for (int x = -para.kernelRadius; x < para.kernelRadius + 1; x++, outPtr++) {
            double xx = x;
            double yy = y;
            double xp = xx * cos(theta) + yy * sin(theta);
            double yp = yy * cos(theta) - xx * sin(theta);
            *outPtr = exp(-CV_PI * (xp * xp / para.sigmaX / para.sigmaX + yp * yp / para.sigmaY / para.sigmaY)) * cos(CV_2PI * para.beta / para.hairWidth * xp + CV_PI);
        }
    }
    return output;
}

void gaborFiltering(cv::Mat& src, cv::Mat& dst, HairDetectionInfo para) {
    const int rows = src.rows;
    const int cols = src.cols;
    const int depth = para.numberOfFilter;
    const int step = src.channels();
    uchar* cube = new uchar[rows * cols * depth];

    // filtering
#pragma omp parallel for 
    for (int curNum = 0; curNum < depth; curNum++) {
        double theta = CV_PI / depth * curNum;
        cv::Mat kernel, tmp;
        kernel = getGaborFilter(theta, para);

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

int entropyThesholding(cv::Mat& glcm) {
    int bestT = 0;
    float minLCM = FLT_MAX;
    float* glcmPtr = (float*)glcm.data;
    const int rows = glcm.rows;
    const int cols = glcm.cols;

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
                entropyA += ((float)r) * ((float)c) * (*curptr) * log2(   (  ((float)r) * ((float)c) + EPSILON  ) / (meanA + EPSILON)   );
                entropyA += meanA * (*curptr) * log2(  meanA / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON  );
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

void extractLChannel(cv::Mat& src, cv::Mat& dst) {
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

/* Gray Level Co-Occurrence Matrix (degree 0) */
void getGrayLevelCoOccurrenceMatrix(cv::Mat& src, cv::Mat& dst) {
    if (src.cols <= 1 || src.rows <= 1) return;
    if (dst.cols != DYNAMICRANGE || dst.rows != DYNAMICRANGE) return;

    int sum = 0;
    uchar* srcPtr = src.data;
    float* dstPtr = (float*)dst.data;
    uchar* curptr = nullptr;
    float* curDstPtr = nullptr;

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols - 1; c++, *curDstPtr += 1.0f, sum += 1) {
            curptr = srcPtr + (r * src.cols + c);
            curDstPtr = dstPtr + (int)(*curptr) * dst.cols + (int)(*(curptr + 1));
        }
    }

    if (sum != 0) {
        dst /= sum; // normalize to [0, 1]
    }
}

void cleanIsolatedComponent(cv::Mat& src, HairDetectionInfo para) {
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

        if ((area > para.minArea)) {
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

void inpaintHair(cv::Mat& src, cv::Mat& dst, cv::Mat& mask, HairDetectionInfo para) {
    int dilation_size = 1;
    cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(mask, mask, element, cv::Point(-1, -1), 2);
    cv::inpaint(src, mask, dst, para.radiusOfInpaint, cv::INPAINT_TELEA);
}

void getHairMaskCPU(cv::Mat& src, cv::Mat& dst, HairDetectionInfo para) {
    cv::Mat chL(cv::Size(src.cols, src.rows), CV_8U);
    extractLChannel(src, chL);
    gaborFiltering(chL, dst, para);
}