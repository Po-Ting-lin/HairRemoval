#include "hair_detection.h"

#define TIMER false

cv::Mat GaborFilter(float theta, HairDetectionParameters para) {
	cv::Mat output(cv::Size(para.kernelRadius * 2 + 1, para.kernelRadius * 2 + 1), CV_64F, cv::Scalar(0.0));
	double* outPtr = (double*)output.data;
	for (int y = -para.kernelRadius; y < para.kernelRadius + 1 ; y++) {
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

void Gabor(cv::Mat& src, cv::Mat& dst, HairDetectionParameters para) {
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
        kernel = GaborFilter(theta, para);

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

/* Gray Level Co-Occurrence Matrix (degree 0) */
void grayLevelCoOccurrenceMatrix(cv::Mat& src, cv::Mat& dst) {
    if (src.cols <= 1 || src.rows <= 1) return;
    if (dst.cols != CV_8U_DYNAMICRANGE || dst.rows != CV_8U_DYNAMICRANGE) return;

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

int entropyThesholding(cv::Mat& glcm) {
    int bestT = 0;
    float minLCM = FLT_MAX;
    float* glcmPtr = (float*)glcm.data;

#pragma omp parallel for
    for (int threshold = 0; threshold < CV_8U_DYNAMICRANGE; threshold++) {
        const int rows = glcm.rows;
        const int cols = glcm.cols;
        float pA = 0.0f;
        float pC = 0.0f;
        float meanA = 0.0f;
        float meanC = 0.0f;
        float entropyA = 0.0f;
        float entropyC = 0.0f;
        float* curptr = nullptr;

        // pA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                pA += (*curptr);
            }
        }

        // pC
        for (int r = threshold + 1; r < CV_8U_DYNAMICRANGE; r++) {
            for (int c = threshold + 1; c < CV_8U_DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                pC += (*curptr);
            }
        }

        // meanA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                meanA += ((float)r) * ((float)c) * (*curptr);
            }
        }
        meanA /= pA;

        // meanC
        for (int r = threshold + 1; r < CV_8U_DYNAMICRANGE; r++) {
            for (int c = threshold + 1; c < CV_8U_DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                meanC += ((float)r) * ((float)c) * (*curptr);
            }
        }
        meanC /= pC;


        // entropyA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                entropyA += ((float)r) * ((float)c) * (*curptr) * log2((((float)r) * ((float)c) + EPSILON) / (meanA + EPSILON));
                entropyA += meanA * (*curptr) * log2(meanA / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON);
            }
        }

        // entropyC
        for (int r = threshold + 1; r < CV_8U_DYNAMICRANGE; r++) {
            for (int c = threshold + 1; c < CV_8U_DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                entropyC += ((float)r) * ((float)c) * (*curptr) * log2((((float)r) * ((float)c) + EPSILON) / (meanC + EPSILON));
                entropyC += meanC * (*curptr) * log2(meanC / (((float)r) + EPSILON) / (((float)c) + EPSILON) + EPSILON);
            }
        }

        if (threshold == 1) {
            int aa = 1;
        }

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

void cleanIsolatedComponent(cv::Mat& src, HairDetectionParameters para) {
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
    cv::Mat look_up_table(cv::Size(1, CV_8U_DYNAMICRANGE), CV_8U, cv::Scalar(0));
    uchar* lutPtr = look_up_table.data;

    for (int i = 0; i < label_to_stay.size(); i++) {
        *(lutPtr + label_to_stay[i]) = CV_8U_DYNAMICRANGE - 1;
    }

    labels.convertTo(labels_uint8, CV_8U);
    cv::LUT(labels_uint8, look_up_table, dst);

    src = dst;
}

void inpaintHair(cv::Mat& src, cv::Mat& dst, cv::Mat& mask, HairDetectionParameters para) {
    int dilation_size = 1;
    cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(mask, mask, element, cv::Point(-1, -1), 2);
    cv::inpaint(src, mask, dst, para.radiusOfInpaint, cv::INPAINT_TELEA);
}

void cvtL(cv::Mat& src, cv::Mat& dst) {
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

bool hairDetection(cv::Mat& src, cv::Mat& dst, bool isGPU) {
    if (!src.data) {
        std::cout << "Error: the image wasn't correctly loaded." << std::endl;
        return false;
    }

    if (src.type() != CV_8UC3) {
        std::cout << "input image must be CV_8UC3! " << std::endl;
        return false;
    }

    HairDetectionParameters para = HairDetectionParameters();
    SetInfo(para);
    
#if TIMER
    auto t1 = std::chrono::system_clock::now();
#endif

    cv::Mat mask(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
    if (isGPU) {
        getHairMask(src, mask, para);
    }
    else {
        cv::Mat chL(cv::Size(src.cols, src.rows), CV_8U);
        cvtL(src, chL);
        Gabor(chL, mask, para);
    }

#if TIMER
    auto t4 = std::chrono::system_clock::now();
#endif

    cv::Mat glcm(cv::Size(CV_8U_DYNAMICRANGE, CV_8U_DYNAMICRANGE), CV_32F, cv::Scalar(0));
    grayLevelCoOccurrenceMatrix(mask, glcm);

#if TIMER
    auto t5 = std::chrono::system_clock::now();
#endif

    if (isGPU) {
        cv::threshold(mask, mask, entropyThesholdingGPU(glcm), CV_8U_DYNAMICRANGE - 1, 0);
    }
    else {
        cv::threshold(mask, mask, entropyThesholding(glcm), CV_8U_DYNAMICRANGE - 1, 0);
    }
    glcm.release();

#if TIMER
    auto t6 = std::chrono::system_clock::now();
#endif

    cleanIsolatedComponent(mask, para);

#if TIMER
    auto t7 = std::chrono::system_clock::now();
#endif

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);

    //inpaintHair(src, dst, mask, para);

    dst = mask;

#if TIMER
    printTime(t1, t4, "get hair mask");
    printTime(t4, t5, "glcm_cal");
    printTime(t5, t6, "entropyThesholding");
    printTime(t6, t7, "cleanIsolatedComponent");
#endif

    return true;
}
