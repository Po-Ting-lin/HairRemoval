#include "HairRemoval.h"

void HairRemoval::splitThreeChannel(cv::Mat src, cv::Mat& c1, cv::Mat& c2, cv::Mat& c3) {
	std::vector<cv::Mat> channels(3);
	split(src, channels);
	c1 = channels[0];
	c2 = channels[1];
	c3 = channels[2];
}


cv::Mat HairRemoval::GaborFilter(double alpha, double beta, double hairW, double theta) {
	double lamd = sqrt(2.0 * log(2) / CV_PI);
	double sigmaX = 8.0 * lamd * hairW / alpha / beta / CV_PI;
	double sigmaY = 0.8 * sigmaX;
	int halfWidth = ceil(3.0 * sigmaX);  // sigmaX > sigamY
	cv::Mat output(cv::Size(halfWidth * 2, halfWidth * 2), CV_64F, cv::Scalar(0.0));
	double* outPtr = (double*)output.data;

	for (int y = -halfWidth; y < halfWidth; y++) {
		for (int x = -halfWidth; x < halfWidth; x++, outPtr++) {
			double xx = x;
			double yy = y;
			double xp = xx * cos(theta) + yy * sin(theta);
			double yp = yy * cos(theta) - xx * sin(theta);
			*outPtr = exp(-CV_PI * (xp * xp / sigmaX / sigmaX + yp * yp / sigmaY / sigmaY)) * cos(CV_2PI * beta / hairW * xp + CV_PI);
		}
	}
	return output;
}

cv::Mat HairRemoval::Gabor(cv::Mat& src) {
    const int rows = src.rows;
    const int cols = src.cols; 
    const int step = src.channels();

    cv::Mat output(cv::Size(cols, rows), CV_8U);
    std::vector<std::vector<uchar>> cube(rows * cols, std::vector<uchar>(this->numberOfFilter, 0));

#pragma omp parallel for 
    for (int curNum = 0; curNum < this->numberOfFilter; curNum++) {
        double theta = CV_PI / this->numberOfFilter * curNum;
        cv::Mat kernel, tmp;
        kernel = HairRemoval::GaborFilter(this->alpha, this->beta, this->hairWidth, theta);
        filter2D(src, tmp, CV_8U, kernel); // tmp.type() == CV_8U

        // put AfterFilter into a cube
        int count = 0;
        uchar* tmpPtr = tmp.data;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cube[count][curNum] = *tmpPtr;
                tmpPtr++;
                count++;
            }
        }
    }

    for (int count = 0; count < rows * cols; count++) {
        int rRow = count / cols;
        int rCol = count % cols;
        uchar* outPtr = output.data + (rRow * cols + rCol);
        *outPtr = *max_element(cube[count].begin(), cube[count].end());
    }
    //cv::normalize(output, output, 255, 0);
    return output;
}

/* Gray Level Co-Occurrence Matrix (degree 0) */
void HairRemoval::grayLevelCoOccurrenceMatrix(cv::Mat& src, cv::Mat& dst) {
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

int HairRemoval::entropyThesholding(cv::Mat& glcm) {
    int bestT = 0;
    double minLCM = DBL_MAX;
    float* glcmPtr = (float*)glcm.data;

#pragma omp parallel for
    for (int threshold = 0; threshold < CV_8U_DYNAMICRANGE; threshold++) {
        const int rows = glcm.rows;
        const int cols = glcm.cols;
        double pA = 0.0;
        double pC = 0.0;
        double meanA = 0.0;
        double meanC = 0.0;
        double entropyA = 0.0;
        double entropyC = 0.0;
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
                meanA += ((double)r) * ((double)c) * (*curptr);
            }
        }
        meanA /= pA;

        // meanC
        for (int r = threshold + 1; r < CV_8U_DYNAMICRANGE; r++) {
            for (int c = threshold + 1; c < CV_8U_DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                meanC += ((double)r) * ((double)c) * (*curptr);
            }
        }
        meanC /= pC;


        // entropyA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                curptr = glcmPtr + (r * cols + c);
                entropyA += ((double)r) * ((double)c) * (*curptr) * log2((((double)r) * ((double)c) + EPSILON) / (meanA + EPSILON));
                entropyA += meanA * (*curptr) * log2(meanA / (((double)r) + EPSILON) / (((double)c) + EPSILON) + EPSILON);
            }
        }

        // entropyC
        for (int r = threshold + 1; r < CV_8U_DYNAMICRANGE; r++) {
            for (int c = threshold + 1; c < CV_8U_DYNAMICRANGE; c++) {
                curptr = glcmPtr + (r * cols + c);
                entropyC += ((double)r) * ((double)c) * (*curptr) * log2((((double)r) * ((double)c) + EPSILON) / (meanC + EPSILON));
                entropyC += meanC * (*curptr) * log2(meanC / (((double)r) + EPSILON) / (((double)c) + EPSILON) + EPSILON);
            }
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

void HairRemoval::cleanIsolatedComponent(cv::Mat& src) {
    cv::Mat labels, stats, centroids;
    int components = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    for (int i = 1; i < components; i++) {
        int bigW = std::max(stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        int smallW = std::min(stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        double ratio = (double)bigW / (double)smallW;
        if (stats.at<int>(i, cv::CC_STAT_AREA) < this->minArea && ratio < this->ratioBBox) {
            src.setTo(0, labels == i);
        }
    }
}

void HairRemoval::inpaintHair(cv::Mat& src, cv::Mat& dst, cv::Mat& mask) {
    int dilation_size = 1;
    cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(mask, mask, element, cv::Point(-1, -1), 2);
    cv::inpaint(src, mask, dst, this->radiusOfInpaint, cv::INPAINT_TELEA);
}

bool HairRemoval::process(cv::Mat& src, cv::Mat& dst) {
    if (!src.data) {
        std::cout << "Error: the image wasn't correctly loaded." << std::endl;
        return false;
    }

    if (src.type() != CV_8UC3) {
        std::cout << "input image must be CV_8UC3! " << std::endl;
        return false;
    }

    cv::Mat cieimage, chL, chA, chB, mask;
    cv::Mat glcm(cv::Size(CV_8U_DYNAMICRANGE, CV_8U_DYNAMICRANGE), CV_32F, cv::Scalar(0));

    cvtColor(src, cieimage, cv::COLOR_RGB2Lab);
    splitThreeChannel(cieimage, chL, chA, chB);
    cieimage.release();
    chA.release();
    chB.release();
    mask = Gabor(chL);
    grayLevelCoOccurrenceMatrix(mask, glcm);
    cv::threshold(mask, mask, entropyThesholding(glcm), CV_8U_DYNAMICRANGE-1, 0);
    glcm.release();
    cleanIsolatedComponent(mask);
    inpaintHair(src, dst, mask);
    return true;
}

// ********************************************************************** //
// take a look
void feed2Cube(cv::Mat& src, std::vector<uchar>& arr, int targetChannel) {
    const int cols = src.cols;
    const int step = src.channels();
    const int rows = src.rows;
    int count = 0;
    for (int y = 0; y < rows; y++) {
        unsigned char* p_row = src.ptr(y) + targetChannel;
        unsigned char* row_end = p_row + cols * step;
        for (; p_row != row_end; p_row += step) {
            *p_row = arr[count];
            count++;
        }
    }
}

