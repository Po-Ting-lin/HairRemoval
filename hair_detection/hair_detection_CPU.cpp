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