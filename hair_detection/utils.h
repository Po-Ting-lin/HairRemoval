#pragma once
#include <opencv2/opencv.hpp>
#include <string>

inline float getRand(void)
{
    return (float)(rand() % 16);
}

inline int getClosedWidth(int width) {
    int number = (int)log2(width);
    return pow(2, number);
}

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

// Rounding up the FFT dimensions to the next power of 2,
// unless the dimension would exceed 1024, 
// in which case it's rounded up to the next multiple of 512.
// Reference: zchee -- https://github.com/zchee/cuda-sample
static int snapTransformSize(int dataSize) {
    int hiBit;
    unsigned int lowPOT, hiPOT;

    dataSize = iAlignUp(dataSize, 16);
    for (hiBit = 31; hiBit >= 0; hiBit--) {
        if (dataSize & (1U << hiBit)) {
            break;
        }
    }
    lowPOT = 1U << hiBit;

    if (lowPOT == (unsigned int)dataSize)
        return dataSize;
    hiPOT = 1U << (hiBit + 1);
    if (hiPOT <= 1024)
        return hiPOT;
    else
        return iAlignUp(dataSize, 512);
}


static void printTime(std::chrono::system_clock::time_point t1, std::chrono::system_clock::time_point t2, std::string name) {
    std::chrono::duration<double> time_lapse = t2 - t1;
    std::cout << name << " time consume: " << time_lapse.count() << " s" << std::endl;
}

static void displayImage(float* src, int width, int height, bool mag) {
    cv::Mat Out(height, width, CV_32F, src, width * sizeof(float));
    cv::Mat Out2(height, width, CV_8U);
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    minMaxLoc(Out, &minVal, &maxVal, &minLoc, &maxLoc);

    for (int i = 0; i < width * height; i++) {
        Out2.data[i] = 255 * ((src[i] - (float)minVal) / ((float)maxVal - (float)minVal));
    }

    cv::Mat Outmag;
    if (mag) {
        cv::resize(Out2, Outmag, cv::Size(Out2.cols / 4, Out2.rows / 4), 5, 5);
    }
    else {
        Out2.copyTo(Outmag);
    }
    namedWindow("here", cv::WINDOW_AUTOSIZE);
    cv::imshow("here", Outmag);
    cv::waitKey(0);
}

static void displayImage(const cv::Mat& image, const char* name, bool mag) {
	cv::Mat Out;
	if (mag) {
		cv::resize(image, Out, cv::Size(image.cols/4, image.rows/4), 5, 5);
	}
	else {
		image.copyTo(Out);
	}
	namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::imshow(name, Out);
	cv::waitKey(0);
}

// RGB to XYZ
static void RGBtoXYZ(uchar R, uchar G, uchar B, float& X, float& Y, float& Z)
{
    float r = (float)R / 255.0f;
    float g = (float)G / 255.0f;
    float b = (float)B / 255.0f;

    r = ((r > 0.04045f) ? pow((r + 0.055f) / 1.055f, 2.4) : (r / 12.92f)) * 100.0f;
    g = ((g > 0.04045f) ? pow((g + 0.055f) / 1.055f, 2.4) : (g / 12.92f)) * 100.0f;
    b = ((b > 0.04045f) ? pow((b + 0.055f) / 1.055f, 2.4) : (b / 12.92f)) * 100.0f;

    //X = 0.4124f * r + 0.3576f * g + 0.1805f * b;
    Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    //Z = 0.0193f * r + 0.1192f * g + 0.9505f * b;
}

// XYZ to CIELab
static void XYZtoLab(float X, float Y, float Z, float& L, float& a, float& b)
{
    //float x = (float)X / 95.047f;
    float y = (float)Y / 100.00f;
    //float z = (float)Z / 108.883f;

    //x = (x > 0.008856f) ? cbrt(x) : (7.787f * x + 0.1379f);
    y = (y > 0.008856f) ? cbrt(y) : (7.787f * y + 0.1379f);
    //z = (z > 0.008856f) ? cbrt(z) : (7.787f * z + 0.1379f);

    L = (116.0f * y) - 16.0f;
    //a = 500.0f * (x - y);
    //b = 200.0f * (y - z);

    // L is 0 to 100, scale to uchar
    L = L * 2.55f;
}

// RGB to CIELab
static void RGBtoLab(float R, float G, float B, float& L, float& a, float& b)
{
    float X, Y, Z;
    RGBtoXYZ(R, G, B, X, Y, Z);
    XYZtoLab(X, Y, Z, L, a, b);
}

static void Display2DArray(int* src, int nx, int ny) {
    int* src_ptr;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            src_ptr = src + j * nx + i;
            std::cout << *src_ptr << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static void Display2DArray(float* src, int nx, int ny) {
    float* src_ptr;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            src_ptr = src + j * nx + i;
            std::cout << *src_ptr << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}