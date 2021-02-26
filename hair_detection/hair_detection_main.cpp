#include "hair_detection_main.h"
#include "hair_inpainting_GPU.cuh"


bool hairDetection(cv::Mat& src, cv::Mat& dst, bool isGPU) {
    if (!src.data) {
        std::cout << "Error: the image wasn't correctly loaded." << std::endl;
        return false;
    }

    if (src.type() != CV_8UC3) {
        std::cout << "input image must be CV_8UC3! " << std::endl;
        return false;
    }

    HairDetectionInfo para;
    
#if TIMER
    auto t1 = std::chrono::system_clock::now();
#endif

    cv::Mat mask(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
    uchar* d_src = nullptr;
    if (isGPU) {
        getHairMaskGPU(src, mask, d_src, para);
    }
    else {
        getHairMaskCPU(src, mask, para);
    }

#if TIMER
    auto t4 = std::chrono::system_clock::now();
#endif

    EntropyBasedThreshold* thresholding = new EntropyBasedThreshold(mask);

#if TIMER
    auto t5 = std::chrono::system_clock::now();
#endif
    cv::threshold(mask, mask, thresholding->Process(), DYNAMICRANGE - 1, 0);

#if TIMER
    auto t6 = std::chrono::system_clock::now();
#endif

    cleanIsolatedComponent(mask, para);

#if TIMER
    auto t7 = std::chrono::system_clock::now();
#endif

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 2);
    cv::Mat removed_dst;
    HairInpaintInfo hair_inpainting_info(
        src.cols,
        src.rows,
        src.channels(),
        2);
    hairInpainting(src, mask, d_src, removed_dst, hair_inpainting_info);

    dst = removed_dst;

#if TIMER
    auto t8 = std::chrono::system_clock::now();
    printTime(t1, t4, "main -- get hair mask");
    printTime(t4, t5, "main -- glcm_cal");
    printTime(t5, t6, "main -- entropyThesholding");
    printTime(t6, t7, "main -- cleanIsolatedComponent");
    printTime(t7, t8, "main -- inpainting");
#endif
    return true;
}
