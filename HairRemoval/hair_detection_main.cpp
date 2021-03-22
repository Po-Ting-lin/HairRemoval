#include "hair_detection_main.h"
#include "hair_inpainting_CPU.h"

bool hairRemoval(cv::Mat& src, cv::Mat& dst, bool isGPU) {
    cv::Mat mask(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
    HairDetectionInfo hair_detection_info(src.cols, src.rows, src.channels(), isGPU);
    HairInpaintInfo hair_inpainting_info(src.cols, src.rows, src.channels(), isGPU);
    EntropyBasedThreshold thresholding(mask, isGPU);
#if L2_TIMER
    auto t1 = getTime();
#endif
    hairDetecting(src, mask, hair_detection_info);
#if L2_TIMER
    auto t4 = getTime();
#endif
    cv::threshold(mask, mask, thresholding.getThreshold(), DYNAMICRANGE - 1, 0);
#if L2_TIMER
    auto t6 = getTime();
#endif
    cleanIsolatedComponent(mask, hair_detection_info);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);
#if L2_TIMER
    auto t7 = getTime();
#endif
    hairInpainting(src, mask, dst, hair_inpainting_info);
#if PEEK_MASK
    displayImage(mask, "mask", false);
#endif
#if L2_TIMER
    auto t8 = getTime();
    printTime(t1, t4, "main -- get hair mask");
    printTime(t4, t6, "main -- entropyThesholding");
    printTime(t6, t7, "main -- cleanIsolatedComponent & morphology");
    printTime(t7, t8, "main -- inpainting");
#endif
    return true;
}
