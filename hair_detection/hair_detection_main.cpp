#include "hair_detection_main.h"
#include "hair_inpainting_GPU.cuh"
#include "hair_inpainting_CPU.h"


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
    
#if L2_TIMER
    auto t1 = getTime();
#endif

    cv::Mat mask(cv::Size(src.cols, src.rows), CV_8U, cv::Scalar(0));
    if (isGPU) {
        getHairMaskGPU(src, mask, para);
    }
    else {
        getHairMaskCPU(src, mask, para);
    }

#if L2_TIMER
    auto t4 = getTime();
#endif

    EntropyBasedThreshold* thresholding = new EntropyBasedThreshold(mask);

#if L2_TIMER
    auto t5 = getTime();
#endif
    cv::threshold(mask, mask, thresholding->Process(), DYNAMICRANGE - 1, 0);

#if L2_TIMER
    auto t6 = getTime();
#endif

    //cleanIsolatedComponent(mask, para);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 2);
    
#if L2_TIMER
    auto t7 = getTime();
#endif

    cv::Mat removed_dst;
    const int rescale_factor = 2;
    const int iters = 500;
    HairInpaintInfo hair_inpainting_info(
        src.cols,
        src.rows,
        src.channels(),
        iters,
        rescale_factor);
    hairInpainting(src, mask, removed_dst, hair_inpainting_info, isGPU);
    dst = removed_dst;

#if L2_TIMER
    auto t8 = getTime();
    printTime(t1, t4, "main -- get hair mask");
    printTime(t4, t5, "main -- glcm_cal");
    printTime(t5, t6, "main -- entropyThesholding");
    printTime(t6, t7, "main -- cleanIsolatedComponent & morphology");
    printTime(t7, t8, "main -- inpainting");
#endif
    return true;
}
