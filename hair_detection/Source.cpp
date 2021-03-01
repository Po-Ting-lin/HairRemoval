#include "hair_detection_main.h"
#include "utils.h"

int main() {
	cv::Mat src, dst;
	src = cv::imread(R"(raw2560.jpg)");

	if (!src.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}

	//displayImage(image, "image", false);
	/*******************************************************/
	bool isGPU = true;
#if L1_TIMER
	auto start = getTime();
#endif
	hairDetection(src, dst, isGPU);
#if L1_TIMER
	auto end = getTime();
	printTime(start, end, "total time consume: ");
#endif
	/*******************************************************/
	displayImage(dst, "output", true);

	src.release();
	dst.release();
	return 0;
}