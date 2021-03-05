#include "hair_detection_main.h"
#include "utils.h"

int main() {
	cv::Mat src, dst;
	src = cv::imread(R"(test.png)");

	if (!src.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}

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
	displayImage(dst, "output", false);

	src.release();
	dst.release();
	return 0;
}