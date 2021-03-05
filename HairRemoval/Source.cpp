#include "hair_detection_main.h"
#include "utils.h"

int main() {
	cv::Mat src, dst;
	src = cv::imread(R"(test22.png)");

	if (!src.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}
	displayImage(src, "src", false);
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
	cv::imwrite("result.png", dst);

	src.release();
	dst.release();
	return 0;
}