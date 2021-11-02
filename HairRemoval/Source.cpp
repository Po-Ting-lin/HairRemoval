#include "hairRemoval.cuh"
#include "warmup.cuh"
#include "utils.h"

int main() {
	warmup();
	cv::Mat src, dst;
	src = cv::imread(R"(../sample/demo1280.png)");
	if (!src.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}
	if (src.type() != CV_8UC3) {
		std::cout << "input image must be CV_8UC3! " << std::endl;
		return false;
	}
	//displayImage(src, "src", false);
	/*******************************************************/
	bool isGPU = true;
#if L1_TIMER
	auto start = getTime();
#endif
	const int width = src.cols;
	const int height = src.rows;
	const int channel = src.channels();
	HairRemoval hr(width, height, channel, isGPU);
	hr.Process(src, dst);
#if L1_TIMER
	auto end = getTime();
	printTime(start, end, "total time consume: ", hr.GetExceedTime());
#endif
	/*******************************************************/
	displayImage(dst, "output", false);
	//cv::imwrite("result.png", dst);

	src.release();
	dst.release();
	return 0;
}