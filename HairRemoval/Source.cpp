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
	HairRemoval hr(src.cols, src.rows, src.channels());
	const int iters = DEBUG || L2_TIMER ? 1 : 10;
	double elapsed_avg_time = 0.0f;
	std::cout << "Iterations: " << iters << std::endl;
	for (int i = 0; i < iters; i++) {
#if L1_TIMER
		auto t1 = getTime();
#endif
		hr.Process(src, dst);
#if L1_TIMER
		auto t2 = getTime();
		std::chrono::duration<double> elapsed_seconds = t2 - t1;
		elapsed_avg_time += elapsed_seconds.count();
#endif
	}
#if L1_TIMER
	std::cout << "Total time consume: " << elapsed_avg_time / iters * 1000.0 << " ms, in iterations " << iters << std::endl;
#endif
	/*******************************************************/
	displayImage(dst, "output", false);
	//cv::imwrite("result.png", dst);

	src.release();
	dst.release();
	gpuErrorCheck(cudaDeviceReset());
	return 0;
}