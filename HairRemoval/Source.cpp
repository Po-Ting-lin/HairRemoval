#include "hairRemoval.cuh"
#include "warmup.cuh"
#include "utils.h"

// 82 ms 20220113 @1280x960
// 71 ms 20220117 @1280x960 gpu histogram
// 29 ms 20220117 @1280x960 inpainting iter unroll (launching a kernel is expensive)


void gridSearch(cv::Mat& src, cv::Mat& dst) {
	float hair_width = 4.0f;
	for (float beta = 0.5f; beta <= 1.0f; beta += 0.1f) {
		for (float alpha = 1.5f; alpha <= 1.5f; alpha += 0.1f) {
			printf("alpha: %f, beta: %f\n", alpha, beta);
			HairRemoval hr(src.cols, src.rows, src.channels(), alpha, beta, hair_width, 2500);
			hr.Process(src, dst);
			displayImage(dst, 0.5f);
		}
	}
}


int main(int argc, char** argv) {
	warmup();
	cv::Mat src;
	src = cv::imread(argc > 1 ? argv[1] : SAMPLE_DIR "/demo1280.png"); // alpha:1.5, beta:0.5, hairwidth:5, iters: 5000
	//src = cv::imread(R"(../sample/IMG_1040.jpg)"); // alpha:1.5, beta:0.5, hairwidth:10, iters: 2500
	if (!src.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}
	if (src.type() != CV_8UC3) {
		std::cout << "input image must be CV_8UC3! " << std::endl;
		return false;
	}
	cv::Mat dst(cv::Size(src.cols, src.rows), CV_8UC3, cv::Scalar(0));
	//displayImage(src, "src", false);
	//gridSearch(src, dst);
	/*******************************************************/
	HairRemoval hr(src.cols, src.rows, src.channels(), 1.5f, 0.5f, 4.0f, 5000);
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
	// Show the result only when an X display is available; otherwise save it.
	if (std::getenv("DISPLAY") != nullptr)
		displayImage(dst, 0.5f);
	else
		cv::imwrite("result.png", dst);

	src.release();
	dst.release();
	Check(cudaDeviceReset());
	return 0;
}