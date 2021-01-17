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
	auto start = std::chrono::system_clock::now();
	hairDetection(src, dst, isGPU);
	auto end = std::chrono::system_clock::now();

	// time
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "total time consume: " << elapsed_seconds.count() << std::endl;
	/*******************************************************/

	displayImage(dst, "output", true);

	return 0;
}