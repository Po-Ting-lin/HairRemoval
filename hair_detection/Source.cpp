#include "hair_detection.h"
#include "utils.h"

int main() {
	cv::Mat image, output;
	image = cv::imread(R"(raw2560.jpg)");

	if (!image.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}

	//displayImage(image, "image", false);

	///*******************************************************/
	bool isGPU = true;
	auto start = std::chrono::system_clock::now();
	hairDetection(image, output, isGPU);
	auto end = std::chrono::system_clock::now();

	// time
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "time consume: " << elapsed_seconds.count() << std::endl;
	/*******************************************************/

	displayImage(output, "output", true);

	return 0;
}