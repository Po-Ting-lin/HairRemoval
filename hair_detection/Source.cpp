#include "hair_detection.h"
#include "utils.h"

int main() {
	cv::Mat image, output;
	//image = cv::imread(R"(raw2560.jpg)");

	//if (!image.data) {
	//	std::cout << "Error: the image wasn't correctly loaded." << std::endl;
	//	return -1;
	//}
	std::string path = "test_image.bin";
	int width = 2560;
	int height = 1920;
	///*******************************************************/
	bool isGPU = true;
	auto start = std::chrono::system_clock::now();
	hairDetection(path, width, height, output, isGPU);
	auto end = std::chrono::system_clock::now();

	// time
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "time consume: " << elapsed_seconds.count() << std::endl;
	/*******************************************************/

	displayImage(output, "output", true);

	return 0;
}