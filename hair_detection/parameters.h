#pragma once
#define TILE_DIM 32
#define BLOCK_DIM 8
#define EPSILON 1e-8
#define D_NUM_STREAMS 6
#define E_NUM_STREAMS 15
#define DYNAMICRANGE 256
#define POWER_OF_TWO 1
#define LOAD_FLOAT(i) d_Src[i]

#define TIMER false
#define DEBUG false

class HairDetectionInfo {
public:
	int numberOfFilter;
	int minArea;
	int radiusOfInpaint;
	int kernelRadius;
	int kernelW;
	int kernelH;
	int kernelX;
	int kernelY;
	float alpha;
	float beta;
	float hairWidth;
	float ratioBBox;
	float sigmaX;
	float sigmaY;

	HairDetectionInfo() {
		numberOfFilter = 8;
		minArea = 200;
		radiusOfInpaint = 5;
		alpha = 1.4f;
		beta = 0.5f;
		hairWidth = 5.0f;
		ratioBBox = 4.0f;
		sigmaX = 8.0f * (sqrt(2.0 * log(2) / CV_PI)) * hairWidth / alpha / beta / CV_PI;
		sigmaY = 0.8f * sigmaX;
		kernelRadius = ceil(3.0f * sigmaX);  // sigmaX > sigamY
		kernelW = 2 * kernelRadius + 1;
		kernelH = 2 * kernelRadius + 1;
		kernelX = kernelRadius;
		kernelY = kernelRadius;
	}
};
