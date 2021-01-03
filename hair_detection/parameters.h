#pragma once
#define TILE_DIM 32
#define BLOCK_DIM 8
#define EPSILON 1e-8
#define D_NUM_STREAMS 6
#define E_NUM_STREAMS 15
#define CV_8U_DYNAMICRANGE 256
#define TIMER true
#define DEBUG false

struct HairDetectionParameters {
	int numberOfFilter = 8;
	int minArea = 200;
	int radiusOfInpaint = 5;
	int kernelRadius = 0;
	int kernelW = 0;
	int kernelH = 0;
	int kernelX = 0;
	int kernelY = 0;
	float alpha = 1.4f;
	float beta = 0.5f;
	float hairWidth = 5.0f;
	float ratioBBox = 4.0f;
	float sigmaX = 0.0f;
	float sigmaY = 0.0f;
};

static void SetInfo(HairDetectionParameters& para) {
	float lamd = sqrt(2.0 * log(2) / CV_PI);
	para.sigmaX = 8.0f * lamd * para.hairWidth / para.alpha / para.beta / CV_PI;
	para.sigmaY = 0.8f * para.sigmaX;
	para.kernelRadius = ceil(3.0f * para.sigmaX);  // sigmaX > sigamY;
	para.kernelW = 2 * para.kernelRadius + 1;
	para.kernelH = 2 * para.kernelRadius + 1;
	para.kernelX = para.kernelRadius;
	para.kernelY = para.kernelRadius;
}