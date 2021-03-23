#pragma once
#define TILE_DIM 32
#define BLOCK_DIM 8
#define EPSILON 1e-8
#define D_NUM_STREAMS 6
#define E_NUM_STREAMS 15
#define DYNAMICRANGE 256
#define POWER_OF_TWO 1
#define LOAD_FLOAT(i) d_Src[i]

#define L1_TIMER true
#define L2_TIMER true
#define L3_TIMER false
#define DEBUG false
#define PEEK_MASK false

class HairDetectionInfo {
public:
	bool IsGPU;
	int Width;
	int Height;
	int Channels;
	int NumberOfFilter;
	int MinArea;
	int KernelRadius;
	int KernelW;
	int KernelH;
	int KernelX;
	int KernelY;
	float Alpha;
	float Beta;
	float HairWidth;
	float RatioBBox;
	float SigmaX;
	float SigmaY;

	HairDetectionInfo(int width, int height, int channels, bool isGPU) {
		Width = width;
		Height = height;
		Channels = channels;
		NumberOfFilter = 8;
		MinArea = 10;
		Alpha = 1.4f;
		Beta = 0.5f;
		HairWidth = 2.0f;
		RatioBBox = 4.0f;
		SigmaX = 8.0f * (sqrt(2.0 * log(2) / CV_PI)) * HairWidth / Alpha / Beta / CV_PI;
		SigmaY = 0.8f * SigmaX;
		KernelRadius = ceil(3.0f * SigmaX);  // sigmaX > sigamY
		KernelW = 2 * KernelRadius + 1;
		KernelH = 2 * KernelRadius + 1;
		KernelX = KernelRadius;
		KernelY = KernelRadius;
		IsGPU = isGPU;
	}
};

class HairInpaintInfo {
public:
	bool IsGPU;
	int RescaleFactor;
	int Width;
	int Height;
	int Channels;
	int MixGpuChannels;
	int NumberOfC1Elements;
	int NumberOfC3Elements;
	int NumberOfC2Elements;
	int Iters;
	int* MinRgb;
	int* MaxRgb;
	float Dt;
	float Cw;

	HairInpaintInfo(int width, int height, int channels, bool isGPU) {
		RescaleFactor = 1;
		Iters = 2000;
		Width = width / RescaleFactor;
		Height = height / RescaleFactor;
		Channels = channels;
		MixGpuChannels = 2;
		Dt = 0.1f;
		Cw = 4.0f;
		MinRgb = new int[3] { 255, 255, 255};
		MaxRgb = new int[3] { 0, 0, 0};
		NumberOfC1Elements = width * height / RescaleFactor / RescaleFactor;
		NumberOfC2Elements = width * height * 2 / RescaleFactor / RescaleFactor;
		NumberOfC3Elements = width * height * channels / RescaleFactor / RescaleFactor;
		IsGPU = isGPU;
	}
};
