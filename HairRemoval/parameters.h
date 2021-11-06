#pragma once
#define DETECT_TILE_X 32
#define DETECT_TILE_Y 32
#define DETECT_UNROLL_Y 4
#define INPAINT_TILE_X 64
#define INPAINT_TILE_Y 32
#define INPAINT_UNROLL_Y 4
#define INPAINT_SMEM_TILE_X 8
#define INPAINT_SMEM_TILE_Y 8
#define STEP 8
#define PAD_STEP 1

#define EPSILON 1e-8
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
	int FFTW;
	int FFTH;
	float Alpha;
	float Beta;
	float HairWidth;
	float RatioBBox;
	float SigmaX;
	float SigmaY;

	HairDetectionInfo() {}
	HairDetectionInfo(int width, int height, int channels) {
		Width = width;
		Height = height;
		Channels = channels;
		NumberOfFilter = 8;
		MinArea = 10;
		Alpha = 1.4f;
		Beta = 0.5f;
		HairWidth = 5.0f;
		RatioBBox = 4.0f;
		SigmaX = 8.0f * (sqrt(2.0 * log(2) / CV_PI)) * HairWidth / Alpha / Beta / CV_PI;
		SigmaY = 0.8f * SigmaX;
		KernelRadius = ceil(3.0f * SigmaX);  // sigmaX > sigamY
		KernelW = 2 * KernelRadius + 1;
		KernelH = 2 * KernelRadius + 1;
		KernelX = KernelRadius;
		KernelY = KernelRadius;
		FFTH = snapTransformSize(Height + KernelH - 1);
		FFTW = snapTransformSize(Width + KernelW - 1);
	}
};

class HairInpaintInfo {
public:
	int Width;
	int Height;
	int Channels;
	int NumberOfC1Elements;
	int NumberOfC3Elements;
	int Iters;
	int* MinRgb;
	int* MaxRgb;

	HairInpaintInfo() {}
	HairInpaintInfo(int width, int height, int channels) {
		Iters = 500;
		Width = width;
		Height = height;
		Channels = channels;
		MinRgb = new int[3] { 255, 255, 255};
		MaxRgb = new int[3] { 0, 0, 0};
		NumberOfC1Elements = width * height;
		NumberOfC3Elements = width * height * channels;
	}
};
