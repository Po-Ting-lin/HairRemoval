#include "hair_inpainting_GPU.cuh"

#define DATA_TILE_DIM 34

__constant__ float dt[1];
__constant__ float center_w[1];

__global__ void PDEHeatDiffusionSMEM(float* mask, float* src, float* tempSrc, int width, int height) {
	__shared__ float smem[DATA_TILE_DIM][DATA_TILE_DIM][3];
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= width || y >= height) return;
	int c1i = y * width + x;
	int c3i = y * (width * 3) + (x * 3);
	int c3ui = (y - 1) * (width * 3) + (x * 3);
	int c3di = (y + 1) * (width * 3) + (x * 3);
	int c3li = y * (width * 3) + ((x - 1) * 3);
	int c3ri = y * (width * 3) + ((x + 1) * 3);

	smem[threadIdx.y + 1][threadIdx.x + 1][0] = tempSrc[c3i]; // put your pixel
	smem[threadIdx.y + 1][threadIdx.x + 1][1] = tempSrc[c3i + 1]; // put your pixel
	smem[threadIdx.y + 1][threadIdx.x + 1][2] = tempSrc[c3i + 2]; // put your pixel

	// serious branching and bank confict
	if (x != 0 && threadIdx.x == 0) {
		smem[threadIdx.y + 1][threadIdx.x][0] = tempSrc[c3li]; // put your left pixel
		smem[threadIdx.y + 1][threadIdx.x][1] = tempSrc[c3li + 1]; // put your left pixel
		smem[threadIdx.y + 1][threadIdx.x][2] = tempSrc[c3li + 2]; // put your left pixel
	}
	if (x != width - 1 && threadIdx.x == TILE_DIM - 1) {
		smem[threadIdx.y + 1][threadIdx.x + 2][0] = tempSrc[c3ri]; // put your right pixel
		smem[threadIdx.y + 1][threadIdx.x + 2][1] = tempSrc[c3ri + 1]; // put your right pixel
		smem[threadIdx.y + 1][threadIdx.x + 2][2] = tempSrc[c3ri + 2]; // put your right pixel
	}
	if (y != 0 && threadIdx.y == 0) {
		smem[threadIdx.y][threadIdx.x + 1][0] = tempSrc[c3ui]; // put your up pixel
		smem[threadIdx.y][threadIdx.x + 1][1] = tempSrc[c3ui + 1]; // put your up pixel
		smem[threadIdx.y][threadIdx.x + 1][2] = tempSrc[c3ui + 2]; // put your up pixel
	}
	if (y != height - 1 && threadIdx.y == TILE_DIM - 1) {
		smem[threadIdx.y + 2][threadIdx.x + 1][0] = tempSrc[c3di]; // put your down pixel
		smem[threadIdx.y + 2][threadIdx.x + 1][1] = tempSrc[c3di + 1]; // put your down pixel
		smem[threadIdx.y + 2][threadIdx.x + 1][2] = tempSrc[c3di + 2]; // put your down pixel
	}
	__syncthreads();

	float center = smem[threadIdx.y + 1][threadIdx.x + 1][0];
	float mask_center = mask[c1i];
	tempSrc[c3i] = center
		+ dt[0] * (smem[threadIdx.y][threadIdx.x + 1][0] + smem[threadIdx.y + 2][threadIdx.x + 1][0] + smem[threadIdx.y + 1][threadIdx.x][0] + smem[threadIdx.y + 1][threadIdx.x + 2][0] - center_w[0] * center)
		- dt[0] * mask_center * (center - src[c3i]);

	center = smem[threadIdx.y + 1][threadIdx.x + 1][1];
	tempSrc[c3i + 1] = center
		+ dt[0] * (smem[threadIdx.y][threadIdx.x + 1][1] + smem[threadIdx.y + 2][threadIdx.x + 1][1] + smem[threadIdx.y + 1][threadIdx.x][1] + smem[threadIdx.y + 1][threadIdx.x + 2][1] - center_w[0] * center)
		- dt[0] * mask_center * (center - src[c3i + 1]);

	center = smem[threadIdx.y + 1][threadIdx.x + 1][2];
	tempSrc[c3i + 2] = center
		+ dt[0] * (smem[threadIdx.y][threadIdx.x + 1][2] + smem[threadIdx.y + 2][threadIdx.x + 1][2] + smem[threadIdx.y + 1][threadIdx.x][2] + smem[threadIdx.y + 1][threadIdx.x + 2][2] - center_w[0] * center)
		- dt[0] * mask_center * (center - src[c3i + 2]);
}

__global__ void PDEHeatDiffusion(float* mask, float* src, float* tempSrc, int width, int height, int ch) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < 1 || y < 1 || x >= width -1 || y >= height - 1) return;
	float center;
	int c3i = y * (width * ch) + (x * ch);
	float mask_center = mask[y * width + x];

	for (int k = 0; k < ch; k++) {
		center = tempSrc[c3i+k];
		tempSrc[c3i+k] = center
			+ dt[0] * (tempSrc[(y - 1) * (width * ch) + (x * ch) + k]
				+ tempSrc[(y + 1) * (width * ch) + (x * ch) + k]
				+ tempSrc[y * (width * ch) + ((x - 1) * ch) + k]
				+ tempSrc[y * (width * ch) + ((x + 1) * ch) + k]
				- center_w[0] * center)
			- dt[0] * mask_center * (center - src[c3i+k]);
	}
}

void hairInpaintingGPU(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info) {
	cv::resize(src, src, cv::Size(info.Width, info.Height));
	cv::resize(mask, mask, cv::Size(info.Width, info.Height));

#if DEBUG
	uchar* h_src = (uchar*)malloc(info.NumberOfC3Elements * sizeof(uchar));
	gpuErrorCheck(cudaMemcpy(h_src, dSrc, info.NumberOfC3Elements * sizeof(uchar), cudaMemcpyDeviceToHost));
	cv::Mat plot_src(info.Height, info.Width, CV_8UC3, h_src);
	displayImage(plot_src, "d_src", true);
#endif
	float* normalized_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	float* normalized_mask = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
	float* normalized_masked_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	normalizeImage(src, mask, normalized_src, normalized_mask, normalized_masked_src, false);

#if DEBUG
	cv::Mat plot_normalized_src(info.Height, info.Width, CV_32FC3, normalized_src);
	cv::Mat plot_normalized_mask(info.Height, info.Width, CV_32FC1, normalized_mask);
	cv::Mat plot_normalized_masked_src(info.Height, info.Width, CV_32FC3, normalized_masked_src);
	displayImage(plot_normalized_src, "normalized_src", true);
	displayImage(plot_normalized_mask, "normalized_mask", true);
	displayImage(plot_normalized_masked_src, "normalized_masked_src", true);
#endif

	float* d_normalized_mask;
	float* d_normalized_masked_src;
	float* d_normalized_masked_src_updated;
	float* d_normalized_masked_src_temp;
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_updated, info.NumberOfC3Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyHostToDevice));

	const int smem_size = DATA_TILE_DIM * DATA_TILE_DIM * info.Channels * sizeof(float);
	dim3 block(TILE_DIM, TILE_DIM);
	dim3 grid(iDivUp(info.Width, TILE_DIM), iDivUp(info.Height, TILE_DIM));

	const float h_const_dt = 0.1f;
	const float h_center_w = 4.0f;
	gpuErrorCheck(cudaMemcpyToSymbol(dt, &h_const_dt, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpyToSymbol(center_w, &h_center_w, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToDevice));

	for (int i = 0; i < info.Iters; i++) {
		PDEHeatDiffusion << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.Channels);
		gpuErrorCheck(cudaDeviceSynchronize());
	}

#if DEBUG
	float* h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
	cv::Mat plot_src(info.Height, info.Width, CV_32FC3, h_result);
	cv::resize(plot_src, plot_src, cv::Size(info.Width * info.Rescale, info.Height * info.Rescale));
	displayImage(plot_src, "h_result", true);
#endif

	float* h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
	cv::Mat dst_mat(info.Height, info.Width, CV_32FC3, h_result);
	cv::resize(dst_mat, dst_mat, cv::Size(info.Width * info.Rescale, info.Height * info.Rescale));
	dst = dst_mat;
	gpuErrorCheck(cudaDeviceReset());
}

// main
void hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info) {

}