#include "hair_inpainting_GPU.cuh"

#define DATA_TILE_DIM 34

__constant__ float d_dt[1];
__constant__ float d_center_w[1];

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
		+ d_dt[0] * (smem[threadIdx.y][threadIdx.x + 1][0] + smem[threadIdx.y + 2][threadIdx.x + 1][0] + smem[threadIdx.y + 1][threadIdx.x][0] + smem[threadIdx.y + 1][threadIdx.x + 2][0] - d_center_w[0] * center)
		- d_dt[0] * mask_center * (center - src[c3i]);

	center = smem[threadIdx.y + 1][threadIdx.x + 1][1];
	tempSrc[c3i + 1] = center
		+ d_dt[0] * (smem[threadIdx.y][threadIdx.x + 1][1] + smem[threadIdx.y + 2][threadIdx.x + 1][1] + smem[threadIdx.y + 1][threadIdx.x][1] + smem[threadIdx.y + 1][threadIdx.x + 2][1] - d_center_w[0] * center)
		- d_dt[0] * mask_center * (center - src[c3i + 1]);

	center = smem[threadIdx.y + 1][threadIdx.x + 1][2];
	tempSrc[c3i + 2] = center
		+ d_dt[0] * (smem[threadIdx.y][threadIdx.x + 1][2] + smem[threadIdx.y + 2][threadIdx.x + 1][2] + smem[threadIdx.y + 1][threadIdx.x][2] + smem[threadIdx.y + 1][threadIdx.x + 2][2] - d_center_w[0] * center)
		- d_dt[0] * mask_center * (center - src[c3i + 2]);
}

__global__ void PDEHeatDiffusion(float* mask, float* src, float* tempSrc, int width, int height, int ch) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < 1 || y < 1 || x >= width -1 || y >= height - 1) return;
	float center;
	int c3i = 0;
	int ch_offset = 0;
	float mask_center = mask[y * width + x];

	for (int k = 0; k < ch; k++) {
		ch_offset = k * width * height;
		c3i = ch_offset + y * width + x;
		center = tempSrc[c3i];
		tempSrc[c3i] = center
			+ d_dt[0] * (tempSrc[ch_offset + (y - 1) * width + x]
				+ tempSrc[ch_offset + (y + 1) * width + x]
				+ tempSrc[ch_offset + y * width + (x - 1)]
				+ tempSrc[ch_offset + y * width + (x + 1)]
				- d_center_w[0] * center)
			- d_dt[0] * mask_center * (center - src[c3i]);
	}
}

__global__ void PDEHeatDiffusion(float* mask, float* src, float* tempSrc, int width, int height, int ch, int iters) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;
	float center;
	int c3i = 0;
	int ch_offset = 0;
	float mask_center = mask[y * width + x];

	for (int i = 0; i < iters; i++) {
		for (int k = 0; k < ch; k++) {
			ch_offset = k * width * height;
			c3i = ch_offset + y * width + x;
			center = tempSrc[c3i];
			tempSrc[c3i] = center
				+ d_dt[0] * (tempSrc[ch_offset + (y - 1) * width + x]
					+ tempSrc[ch_offset + (y + 1) * width + x]
					+ tempSrc[ch_offset + y * width + (x - 1)]
					+ tempSrc[ch_offset + y * width + (x + 1)]
					- d_center_w[0] * center)
				- d_dt[0] * mask_center * (center - src[c3i]);
		}
	}
}

void hairInpaintingGPU(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info) {
	float* d_normalized_mask;
	float* d_normalized_masked_src;
	float* d_normalized_masked_src_temp;
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyHostToDevice));

	const int smem_size = DATA_TILE_DIM * DATA_TILE_DIM * info.Channels * sizeof(float);
	dim3 block(TILE_DIM, TILE_DIM);
	dim3 grid(iDivUp(info.Width, TILE_DIM), iDivUp(info.Height, TILE_DIM));

	const float h_const_dt = 0.1f;
	const float h_center_w = 4.0f;
	gpuErrorCheck(cudaMemcpyToSymbol(d_dt, &h_const_dt, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_center_w, &h_center_w, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToDevice));

	for (int i = 0; i < 1000; i++) {
		PDEHeatDiffusion << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.Channels);
	}
	gpuErrorCheck(cudaDeviceSynchronize());
	
	float* h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
	dst = h_result;

	gpuErrorCheck(cudaFree(d_normalized_mask));
	gpuErrorCheck(cudaFree(d_normalized_masked_src));
	gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
	gpuErrorCheck(cudaDeviceReset());
}

void hairInpaintingMix(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info) {
	float* d_normalized_mask;
	float* d_normalized_masked_src;
	float* d_normalized_masked_src_temp;
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC2Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC2Elements * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC2Elements * sizeof(float), cudaMemcpyHostToDevice));

	const int smem_size = DATA_TILE_DIM * DATA_TILE_DIM * info.MixGpuChannels * sizeof(float);
	dim3 block(TILE_DIM, TILE_DIM);
	dim3 grid(iDivUp(info.Width, TILE_DIM), iDivUp(info.Height, TILE_DIM));

	const float h_const_dt = info.Dt;
	const float h_center_w = info.Cw;
	gpuErrorCheck(cudaMemcpyToSymbol(d_dt, &h_const_dt, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_center_w, &h_center_w, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC2Elements * sizeof(float), cudaMemcpyDeviceToDevice));

	bool first_process = true;
	bool* is_finish_process = new bool[1]{false};
	float* img_u = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
	std::unique_ptr<std::thread> thread_ptr;
	for (int i = 0; i < info.Iters; i++) {
		PDEHeatDiffusion << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.MixGpuChannels);
		
		// CPU
		if (first_process) {
			thread_ptr = std::unique_ptr<std::thread>(new std::thread(hairInpaintingAsync, normalized_mask, normalized_masked_src, img_u, info, is_finish_process));
			first_process = false;
		}
	}	
	while (true) {
		if (is_finish_process[0]) {
			thread_ptr->join();
			break;
		}
	}
	gpuErrorCheck(cudaDeviceSynchronize());


	float* h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	// R G channels
	gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC2Elements * sizeof(float), cudaMemcpyDeviceToHost));
	// B channels
	memcpy(&h_result[info.MixGpuChannels * info.Width * info.Height], img_u, info.NumberOfC1Elements * sizeof(float));
	dst = h_result;

	gpuErrorCheck(cudaFree(d_normalized_mask));
	gpuErrorCheck(cudaFree(d_normalized_masked_src));
	gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
	gpuErrorCheck(cudaDeviceReset());
}

void hairInpaintingAsync(float* normalized_mask, float* normalized_masked_src, float* dst, HairInpaintInfo info, bool* isFinish) {
	int mix_cpu_channels = info.Channels - info.MixGpuChannels;
	int normalized_masked_src_offset = info.MixGpuChannels * info.Width * info.Height;
	memcpy(dst, &normalized_masked_src[normalized_masked_src_offset], info.NumberOfC1Elements * sizeof(float));
	PDEHeatDiffusionCPU(normalized_mask, &normalized_masked_src[normalized_masked_src_offset], dst, mix_cpu_channels, info);
	std::cout << "finish the cpu process" << std::endl;
	isFinish[0] = true;
}

// main
void hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info, bool isGPU) {
	cv::resize(src, src, cv::Size(info.Width, info.Height));
	cv::resize(mask, mask, cv::Size(info.Width, info.Height));

	float* normalized_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	float* normalized_mask = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
	float* normalized_masked_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	normalizeImage(src, mask, normalized_src, normalized_mask, normalized_masked_src, true);

	float* h_dst_array = nullptr; // 3 channels
	float* h_result_temp = (float*)malloc(info.NumberOfC3Elements * sizeof(float));;
	if (true) {
		hairInpaintingGPU(normalized_mask, normalized_masked_src, h_dst_array, info);
		//hairInpaintingMix(normalized_mask, normalized_masked_src, h_dst_array, info);
	}
	else{
		hairInpaintingCPU(normalized_mask, normalized_masked_src, h_dst_array, info);
	}

	memcpy(h_result_temp, h_dst_array, info.NumberOfC3Elements * sizeof(float));
	convertToMatArrayFormat(h_result_temp, h_dst_array, info);
	cv::Mat dst_mat(info.Height, info.Width, CV_32FC3, h_dst_array);
	cv::resize(dst_mat, dst_mat, cv::Size(info.Width * info.Rescale, info.Height * info.Rescale));
	dst = dst_mat;

	free(h_result_temp);
	free(normalized_src);
	free(normalized_mask);
	free(normalized_masked_src);
}