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
#if L3_TIMER
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	auto t0 = getTime();
#endif
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

	for (int i = 0; i < info.Iters; i++) {
		PDEHeatDiffusion << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.Channels);
	}
	gpuErrorCheck(cudaDeviceSynchronize());
#if L3_TIMER
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	auto t1 = getTime();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printTime(t0, t1, "hairInpaintingOnlyGPU");
	std::cout << "PDEHeatDiffusionGPU: " << milliseconds << std::endl;
#endif
	float* h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC3Elements * sizeof(float), cudaMemcpyDeviceToHost));
	dst = h_result;

	gpuErrorCheck(cudaFree(d_normalized_mask));
	gpuErrorCheck(cudaFree(d_normalized_masked_src));
	gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
	gpuErrorCheck(cudaDeviceReset());
}

void hairInpaintingMix(float* normalized_mask, float* normalized_masked_src, float*& dst, HairInpaintInfo info) {
#if L3_TIMER
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	const float h_const_dt = info.Dt;
	const float h_center_w = info.Cw;
	const int smem_size = DATA_TILE_DIM * DATA_TILE_DIM * info.MixGpuChannels * sizeof(float);
	bool first_process = true;
	bool* is_finish_process = new bool[1]{ false };
	float* h_result = nullptr;
	float* d_normalized_mask;
	float* d_normalized_masked_src;
	float* d_normalized_masked_src_temp;
	float* img_u = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
	std::unique_ptr<std::thread> thread_ptr;

	gpuErrorCheck(cudaMemcpyToSymbol(d_dt, &h_const_dt, 1 * sizeof(float)));
	gpuErrorCheck(cudaMemcpyToSymbol(d_center_w, &h_center_w, 1 * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_mask, info.NumberOfC1Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src, info.NumberOfC2Elements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_normalized_masked_src_temp, info.NumberOfC2Elements * sizeof(float)));
	gpuErrorCheck(cudaMemcpy(d_normalized_mask, normalized_mask, info.NumberOfC1Elements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src, normalized_masked_src, info.NumberOfC2Elements * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_normalized_masked_src_temp, d_normalized_masked_src, info.NumberOfC2Elements * sizeof(float), cudaMemcpyDeviceToDevice));

	dim3 block(TILE_DIM, TILE_DIM);
	dim3 grid(iDivUp(info.Width, TILE_DIM), iDivUp(info.Height, TILE_DIM));
	for (int i = 0; i < info.Iters; i++) {
		PDEHeatDiffusion << <grid, block >> > (d_normalized_mask, d_normalized_masked_src, d_normalized_masked_src_temp, info.Width, info.Height, info.MixGpuChannels);
		
		// new thread1 to run CPU hairInpainting
		if (first_process) {
			thread_ptr = std::unique_ptr<std::thread>(new std::thread(hairInpaintingAsync, normalized_mask, normalized_masked_src, img_u, info, is_finish_process));
			first_process = false;
		}
	}
	
	h_result = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	//std::cout << "main thread copy RG channels: " << std::endl;
	gpuErrorCheck(cudaMemcpy(h_result, d_normalized_masked_src_temp, info.NumberOfC2Elements * sizeof(float), cudaMemcpyDeviceToHost));
	
	// main thread wait thread1 finish
	while (true) {
		if (is_finish_process[0]) {
			thread_ptr->join();
			//std::cout << "main thread copy B channels: " << std::endl;
			memcpy(&h_result[info.MixGpuChannels * info.Width * info.Height], img_u, info.NumberOfC1Elements * sizeof(float));
			break;
		}
	}
	dst = h_result;
#if L3_TIMER
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "PDEHeatDiffusionGPU: " << milliseconds << std::endl;
#endif
	
	gpuErrorCheck(cudaFree(d_normalized_mask));
	gpuErrorCheck(cudaFree(d_normalized_masked_src));
	gpuErrorCheck(cudaFree(d_normalized_masked_src_temp));
	gpuErrorCheck(cudaDeviceReset());
}

void hairInpaintingAsync(float* normalized_mask, float* normalized_masked_src, float* dst, HairInpaintInfo info, bool* isFinish) {
#if L3_TIMER
	auto t0 = getTime();
#endif
	int mix_cpu_channels = info.Channels - info.MixGpuChannels;
	int normalized_masked_src_offset = info.MixGpuChannels * info.Width * info.Height;
	memcpy(dst, &normalized_masked_src[normalized_masked_src_offset], info.NumberOfC1Elements * sizeof(float));
	PDEHeatDiffusionCPU(normalized_mask, &normalized_masked_src[normalized_masked_src_offset], dst, mix_cpu_channels, info);
#if L3_TIMER
	auto t1 = getTime();
	printTime(t0, t1, "hairInpaintingAsync");
#endif
	isFinish[0] = true;
}

void hairInpainting(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info, bool isGPU) {
#if L3_TIMER
	auto t0 = getTime();
#endif
	cv::resize(src, src, cv::Size(info.Width, info.Height));
	cv::resize(mask, mask, cv::Size(info.Width, info.Height));
#if L3_TIMER
	auto t1 = getTime();
#endif
	float* normalized_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	float* normalized_mask = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
	float* normalized_masked_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	normalizeImage(src, mask, normalized_src, normalized_mask, normalized_masked_src);
#if L3_TIMER
	auto t2 = getTime();
#endif
	float* h_dst_array = nullptr;
	uchar* h_dst_RGB_array = (uchar*)malloc(info.NumberOfC3Elements * sizeof(uchar));
	if (isGPU) {
		hairInpaintingGPU(normalized_mask, normalized_masked_src, h_dst_array, info);
		//hairInpaintingMix(normalized_mask, normalized_masked_src, h_dst_array, info);
	}
	else{
		hairInpaintingCPU(normalized_mask, normalized_masked_src, h_dst_array, info);
	}
#if L3_TIMER
	auto t3 = getTime();
#endif
	convertToMatArrayFormat(h_dst_array, h_dst_RGB_array, info);
	cv::Mat dst_mat(info.Height, info.Width, CV_8UC3, h_dst_RGB_array);
	cv::resize(dst_mat, dst_mat, cv::Size(info.Width * info.Rescale, info.Height * info.Rescale));
	dst = dst_mat;
#if L3_TIMER
	auto t4 = getTime();
	printTime(t0, t1, "resize1");
	printTime(t1, t2, "normalizeImage");
	printTime(t2, t3, "main inpainting");
	printTime(t3, t4, "resize2");
#endif
	free(normalized_src);
	free(normalized_mask);
	free(normalized_masked_src);
}