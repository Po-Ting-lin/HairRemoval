#include "entropy_thresholding.cuh"

#define TILE_DIM 32
#define EPSILON 1e-8
#define NUM_STREAMS 15

__global__ void preSumXMatrixKernel(float* src, int nx, int raw_width, int new_width) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int diff = raw_width - new_width;
    if (x < raw_width && y < raw_width) {
        if ((x < new_width) && (x >= new_width - diff)) {
            src[y * nx + x] += src[y * nx + x + diff];
        }
    }
}

__global__ void preSumYMatrixKernel(float* src, int nx, int raw_width, int new_width) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int diff = raw_width - new_width;
    if (x < raw_width && y < raw_width) {
        if ((x < new_width && y < new_width) && (y >= new_width - diff)) {
            src[y * nx + x] += src[(y + diff) * nx + x];
        }
    }
}

__global__ void sumMatirxKernel(float* src, int nx, int multiple_width, float* sum) {
    __shared__ float smem[TILE_DIM][TILE_DIM];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < multiple_width && y < multiple_width) {
        if (multiple_width < blockDim.x) {
            if (threadIdx.x >= multiple_width || threadIdx.y >= multiple_width) {
                smem[threadIdx.y][threadIdx.x] = 0.0f;
            }
            else {
                smem[threadIdx.y][threadIdx.x] = src[y * nx + x];
            }
        }
        else {
            smem[threadIdx.y][threadIdx.x] = src[y * nx + x];
        }

        __syncthreads();

        for (int offx = blockDim.x / 2; offx > 0; offx /= 2) {
            if (threadIdx.x < offx) {
                smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y][threadIdx.x + offx];

                __syncthreads();

                if (threadIdx.y < offx) {
                    smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y + offx][threadIdx.x];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            sum[blockIdx.y * gridDim.x + blockIdx.x] = smem[threadIdx.y][threadIdx.x];
            //printf("x: %d, y: %d -- %f\n", blockIdx.x, blockIdx.y, sum[blockIdx.y * gridDim.x + blockIdx.x]);
        }
    }
}

__global__ void sumSumMatrixKernel(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold) {
    __shared__ float smem[2 * TILE_DIM];
    int tid = threadIdx.x;

    // put the data in that block from DRAM to shared memory
    if (tid < sum_matrix_size) {
        smem[tid] = sum_matrix[tid];
    }
    else {
        smem[tid] = 0.0f;
    }
    __syncthreads(); // important

    // unrolling warp
    if (tid < 32) {
        volatile float* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        d_pA[threshold] = (float)(smem[0]);
    }
}

__global__ void dividePArrayKernel(float* d_p, float* d_m, int size) {
    int tid = threadIdx.x;
    if (tid < size) {
        d_m[tid] = __fdividef(d_m[tid], d_p[tid]);
    }
}

__global__ void multiplyRCKernel(float* d_data_rc, float* d_data, int nx, bool reversed) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int cx = reversed ? nx - x - 1 : x;
    int cy = reversed ? nx - y - 1 : y;

    if (x < nx && y < nx) {
        d_data_rc[y * nx + x] = d_data[y * nx + x] * cx * cy;
    }
}

__global__ void computeEntropyMatrixKernel(float* d_data_computed, float* d_data, int nx, float* d_mA, int threshold, bool reversed) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int cx = reversed ? nx - x - 1 : x;
    int cy = reversed ? nx - y - 1 : y;

    if (x < nx && y < nx) {
        float meanA = d_mA[threshold];
        float p = d_data[y * nx + x];
        float value = p * cx * cy * log2f(((float)cx * cy + EPSILON) / (meanA + EPSILON));
        value += meanA * p * log2f(meanA / ((float)cx + EPSILON) / ((float)cy + EPSILON) + EPSILON);
        d_data_computed[y * nx + x] = value;
    }
}

__global__ void reversedDataKernel(float* d_data, float* d_reversed_data, int nx) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int rx = nx - x - 1;
    int ry = nx - y - 1;

    if (x < nx && y < nx) {
        d_reversed_data[ry * nx + rx] = d_data[y * nx + x];
    }
}

int entropyThesholdingGPU(cv::Mat& glcm) {
    int dynamic_range = 256;
    float
        * h_data,
        * h_reversed_data,
        * h_eA,
        * h_eC,
        * h_AC;

    float
        * d_data,
        * d_reversed_data,
        * d_pA,
        * d_mA,
        * d_eA,
        * d_pC,
        * d_mC,
        * d_eC;

    auto t1 = std::chrono::system_clock::now();

    // host 
    h_data = (float*)glcm.data;
    h_reversed_data = (float*)malloc(dynamic_range * dynamic_range * sizeof(float*));
    h_eA = (float*)malloc(dynamic_range * sizeof(float*));
    h_eC = (float*)malloc(dynamic_range * sizeof(float*));
    h_AC = (float*)malloc(dynamic_range * sizeof(float*));

    for (int i = 0, int j = dynamic_range * dynamic_range - 1; i < dynamic_range * dynamic_range; i++, j--) {
        h_reversed_data[j] = h_data[i];
    }

    auto t2 = std::chrono::system_clock::now();

    // device
    gpuErrorCheck(cudaMalloc((void**)&d_data, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_reversed_data, dynamic_range * dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eA, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_pC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_mC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_eC, dynamic_range * sizeof(float)));
    gpuErrorCheck(cudaMemcpy(d_data, h_data, dynamic_range * dynamic_range * sizeof(float), cudaMemcpyHostToDevice));

    auto t3 = std::chrono::system_clock::now();

    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    entropyCPU(h_data, h_eA, dynamic_range, false);
    entropyCPU(h_reversed_data, h_eC, dynamic_range, true);

    auto t35 = std::chrono::system_clock::now();

    bool reversed = false;
    getAreaArray(d_data, dynamic_range, d_pA, stream);
    getMeanArray(d_data, dynamic_range, d_pA, d_mA, reversed, stream);
    getEntropyArray(d_data, dynamic_range, d_mA, d_eA, reversed, stream);

    reversed = true;
    reversedData(d_data, d_reversed_data, dynamic_range);
    getAreaArray(d_reversed_data, dynamic_range, d_pC, stream);
    getMeanArray(d_reversed_data, dynamic_range, d_pC, d_mC, reversed, stream);
    getEntropyArray(d_reversed_data, dynamic_range, d_mC, d_eC, reversed, stream);
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    auto t4 = std::chrono::system_clock::now();

    gpuErrorCheck(cudaMemcpy(h_eA, d_eA, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(h_eC, d_eC, dynamic_range * sizeof(float), cudaMemcpyDeviceToHost));

    auto t5 = std::chrono::system_clock::now();

    float min_value = FLT_MAX;
    int min_t = -1;
    for (int i = 0, int j = dynamic_range - 1; i < dynamic_range; i++, j--) {
        h_AC[i] = h_eA[i] + h_eC[j];
        //printf("i: %d, A:%f, C: %f, AC: %f\n", i, h_eA[i], h_eC[j], h_AC[i]);
        if (h_AC[i] < min_value) {
            min_t = i;
            min_value = h_AC[i];
        }
    }

    auto t6 = std::chrono::system_clock::now();

    //printTime(t1, t2, "make reverse data");
    //printTime(t2, t3, "H to D");
    //printTime(t3, t35, "CPU entropy");
    //printTime(t35, t4, "kerenl");
    //printTime(t4, t5, "D to H");
    //printTime(t5, t6, "conbine");

    //printf("min threshold : %d\n", min_t);

    gpuErrorCheck(cudaFree(d_data));
    gpuErrorCheck(cudaFree(d_reversed_data));
    gpuErrorCheck(cudaFree(d_pA));
    gpuErrorCheck(cudaFree(d_mA));
    gpuErrorCheck(cudaFree(d_eA));
    gpuErrorCheck(cudaFree(d_pC));
    gpuErrorCheck(cudaFree(d_mC));
    gpuErrorCheck(cudaFree(d_eC));
    free(h_reversed_data);
    free(h_eA);
    free(h_eC);
    free(h_AC);
    gpuErrorCheck(cudaDeviceReset());
    return min_t;
}

void reversedData(float* d_data, float* d_reversed_data, int full_width) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
    reversedDataKernel << <grid, block >> > (d_data, d_reversed_data, full_width);
    gpuErrorCheck(cudaDeviceSynchronize());
}

void getAreaArray(float* d_data, int full_width, float* d_pA, cudaStream_t* stream) {
    float* d_buf, * d_sum_matrix;
    int sum_matirx_size = 2 * TILE_DIM;
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * full_width * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * full_width * sizeof(float)));

    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int idx = i % NUM_STREAMS;
        int buf_offset = i * full_width * full_width;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = getClosedWidth(raw_width);
        gpuErrorCheck(cudaMemcpyAsync(&d_buf[buf_offset], d_data, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice, stream[idx]));
        sumMatrixStream(&d_buf[buf_offset], d_pA, &d_sum_matrix[sum_matrix_offset], full_width, raw_width, multiple_width, i, stream[idx]);
    }

    gpuErrorCheck(cudaStreamSynchronize(*stream));
    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void getMeanArray(float* d_data, int full_width, float* d_pA, float* d_mA, bool reversed, cudaStream_t* stream) {
    float* d_buf, * d_data_rc, * d_sum_matrix;
    int sum_matirx_size = 2 * TILE_DIM;
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));
    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_data_rc, full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * full_width * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * full_width * sizeof(float)));

    // r * c * element
    multiplyRCKernel << <rc_grid, rc_block >> > (d_data_rc, d_data, full_width, reversed);
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int idx = i % NUM_STREAMS;
        int buf_offset = i * full_width * full_width;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = getClosedWidth(raw_width);
        gpuErrorCheck(cudaMemcpyAsync(&d_buf[buf_offset], d_data_rc, full_width * full_width * sizeof(float), cudaMemcpyDeviceToDevice, stream[idx]));
        sumMatrixStream(&d_buf[buf_offset], d_mA, &d_sum_matrix[sum_matrix_offset], full_width, raw_width, multiple_width, i, stream[idx]);
    }
    gpuErrorCheck(cudaStreamSynchronize(*stream));

    // divide area
    dividePArrayKernel << <1, full_width >> > (d_pA, d_mA, full_width);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_data_rc));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void getEntropyArray(float* d_data, int full_width, float* d_mA, float* d_eA, bool reversed, cudaStream_t* stream) {
    float* d_buf, * d_sum_matrix;
    int sum_matirx_size = 2 * TILE_DIM;
    dim3 rc_block(TILE_DIM, TILE_DIM);
    dim3 rc_grid(iDivUp(full_width, TILE_DIM), iDivUp(full_width, TILE_DIM));

    gpuErrorCheck(cudaMalloc((void**)&d_buf, full_width * full_width * full_width * sizeof(float)));
    gpuErrorCheck(cudaMalloc((void**)&d_sum_matrix, sum_matirx_size * full_width * sizeof(float)));
    gpuErrorCheck(cudaMemset(d_sum_matrix, 0.0f, sum_matirx_size * full_width * sizeof(float)));

    for (int i = TILE_DIM - 1; i < full_width; i++) {
        int raw_width = i + 1;
        int idx = i % NUM_STREAMS;
        int buf_offset = i * full_width * full_width;
        int sum_matrix_offset = i * sum_matirx_size;
        int multiple_width = getClosedWidth(raw_width);
        computeEntropyMatrixKernel << <rc_grid, rc_block, 0, stream[idx] >> > (&d_buf[buf_offset], d_data, full_width, d_mA, i, reversed);
        sumMatrixStream(&d_buf[buf_offset], d_eA, &d_sum_matrix[sum_matrix_offset], full_width, raw_width, multiple_width, i, stream[idx]);
    }
    gpuErrorCheck(cudaStreamSynchronize(*stream));

    gpuErrorCheck(cudaFree(d_buf));
    gpuErrorCheck(cudaFree(d_sum_matrix));
}

void sumMatrixStream(float* d_buf, float* d_arr, float* d_sum_matrix, int full_width, int raw_width, int multiple_width, int threshold, cudaStream_t stream) {
    dim3 pre_sum_block(TILE_DIM, TILE_DIM);
    dim3 pre_sum_grid(iDivUp(raw_width, TILE_DIM), iDivUp(raw_width, TILE_DIM));
    dim3 sum_block(TILE_DIM, TILE_DIM);
    dim3 sum_grid(iDivUp(multiple_width, TILE_DIM), iDivUp(multiple_width, TILE_DIM));

    if (raw_width != multiple_width) {
        preSumXMatrixKernel << <pre_sum_grid, pre_sum_block, 0, stream >> > (d_buf, full_width, raw_width, multiple_width);
        preSumYMatrixKernel << <pre_sum_grid, pre_sum_block, 0, stream >> > (d_buf, full_width, raw_width, multiple_width);
    }
    sumMatirxKernel << <sum_grid, sum_block, TILE_DIM * (TILE_DIM) * sizeof(float), stream >> > (d_buf, full_width, multiple_width, d_sum_matrix);
    sumSumMatrixKernel << <1, 2 * TILE_DIM, 2 * TILE_DIM * sizeof(float), stream >> > (d_sum_matrix, d_arr, sum_grid.x * sum_grid.x, threshold);
}

void entropyCPU(float* h_data, float* h_e, int width, bool reversed) {
#pragma omp parallel for
    for (int threshold = 0; threshold < TILE_DIM - 1; threshold++) {
        const int cols = width;
        float p = 0.0f;
        float mean = 0.0f;
        float entropy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;

        // pA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                p += h_data[r * cols + c];
            }
        }

        // meanA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                cx = reversed ? width - c - 1 : c;
                cy = reversed ? width - r - 1 : r;
                mean += cx * cy * h_data[r * cols + c];
            }
        }
        mean /= p;

        // entropyA
        for (int r = 0; r < threshold + 1; r++) {
            for (int c = 0; c < threshold + 1; c++) {
                cx = reversed ? width - c - 1 : c;
                cy = reversed ? width - r - 1 : r;
                entropy += cy * cx * h_data[r * cols + c] * log2((cy * cx + EPSILON) / (mean + EPSILON));
                entropy += mean * h_data[r * cols + c] * log2(mean / (cy + EPSILON) / (cx + EPSILON) + EPSILON);
            }
        }
        h_e[threshold] = entropy;
    }
}