#include "hairRemovalEngine.cuh"

__global__ void extractLChannelWithInstrinicFunction(uchar* src, float* dst, int nx, int ny, int nz) {
    int x = threadIdx.x + TILE_DIM * blockIdx.x;
    int y = threadIdx.y + TILE_DIM * blockIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_DIM) {
        // take pixel from DRAM
        uchar R = *(src + ((y + i) * nx * nz) + (x * nz) + 0);
        uchar G = *(src + ((y + i) * nx * nz) + (x * nz) + 1);
        uchar B = *(src + ((y + i) * nx * nz) + (x * nz) + 2);

        // RGB to XYZ
        float r = fdividef((float)R, 255.0f);
        float g = fdividef((float)G, 255.0f);
        float b = fdividef((float)B, 255.0f);
        r = ((r > 0.04045f) ? __powf(fdividef(r + 0.055f, 1.055f), 2.4f) : fdividef(r, 12.92f)) * 100.0f;
        g = ((g > 0.04045f) ? __powf(fdividef(g + 0.055f, 1.055f), 2.4f) : fdividef(g, 12.92f)) * 100.0f;
        b = ((b > 0.04045f) ? __powf(fdividef(b + 0.055f, 1.055f), 2.4f) : fdividef(b, 12.92f)) * 100.0f;

        // XYZ to LAB
        float Y = fdividef(0.2126f * r + 0.7152f * g + 0.0722f * b, 100.0f);
        Y = (Y > 0.008856f) ? cbrtf(Y) : fmaf(7.787f, Y, 0.1379f);
        float L = fmaf(116.0f, Y, -16.0f) * 2.55f;

        // set pixel to DRAM
        *(dst + (y + i) * nx + x) = L;
    }
}

__global__ void padDataClampToBorderKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int dataH, int dataW, int kernelH, int kernelW, int kernelY, int kernelX) {
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;

    if (y < fftH && x < fftW) {
        int dy, dx;
        if (y < dataH) {
            dy = y;
        }
        if (x < dataW) {
            dx = x;
        }
        if (y >= dataH && y < borderH) {
            dy = dataH - 1;
        }
        if (x >= dataW && x < borderW) {
            dx = dataW - 1;
        }
        if (y >= borderH) {
            dy = 0;
        }
        if (x >= borderW) {
            dx = 0;
        }
        d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
    }
}

__global__ void padKernelKernel(float* d_Dst, float* d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX) {
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (y < kernelH && x < kernelW)
    {
        int ky = y - kernelY;
        if (ky < 0)
        {
            ky += fftH;
        }
        int kx = x - kernelX;
        if (kx < 0)
        {
            kx += fftW;
        }
        d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
    }
}

__global__ void modulateAndNormalizeKernel(fComplex* d_Dst, fComplex* d_DataSrc, fComplex* d_KernelSrc, int dataSize, float c) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= dataSize) return;
    fComplex a = d_KernelSrc[i];
    fComplex b = d_DataSrc[i];
    fComplex d;
    mulAndScaleModified(a, b, c, d);
    d_Dst[i] = d;
}

__global__ void cubeReductionKernel(float* d_Src, uchar* d_Dst, int fftH, int fftW, int dataH, int dataW, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dataW && y < dataH) {
        float max_value = 0.0f;
        float current_value = 0.0f;
        int offset = 0;
        for (int i = 0; i < depth; i++) {
            offset = i * fftH * fftW;
            current_value = d_Src[y * fftW + x + offset];
            if (current_value > max_value) {
                max_value = current_value;
            }
        }
        d_Dst[y * dataW + x] = (uchar)min(max_value, 255.0f);
    }
}

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

__global__ void sumMatirxKernel(float* src, int nx, int multiple_width, float* d_sum_matrix) {
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
            d_sum_matrix[blockIdx.y * gridDim.x + blockIdx.x] = smem[threadIdx.y][threadIdx.x];
            //printf("x: %d, y: %d -- %f\n", blockIdx.x, blockIdx.y, sum[blockIdx.y * gridDim.x + blockIdx.x]);
        }
    }
}

__global__ void sumSumMatrixKernel(float* sum_matrix, float* d_pA, int sum_matrix_size, int threshold) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // put the data in that block from DRAM to shared memory
    if (tid < sum_matrix_size) {
        smem[tid] = sum_matrix[tid];
    }
    else {
        smem[tid] = 0.0f;
    }
    __syncthreads();

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

__global__ void multiplyRCKernel(float* d_data_rc, float* d_data, int nx, bool reversed) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int cx = reversed ? nx - x - 1 : x;
    int cy = reversed ? nx - y - 1 : y;

    if (x < nx && y < nx) {
        d_data_rc[y * nx + x] = d_data[y * nx + x] * cx * cy;
    }
}

__global__ void dividePArrayKernel(float* d_p, float* d_m, int size) {
    int tid = threadIdx.x;
    if (tid < size) {
        d_m[tid] = __fdividef(d_m[tid], d_p[tid]);
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

__global__ void pdeHeatDiffusionSMEM(float* mask, float* src, float* tempSrc, int width, int height, int ch) {
    __shared__ float smem[TILE_DIM][TILE_DIM];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x < 0 || y < 0 || x >= width || y >= height) return;

    int c3i = z * width * height + y * width + x;
    smem[threadIdx.y][threadIdx.x] = tempSrc[c3i];

    __syncthreads();

    //if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    float center = smem[threadIdx.y][threadIdx.x];

    float tmp = 0.0f;
    float count = 0.0f;

    if (threadIdx.y - 1 > -1) {
        tmp += smem[threadIdx.y - 1][threadIdx.x]; count++;
    }
    if (threadIdx.y + 1 < TILE_DIM) {
        tmp += smem[threadIdx.y + 1][threadIdx.x]; count++;
    }
    if (threadIdx.x - 1 > -1) {
        tmp += smem[threadIdx.y][threadIdx.x - 1]; count++;
    }
    if (threadIdx.x + 1 < TILE_DIM) {
        tmp += smem[threadIdx.y][threadIdx.x + 1]; count++;
    }
    tempSrc[c3i] = fmaf(-0.2f, fmaf(count, center, fmaf(mask[y * width + x], (center - src[c3i]), -tmp)), center);
}

__global__ void pdeHeatDiffusionSMEM2(float* mask, float* src, float* tempSrc, int width, int height, int ch) {
    __shared__ float smem[TILE_DIM2][TILE_DIM2];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x < 0 || y < 0 || x >= width || y >= height) return;

    int c3i = z * width * height + y * width + x;
    smem[threadIdx.y][threadIdx.x] = tempSrc[c3i];

    __syncthreads();

    //if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    float center = smem[threadIdx.y][threadIdx.x];
    float tmp = 0.0f;
    float count = 0.0f;

    if (threadIdx.y - 1 > -1) {
        tmp = tmp + smem[threadIdx.y - 1][threadIdx.x]; count++;
    }
    if (threadIdx.y + 1 < TILE_DIM2) {
        tmp = tmp + smem[threadIdx.y + 1][threadIdx.x]; count++;
    }
    if (threadIdx.x - 1 > -1) {
        tmp = tmp + smem[threadIdx.y][threadIdx.x - 1]; count++;
    }
    if (threadIdx.x + 1 < TILE_DIM2) {
        tmp = tmp + smem[threadIdx.y][threadIdx.x + 1]; count++;
    }
    tempSrc[c3i] = fmaf(-0.2f, fmaf(count, center, fmaf(mask[y * width + x], (center - src[c3i]), -tmp)), center);
}


__global__ void pdeHeatDiffusion(float* mask, float* src, float* tempSrc, int width, int height, int ch) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1 || z >= ch) return;

    int ch_offset = z * width * height;
    int c3i = ch_offset + y * width + x;
    float center = tempSrc[c3i];
    tempSrc[c3i] = center
        + d_dt[0] * (tempSrc[ch_offset + (y - 1) * width + x]
            + tempSrc[ch_offset + (y + 1) * width + x]
            + tempSrc[ch_offset + y * width + (x - 1)]
            + tempSrc[ch_offset + y * width + (x + 1)]
            - d_center_w[0] * center)
        - d_dt[0] * mask[y * width + x] * (center - src[c3i]);
}
