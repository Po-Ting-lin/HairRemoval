#include "hairRemovalEngine.cuh"

__global__ void extractLChannelKernel(uchar* src, float* dst, int nx, int ny, int nz) {
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

__global__ void pdeHeatDiffusionSMEMKernel(float* mask, float* src, float* dst, int width, int height) {
    __shared__ float smem[(BlockDim_x + 2) * Step][(BlockDim_y + 2) * Step];
    const int x = blockIdx.x * Step * BlockDim_x + threadIdx.x - BlockDim_x;
    const int y = blockIdx.y * Step * BlockDim_y + threadIdx.y - BlockDim_y;;

    // locate at each block (a thread block map into src pointer)
    dst += y * width + x;
    src += y * width + x;

    // put into active space
    for (int yy = 1; yy < 1 + Step; yy++) {
        for (int xx = 1; xx < 1 + Step; xx++) {
            smem[yy * BlockDim_y + threadIdx.y][xx * BlockDim_x + threadIdx.x]
                = dst[yy * BlockDim_y * width + xx * BlockDim_x];
        }
    }

    // corner space
    smem[threadIdx.y][threadIdx.x] = dst[1 * BlockDim_y * width + 1 * BlockDim_x];
    smem[threadIdx.y][(1 + Step) * BlockDim_x + threadIdx.x] = dst[1 * BlockDim_y * width + Step * BlockDim_x];
    smem[(1 + Step) * BlockDim_y + threadIdx.y][threadIdx.x] = dst[Step * BlockDim_y * width + 1 * BlockDim_x];
    smem[(1 + Step) * BlockDim_y + threadIdx.y][(1 + Step) * BlockDim_x + threadIdx.x] = dst[Step * BlockDim_y * width + Step * BlockDim_x];

    // put into left space
    for (int yy = 1; yy < Step + 1; yy++) {
        //if (y < height - BlockDim_y * (1 + Step) && y >= 0)
        //    printf("%d - %d\n", y + yy * BlockDim_y, x);
        smem[yy * BlockDim_y + threadIdx.y][threadIdx.x] 
            = (x >= 0) ? dst[yy * BlockDim_y * width] : 0;
    }

    // put into right space
    for (int yy = 1; yy < Step + 1; yy++) {
        smem[yy * BlockDim_y + threadIdx.y][(1 + Step) * BlockDim_x + threadIdx.x]
            = (x < width - (1 + Step) * BlockDim_x) ? dst[yy * BlockDim_y * width + (1 + Step) * BlockDim_x] : 0;
    }

    // put into top space
    for (int xx = 1; xx < Step + 1; xx++) {
        smem[threadIdx.y][xx * BlockDim_x + threadIdx.x]
            = (y >= 0) ? dst[xx * BlockDim_x] : 0;
    }

    // put into bottom space
    for (int xx = 1; xx < Step + 1; xx++) {
        smem[(1 + Step) * BlockDim_y + threadIdx.y][xx * BlockDim_x + threadIdx.x]
            = (y < height - (1 + Step) * BlockDim_y) ? dst[(1 + Step) * BlockDim_y * width + xx * BlockDim_x] : 0;
    }
    __syncthreads();


    for (int yy = 1; yy < 1 + Step; yy++) {
        for (int xx = 1; xx < 1 + Step; xx++) {
            int index = yy * BlockDim_y * width + xx * BlockDim_x;
            float center = smem[yy * BlockDim_y + threadIdx.y][xx * BlockDim_x + threadIdx.x];
            dst[index] =
                center + 0.2f * (
                  smem[yy * BlockDim_y + threadIdx.y + 1][xx * BlockDim_x + threadIdx.x]
                + smem[yy * BlockDim_y + threadIdx.y - 1][xx * BlockDim_x + threadIdx.x]
                + smem[yy * BlockDim_y + threadIdx.y][xx * BlockDim_x + threadIdx.x + 1]
                + smem[yy * BlockDim_y + threadIdx.y][xx * BlockDim_x + threadIdx.x - 1]
                - 4.0f * center)
                - 0.2f * mask[index] * (center - src[index]);
        }
    }
}

__global__ void pdeHeatDiffusionKernel(float* mask, float* src, float* tempSrc, int width, int height, int ch) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x < 0 || y < 0 || x >= width || y >= height || z >= ch) return;

    src += z * width * height;
    tempSrc += z * width * height;
    float center = tempSrc[y * width + x];
    int i = y * width + x;
    tempSrc[i] = center
        + 0.2f 
        * (tempSrc[max(0, y - 1) * width + x]
        + tempSrc[min(height - 1, y + 1) * width + x]
        + tempSrc[y * width + max(0, x - 1)]
        + tempSrc[y * width + min(width - 1, x + 1)]
        - 4.0f * center)
        - 0.2f * mask[i] * (center - src[i]);
}
