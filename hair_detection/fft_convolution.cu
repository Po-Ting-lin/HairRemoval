#include "fft_convolution.cuh"

#define PI 3.14159
#define BLOCKDIM 32

#define LOAD_FLOAT(i) d_Src[i]
#define LOAD_FCOMPLEX(i) d_Src[i]
#define LOAD_FCOMPLEX_A(i) d_SrcA[i]
#define LOAD_FCOMPLEX_B(i) d_SrcB[i]


// Rounding up the FFT dimensions to the next power of 2,
// unless the dimension would exceed 1024, 
// in which case it's rounded up to the next multiple of 512.
int snapTransformSize(int dataSize)
{
    int hiBit;
    unsigned int lowPOT, hiPOT;

    dataSize = iAlignUp(dataSize, 16);
    for (hiBit = 31; hiBit >= 0; hiBit--)
        if (dataSize & (1U << hiBit))
        {
            break;
        }
    lowPOT = 1U << hiBit;
    if (lowPOT == (unsigned int)dataSize)
    {
        return dataSize;
    }
    hiPOT = 1U << (hiBit + 1);
    if (hiPOT <= 1024)
    {
        return hiPOT;
    }
    else
    {
        return iAlignUp(dataSize, 512);
    }
}

float getRand(void)
{
    return (float)(rand() % 16);
}

__global__ void padKernelKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
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

__global__ void padDataClampToBorderKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;

    if (y < fftH && x < fftW)
    {
        int dy, dx;

        if (y < dataH)
        {
            dy = y;
        }

        if (x < dataW)
        {
            dx = x;
        }

        if (y >= dataH && y < borderH)
        {
            dy = dataH - 1;
        }

        if (x >= dataW && x < borderW)
        {
            dx = dataW - 1;
        }

        if (y >= borderH)
        {
            dy = 0;
        }

        if (x >= borderW)
        {
            dx = 0;
        }

        d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
    }
}

__global__ void modulateAndNormalizeKernel(
    fComplex* d_Dst,
    fComplex* d_DataSrc,
    fComplex* d_KernelSrc,
    int dataSize,
    float c
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= dataSize) return;

    fComplex a = d_KernelSrc[i];
    fComplex b = d_DataSrc[i];
    fComplex d;

    //mulAndScale(a, b, c);
    mulAndScaleModified(a, b, c, d);

    //d_Dst[i] = a;
    d_Dst[i] = d;
}

__global__ void cubeReductionKernel(
    float* d_Src,
    uchar* d_Dst,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int depth
) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dataW && y < dataH) {
        float max_value = 0.0f;
        float current_value = 0.0f;
        int offset = 0;
        for (int i = 0; i < depth; i++) {
            offset = i * fftH * fftW;
            current_value = d_Src[y * fftW + x + offset]; // load from DRAM
            if (current_value > max_value) {
                max_value = current_value;
            }
        }
        d_Dst[y * dataW + x] = (uchar)min(max_value, 255.0f); // write into DRAM
    }
}

void padKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 block(BLOCKDIM, BLOCKDIM);
    dim3 grid(iDivUp(kernelW, block.x), iDivUp(kernelH, block.y));
    padKernelKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
        );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
    cudaDeviceSynchronize();
}

void padDataClampToBorder(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 block(BLOCKDIM, BLOCKDIM);
    dim3 grid(iDivUp(fftW, block.x), iDivUp(fftH, block.y));
    padDataClampToBorderKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
        );
    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

void convolutionClampToBorderCPU(
    float* h_Result,
    float* h_Data,
    float* h_Kernel,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    for (int y = 0; y < dataH; y++)
        for (int x = 0; x < dataW; x++)
        {
            double sum = 0;

            for (int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for (int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++)
                {
                    int dy = y + ky;
                    int dx = x + kx;

                    if (dy < 0) dy = 0;

                    if (dx < 0) dx = 0;

                    if (dy >= dataH) dy = dataH - 1;

                    if (dx >= dataW) dx = dataW - 1;

                    sum += h_Data[dy * dataW + dx] * h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)];
                }

            h_Result[y * dataW + x] = (float)sum;
        }
}

float* GaborFilterCube(HairDetectionParameters para) {
    float* output = new float[para.kernelW * para.kernelH * para.numberOfFilter];
    float* output_ptr = output;
    for (int curNum = 0; curNum < para.numberOfFilter; curNum++) {
        float theta = (float)CV_PI / para.numberOfFilter * curNum;
        for (int y = -para.kernelRadius; y < para.kernelRadius + 1; y++) {
            for (int x = -para.kernelRadius; x < para.kernelRadius + 1; x++, output_ptr++) {
                float xx = x;
                float yy = y;
                float xp = xx * cos(theta) + yy * sin(theta);
                float yp = yy * cos(theta) - xx * sin(theta);
                *output_ptr = exp((float)(-CV_PI) * (xp * xp / para.sigmaX / para.sigmaX + yp * yp / para.sigmaY / para.sigmaY)) * cos((float)CV_2PI * para.beta / para.hairWidth * xp + (float)CV_PI);
            }
        }
    }
    return output;
}


void CubeReduction(
    float* d_Src,
    uchar* d_Dst,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int depth
) 
{
    dim3 block(BLOCKDIM, 8);
    dim3 grid(iDivUp(dataW, block.x), iDivUp(dataH, block.y));
    cubeReductionKernel << <grid, block >> > (d_Src, d_Dst, fftH, fftW, dataH, dataW, depth);
    getLastCudaError("CubeReductionKernel<<<>>> execution failed\n");
}

void modulateAndNormalize(
    fComplex * d_Dst,
    fComplex * d_DataSrc,
    fComplex * d_KernelSrc,
    int fftH,
    int fftW,
    int padding
)
{
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);

    modulateAndNormalizeKernel << <iDivUp(dataSize, 256), 256 >> > (
        d_Dst,
        d_DataSrc,
        d_KernelSrc,
        dataSize,
        1.0f / (float)(fftW * fftH)
        );
    getLastCudaError("modulateAndNormalize() execution failed\n");
}