#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
#define getLastCudaError(msg) {__getLastCudaError (msg, __FILE__, __LINE__);}

template< typename T >
inline void gpuAssert(T code, const char* file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}

inline void __getLastCudaError(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Overload error code
////////////////////////////////////////////////////////////////////////////////


// basic cuda errors
static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

#ifdef _CUFFT_H_
// cuFFT API errors
static const char* _cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    }

    return "<unknown>";
}
#endif