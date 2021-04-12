#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Timer {

    float time;
    const uint64_t gpu;
    cudaEvent_t t0, t1;

public:

    Timer(uint64_t gpu = 0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
    }

    ~Timer() {
        cudaSetDevice(gpu);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    void start() {
        cudaSetDevice(gpu);
        cudaEventRecord(t0, 0);
    }

    void stop(std::string label) {
        cudaSetDevice(gpu);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&time, t0, t1);
        std::cout << "TIMING: " << time << " ms (" << label << ")" << std::endl;
    }
};