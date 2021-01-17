#pragma once


struct EntropyThresholdDeviceInfo {
    int fullWidth;
    int targetWidth;
    int multipleWidth;
    int startThreshold;
    int sumMatrixSize;
    dim3* preSumBlock;
    dim3* preSumGrid;
    int preSumSmemSize;
    dim3* sumBlock;
    dim3* sumGrid;
    int sumSmemSize;
    dim3* sumSumBlock;
    dim3* sumSumGrid;
    int sumSumSmemSize;

    EntropyThresholdDeviceInfo(int full_width) :
        targetWidth(0),
        multipleWidth(0),
        fullWidth(full_width),
        startThreshold(TILE_DIM - 1),
        preSumBlock(new dim3(TILE_DIM, TILE_DIM)),
        preSumGrid(nullptr),
        sumBlock(new dim3(TILE_DIM, TILE_DIM)),
        sumGrid(nullptr),
        sumSumBlock(new dim3(iDivUp(full_width, TILE_DIM)* iDivUp(full_width, TILE_DIM))),
        sumSumGrid(new dim3(1)),
        preSumSmemSize(0),
        sumSmemSize(TILE_DIM* TILE_DIM * sizeof(float)),
        sumSumSmemSize(iDivUp(full_width, TILE_DIM)* iDivUp(full_width, TILE_DIM) * sizeof(float)),
        sumMatrixSize(iDivUp(full_width, TILE_DIM)* iDivUp(full_width, TILE_DIM)) { };
};