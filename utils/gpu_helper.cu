#include <iostream>
using namespace std;
#include <stdio.h>
#include <fstream>
#include <time.h>

/*
This file define helper functions for CUDA programming, such as: error checking, print device info, save data to file, etc.
*/

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer {
    cudaEvent_t start, stop;
    cudaStream_t stream;  // stream to time (default 0)

    GpuTimer(cudaStream_t s = 0) : stream(s) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() {
        cudaEventRecord(start, stream);   // no sync here
    }
    void Stop() {
        cudaEventRecord(stop, stream);
    }
    float Elapsed() {
        cudaEventSynchronize(stop);       // wait for work between start/stop
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};


void printDeviceInfo() 
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Max threads per block: %d\n", devProv.maxThreadsPerBlock);
    printf("Max threads dimensions (x,y,z): (%d, %d, %d)\n",
           devProv.maxThreadsDim[0], devProv.maxThreadsDim[1], devProv.maxThreadsDim[2]);
    printf("Max grid size (x,y,z): (%d, %d, %d)\n",
           devProv.maxGridSize[0], devProv.maxGridSize[1], devProv.maxGridSize[2]);

    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("Max block per SM: %d\n", devProv.maxBlocksPerMultiProcessor);
    
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}


__device__ float relu(float x) 
{
    return x > 0.0f ? x : 0.0f;
}

__global__ void applyReLU_vector(float* data, int size) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = relu(data[tid]);
    }
}