#include <cuda_runtime.h>
#include <iostream>

__global__ void shift_array(int* d_a, int N) {
    // register mem
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int temp;

    if (i < N - 1) {
        temp = d_a[i + 1];
    }

    __syncthreads();

    if (i < N - 1) {
        d_a[i] = temp;
    }
}

// but __syncthreads() only syncs within a block, actually just better to copy to an out array

__global__ void shift_array_better(int* d_a, int N, int* d_out){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N - 1) {
        d_out[i] = d_a[i + 1];
    }
}