#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_addition(const float* d_a, const float* d_b, float* d_out, int N){
    int i = threadIdx.x;
    if (i < N) {
        d_out[i] = d_a[i] + d_b[i];
    }
}

int main() {
    int N = 4;
    float h_a[] = {1, 2, 3, 4};
    float h_b[] = {10, 20, 30, 40};
    float *h_out = (float*)malloc(N * sizeof(float));

    float *d_a, *d_b, *d_out;

    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_addition<<<1, N>>>(d_a, d_b, d_out, N);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_out)

    return 0
}