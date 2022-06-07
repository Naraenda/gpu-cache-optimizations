#pragma diag_suppress 177

#include <iostream>
#include <cuda_fp16.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define SIZE  1024

template <typename T>
__global__ void matrix_naive(T *o, T *a, T *b, int depth, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = tid % width;
    int y = tid / width;
    
    if (tid >= width * height) {
        return;
    }

    float result = 0;
    for (int i = 0; i < depth; i++) {
        result += a[y * width + i] * b[x + i * width];
    }
    o[tid] = result;
    
}
template <typename T>
__global__ void matrix_column(T *o, T *a, T *b, int depth, int width, int height, int col_w) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x;
    int y;
    
    if (tid >= width * height) {
        return;
    }

    int col_size = height * col_w;

    // Column identifier
    int col_id  = tid / col_size;

    // Thread id within column
    int col_tid = tid % col_size;

    // Column offset
    int col_off = col_id * col_w;

    // True width of column
    //
    // int col_tw  = col_w;
    // if (col_id == width  / col_w) {
    //     col_tw  = width - col_id * col_w;
    // }
    //
    // A shorter version of the above since col_tw is always bounded by col_w
    // In PTX it replaces a setp and selp with a single minimum instruction
    int col_tw = min(width - col_id * col_w, col_w);

    x = (col_tid % col_tw) + col_off;
    y =  col_tid / col_tw;
    tid = x + (y * width);

    float result = 0;
    for (int i = 0; i < depth; i++) {
        result += a[y * width + i] * b[x + i * width];
    }
    o[tid] = result;
    
}

void matrix_cublas(float *o, float *a, float *b, int size) {
    cublasHandle_t handle;
    
    cublasStatus_t stat;
    float alpha = 1.0f;
    float beta  = 0.0f;
    cublasCreate(&handle);
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, a, size, b, size, &beta, o, size);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed");
    }
    cublasDestroy(handle);
}

template <typename T>
void benchmark(int bench_count, int block_size, int column_width, int size) {
    int N = size * size;

    // Host Memory
    T *h_a, *h_b, *h_o;
    h_a = (T*)malloc(sizeof(T) * N);
    h_b = (T*)malloc(sizeof(T) * N);
    h_o = (T*)malloc(sizeof(T) * N);

    std::srand(0);
    bool *check = (bool*)malloc(sizeof(bool) * N);
    for (size_t i = 0; i < N; i++) {
        h_a[i] = std::rand();
        h_b[i] = std::rand();
    }

    // Device Memory
    T *d_a, *d_b, *d_o;
    cudaMalloc(&d_a, sizeof(T) * N);
    cudaMalloc(&d_b, sizeof(T) * N);
    cudaMalloc(&d_o, sizeof(T) * N);

    // Host to Device Copy
    cudaMemcpy(d_a, h_a, sizeof(T) * N, cudaMemcpyHostToDevice);

    // Kernel Execution
    int block_count = N / block_size;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaMemcpy(d_a, h_a, sizeof(T) * N, cudaMemcpyHostToDevice);

    // run benchmark
    cudaEvent_t gpu_start, gpu_stop;
    
    std::cout << "Running benchmark." << std::endl;
    float naive_time = 0;
    for (size_t i = 0; i < bench_count; i++) {
        float execution_time;
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_stop);

        //cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
        
        cudaEventRecord(gpu_start);
        if (column_width == -1) {
            matrix_cublas(d_o, d_a, d_b, size);
        } else if (column_width == 0) {
            matrix_naive<T><<<block_count, block_size>>>(d_o, d_a, d_b, size, size, size);
        } else {
            matrix_column<T><<<block_count, block_size>>>(d_o, d_a, d_b, size, size, size, column_width);
        }
        
        // CUBLAS:

        cudaEventRecord(gpu_stop);
        
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&execution_time, gpu_start, gpu_stop);

        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);

        naive_time += execution_time / bench_count;
    }

    // Device to Host Copy
    cudaMemcpy(h_o, d_o, sizeof(T) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_o);

    free(h_a);
    free(h_o);
}

int main(int argc, char *argv[]) {

    int l1_size = 64 * 1024;
    int l2_size = 0;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);    
    int bench_count = 1;

    bool run_naive = false;

    int column_width = 32;
    if (argc >= 2) {
        char* strp;
        column_width = std::strtol(argv[1], &strp, 10);
    }

    int block_size = 128;
    if (argc >= 3) {
        char* strp;
        block_size = std::strtol(argv[2], &strp, 10);
    }

    int size = SIZE;
    if (argc >= 4) {
        char* strp;
        size = std::strtol(argv[3], &strp, 10);
    }

    std::cout << "column width: " << column_width << std::endl;
    std::cout << "block size  : " << block_size << std::endl;
    std::cout << "input size  : " << size << std::endl;

    //benchmark<half>(N, bench_count, block_size, column_width);
    benchmark<float>(bench_count, block_size, column_width, size);
    //benchmark<double>(N, bench_count, block_size, column_width);
}