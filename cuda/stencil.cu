#pragma diag_suppress 177

#include <iostream>
#include <cuda_fp16.h>

#define SIZE  4096
#define STENCIL_SIZE 9


struct remapped_index {
    int x, y, idx;
    int debug;
};


template <typename T>
__global__ void stencil_naive(T *output, T *input, int width, int height, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = tid % width;
    int y = tid / width;
    
    if (tid >= N) {
        return;
    }

    T result = 0;
    for (int d_i = -STENCIL_SIZE/2; d_i <= STENCIL_SIZE/2; d_i++) {
        int i = min(height-1, max(0, y + d_i));
        for (int d_j = -STENCIL_SIZE/2; d_j <= STENCIL_SIZE/2; d_j++) {
            int j = min(width-1, max(0, x + d_j));
            result += input[i * width + j];
        }
    }
    output[tid] = result / static_cast<T>(STENCIL_SIZE * STENCIL_SIZE);
}

template <typename T>
__global__ void stencil_column(T *output, T *input, int width, int height, int N, int col_w) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x;
    int y;
    
    if (tid >= N) {
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

    T result = 0;
    for (int d_i = -STENCIL_SIZE/2; d_i <= STENCIL_SIZE/2; d_i++) {
        int i = min(height-1, max(0, y + d_i));
        for (int d_j = -STENCIL_SIZE/2; d_j <= STENCIL_SIZE/2; d_j++) {
            int j = min(width-1, max(0, x + d_j));
            result += input[i * width + j];
        }
    }
    output[tid] = result / static_cast<T>(STENCIL_SIZE * STENCIL_SIZE);
}

template <typename T>
__global__ void stencil_tile(T *output, T *input, int width, int height, int N {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }

    int tid = y * width + x;

    T result = 0;
    for (int d_i = -STENCIL_SIZE/2; d_i <= STENCIL_SIZE/2; d_i++) {
        int i = min(height-1, max(0, y + d_i));
        for (int d_j = -STENCIL_SIZE/2; d_j <= STENCIL_SIZE/2; d_j++) {
            int j = min(width-1, max(0, x + d_j));
            result += input[i * width + j];
        }
    }
    output[tid] = result / static_cast<T>(STENCIL_SIZE * STENCIL_SIZE);

}

template <typename T>
void benchmark(int N, int bench_count, int block_size, int column_width, int size) {

    // Host Memory
    T *h_a, *h_o;
    h_a = (T*)malloc(sizeof(T) * N);
    h_o = (T*)malloc(sizeof(T) * N);

    std::srand(0);
    bool *check = (bool*)malloc(sizeof(bool) * N);
    for (size_t i = 0; i < N; i++) {
        h_a[i] = std::rand();
    }

    // Device Memory
    T *d_a, *d_o;
    cudaMalloc(&d_a, sizeof(T) * N);
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
        if (column_width > 0) {
            stencil_column<T><<<block_count, block_size>>>(d_o, d_a, size, size, size * size, column_width);
        } else {
            stencil_naive<T><<<block_count, block_size>>>(d_o, d_a, size, size, size * size);
        }
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
    //int column_width = (l1_size - sizeof(float) * (STENCIL_SIZE * STENCIL_SIZE)) / (sizeof(float) * STENCIL_SIZE);
    //column_width = (column_width / 8) * 8;
    //column_width = 32;
    int bench_count = 100;

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
    benchmark<float>(size * size, bench_count, block_size, column_width, size);
    //benchmark<double>(N, bench_count, block_size, column_width);
}

/*
    measuring with cuda events does not show different results
    change datatypes
    run benchmark to produce graphs 
    inspect ptx code for weird optimizations
    for the paper don't change the theorhetical analysis
    solve asymptote to compare with 32 fixed value
*/