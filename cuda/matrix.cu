#define SIZE  1024

__global__ void matrix_naive(float *o, float *a, float *b, int depth, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = tid % width;
    int y = tid / width;
    
    if (tid >= width * height) {
        return;
    }

    float result = 0;
    for (int i = 0; i < depth; i++)
    {
        result += a[y * width + i] * b[x + i * width];
    }
    o[tid] = result;
    
}

int main() {
    float *a, *b, *out;
    a   = (float*)malloc(sizeof(float) * SIZE * SIZE);
    b   = (float*)malloc(sizeof(float) * SIZE * SIZE);
    out = (float*)malloc(sizeof(float) * SIZE * SIZE);

    matrix_naive<<<2048, 512>>>(out, a, b, SIZE, SIZE, SIZE);
}