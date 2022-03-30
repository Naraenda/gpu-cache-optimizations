#define WIDTH  1024
#define HEIGHT 1024
#define STENCIL_SIZE 5

__global__ void stencil_naive(float *output, float *input, int width, int height, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = tid % width;
    int y = tid / width;
    
    if (tid >= N) {
        return;
    }

    float result = 0;
    for (int d_i = -STENCIL_SIZE/2; d_i <= STENCIL_SIZE/2; d_i++) {
        int i = min(height-1, max(0, y + d_i));
        for (int d_j = -STENCIL_SIZE/2; d_j <= STENCIL_SIZE/2; d_j++) {
            int j = min(width-1, max(0, x + d_j));
            result += input[i * width + j];
        }
    }
    output[tid] = result / (STENCIL_SIZE * STENCIL_SIZE);
}

int main() {
    float *a, *out;
    a   = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
    out = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);

    stencil_naive<<<2048, 512>>>(out, a, WIDTH, HEIGHT, WIDTH * HEIGHT);
}