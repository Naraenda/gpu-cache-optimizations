#include <stdlib.h>
#include <chrono>
#include <iostream>

template <typename T>
struct Matrix {
    int width;
    int height;
    T *array;

public:
    Matrix<T>(int width, int height) {
        this->width  = width;
        this->height = height;
        this->array  = (T*) malloc(sizeof(T) * width * height);
    }

    inline T Get(int x, int y) {
        return this->array[y * this->width + x];
    }

    inline void Set(int x, int y, T value) {
        this->array[y * this->width + x] = value;
    }

    void Free() {
        this->width = -1;
        this->height = -1;
        free(array);
    }
};

template <typename T>
void multiply(Matrix<T> a, Matrix<T> b, Matrix<T> out) {
    int depth = a.height;
    for (int y = 0; y < b.height; y++) {
        for (int x = 0; x < a.width; x++) {
            T result = 0;
            for (int d = 0; d < depth; d++) {
                result += a.Get(x, d) * b.Get(d, y);
            }
            out.Set(x, y, result);
        }
    }
}

template <typename T>
void multiply_col(Matrix<T> a, Matrix<T> b, Matrix<T> out) {
    int csize = 512;
    int depth = a.height;
    int cols = (a.width + csize - 1) / csize;

    for (int c = 0; c < cols; c++) {
        for (int y = 0; y < b.height; y++) {
            for (int x_ = 0; x_ < csize; x_++) {
                int x = c * csize + x_;
                T result = 0;
                for (int d = 0; d < depth; d++) {
                    result += a.Get(x, d) * b.Get(d, y);
                }
                out.Set(x, y, result);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int test_count = 100;
    std::chrono::steady_clock::time_point start, stop;
    std::chrono::microseconds duration;

    Matrix<float> a = Matrix<float>(512, 512);
    Matrix<float> b = Matrix<float>(512, 512);
    Matrix<float> c = Matrix<float>(512, 512);
    
    std::cout << "# Naive Multiply" << std::endl;
    multiply<float>(a, b, c);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        multiply<float>(a, b, c);
    }
    stop  = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() / test_count << " us" << std::endl;


    std::cout << "sleeping..." << std::endl;
    _sleep(500);

    std::cout << "# Column Multiply" << std::endl;
    multiply_col<float>(a, b, c);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        multiply_col<float>(a, b, c);
    }
    stop  = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() / test_count << " us" << std::endl;
}