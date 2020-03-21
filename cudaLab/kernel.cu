
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>
//#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#pragma comment (lib, "curand.lib")

#define MY_MAX_INT 9
#define MY_MIN_INT 1

void Print(unsigned int* a, const size_t nrow, const size_t ncol)
{
    for (size_t i = 0; i < nrow; i++)
    {
        for (size_t j = 0; j < ncol; j++)
        {
            std::cout << a[i * ncol + j] << " ";
        }
        std::cout << std::endl;
    }
}




#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void scaleToMinMax(unsigned int* a, const size_t size, const int min, const int max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        *(a+i)%= (max - min) + min+1;
    } 
}

unsigned int main()
{
    unsigned int* dev_a = 0;
    unsigned int* dev_b = 0;
    unsigned int* host_a;
    unsigned int* host_b;

    constexpr size_t NRowBlock = 2;
    constexpr size_t NElementsInBlock = 4;
    constexpr size_t NRowElements = NRowBlock*NElementsInBlock;
    constexpr size_t NCol = 2;
    constexpr size_t dataSize = NRowElements * NCol;

//    cudaError_t cudaStatus;

    curandGenerator_t gen;

    /* Allocate n floats on host */
    host_a = (unsigned int*)calloc(dataSize, sizeof(unsigned int));

    /* Allocate n floats on host */
    host_b = (unsigned int*)calloc(dataSize, sizeof(unsigned int));

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void**)&dev_a, dataSize * sizeof(unsigned int)));

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void**)&dev_b, dataSize * sizeof(unsigned int)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
        1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerate(gen, dev_a, dataSize));

    ////////////////////////////////////////////////////////////////

    scaleToMinMax << <dataSize,1 >> > (dev_a, dataSize, MY_MIN_INT, MY_MAX_INT);


    ////////////////////////////////////////////////////////////////

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(host_a, dev_a, dataSize* sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(host_b, dev_b, dataSize * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    // cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
    CUDA_CALL(cudaDeviceReset());

    /* Show result */
    Print(host_a, NRowElements, NCol);
    printf("\n");

    Print(host_b, NCol, NRowElements);
    printf("\n");

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(dev_a));
    CUDA_CALL(cudaFree(dev_b));
    free(host_a);
    free(host_b);

    return 0;
}

