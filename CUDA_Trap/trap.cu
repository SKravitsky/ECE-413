#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// includes, kernels
#include "trap_kernel.cu"

#define BLOCK_DIM 128
#define LEFT_ENDPOINT 10
#define RIGHT_ENDPOINT 1005
#define NUM_TRAPEZOIDS 100000000

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);

int 
main(void) 
{
	struct timeval start, stop;

    int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);
	
	gettimeofday(&start, NULL);
	double reference = compute_gold(a, b, n, h);
	gettimeofday(&stop, NULL);
	printf("CPU Execution time = %fus. \n", (float)(stop.tv_usec - start.tv_usec + (stop.tv_usec - start.tv_usec)/(float)1000000));


	/* Write this function to complete the trapezoidal on the GPU. */
	double gpu_result = compute_on_device(a, b, n, h);
	
	printf("Reference solution computed on the CPU = %f \n", reference);
	printf("Solution computed on the GPU = %f \n", gpu_result);
} 

/* Complete this function to perform the trapezoidal rule on the GPU. */
double 
compute_on_device(float a, float b, int n, float h)
{
    struct timeval start, stop;

	int num_columns = 8192 / BLOCK_DIM;

	double result;

	float *results = (float *)malloc(num_columns * sizeof(float));
	float *R_dev;

    cudaMalloc((void**)&R_dev, num_columns * sizeof(float));

	dim3 dimBlock(BLOCK_DIM, 1, 1);
	dim3 dimGrid(num_columns, 1);

	gettimeofday(&start, NULL);
	trap_kernel <<< dimGrid, dimBlock >>> (a, b, n, h, R_dev);
	cudaThreadSynchronize();
	gettimeofday(&stop, NULL);

	printf("GPU Execution time = %fus. \n", (float)(stop.tv_usec - start.tv_usec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	cudaMemcpy(results, R_dev, num_columns * sizeof(float), cudaMemcpyDeviceToHost);

	result = ((F(b)) + (F(a))) / 2.0;
	for(int i = 0; i < num_columns; i++)
	{
		result += results[i];
	}
	result *= h;

	free(results);
	cudaFree(R_dev);


    return result;
}



