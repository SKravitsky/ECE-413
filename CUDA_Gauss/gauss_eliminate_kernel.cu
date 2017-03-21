 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_reduce_kernel(float *U, int i, int num_col)
{
	float F = U[i * num_col + i];

	unsigned int column = ((blockDim.x * threadIdx.y) + threadIdx.x) + ((blockDim.x * blockDim.y) * blockIdx.y) + i + 1;
	unsigned int stride = gridDim.y * blockDim.y * blockDim.x; 	

	for( unsigned int j = column; j < num_col; j += stride)
	{
		U[i * num_col + j] /= F;
	}
}

__global__ void gauss_eliminate_kernel(float *U, int i, int num_col)
{
	__shared__ float p[16][16]

	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y + i + 1;
	unsigned int r_stride = gridDim.y * blockDim.y;

	unsigned int column = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int c_stride = gridDim.x * blockDim.x;

	U[i * num_col + i] = 1;

	float temp;

	for(unsigned int j = row; j < num_col; j += r_stride)
	{
		temp = U[j * num_col + i];
		__syncthreads();

		for(unsigned int k = column; k < num_col; k += c_stride)
		{
			p[threadIdx.x][threadIdx.y] = U[i * num_col + k];
			U[j * num_col + k] -= __fmul_rn(temp, p[threadIdx.x][threadIdx.y]);
		}
	}
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
