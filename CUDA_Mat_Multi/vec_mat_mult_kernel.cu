/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Creating the threadID counter and a tem variable for the sum
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float final = 0;

	//This is basically the samme as the compute_gold function but with the threadIDs as an iterator
	for(unsigned int i = 0; i < MATRIX_SIZE; i++)
	{
		float a = Ad[tid * MATRIX_SIZE + i];
		float b = Xd[i];
		final += a * b;
	}

	Yd[tid] = final;

}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	//Creating the shared variables
	__shared__ float a[16][16];
	__shared__ float b[16];

	//Separating the threadId variables so they can be used by the shared variables
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*blockDim.y;

	int final = 0;

	for(unsigned int i = 0; i < MATRIX_SIZE; i += 16)
	{
		//Splitting the work over the threads
		a[ty][tx] = Ad[row * MATRIX_SIZE + i + tx];
		b[tx] = Xd[tx + i];

		//Making sure all the threads in the block have finished their computation before putting it into the final array
		__syncthreads();
		if(threadIdx.x==0)
		{
			for(unsigned int j =0; j < blockDim.x; j++)
			{
				final += a[tx][j] * b[j];
			}
		}
		__syncthreads();
	}
	
	if(threadIdx.x == 0)
	{
		Yd[row] = final;
	}
	
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
