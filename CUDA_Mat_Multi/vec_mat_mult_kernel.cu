/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
