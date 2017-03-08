#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

#define BLOCK_SIZE 256
#define GRID_SIZE 128

__device__ void lock(int *mutex);
__device__ void unlock(int *mutex);

__global__ void vector_dot_product_kernel(int num_elements, float* a, float* b, float* result, int *mutex)
{
	__shared__ float sum[BLOCK_SIZE];
	
	int tx = threadId.x;
	int tid = blockDim.x * blockId.x + threadId.x;
	int stride = blockDim.x * gridDim.x;

	float local_sum = 0.0f;
	unsigned int i = tid;

	while(i < num_elements)
	{
		local_sum += a[i] * b[i];
		i += stride;
	}

	sum[threadId.x] = local_sum;
	__syncthreads();

	for(int stride = blockDim.x/2; stride > 0; stide /= 2)
	{
		if(tx < stride)
			sum[tx] += sum[tx + stride];
		__syncthreads();
	}

	if(threadId.x == 0)
	{
		lock(mutx);
		result[0] += sum[0];
		unlock(mutex);
	}

}


__device__ void lock(int *mutex)
{
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex)
{
	atomicExch(mutex, 0);
}

#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
