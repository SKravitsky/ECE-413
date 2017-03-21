/* Write GPU kernels to compete the functionality of estimating the integral via the trapezoidal rule. */ 

#define F(n) ((n) + 1)/sqrt((n) * (n) + (n) + 1)

__global__ void trap_kernel(float a, float b, int n, float h, float *Result_Vector)
{
	__shared__ float p[128];

	unsigned int column = blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int stride = blockDim.x * gridDim.x;


	double temp;
	double sum = 0;

	for(unsigned int i = column; i < n; i += stride)
	{
		temp = a + (i * h);
		sum += F(temp);
	}

	p[threadIdx.x] = sum;

	for(unsigned int j = 1; (j << 1) <= blockDim.x; j <<= 1)
	{
        __syncthreads();
        if (threadIdx.x + j < blockDim.x)
        {
            p[threadIdx.x] += p[threadIdx.x + j];
        }
    }

    if (threadIdx.x == 0)
    {
        Result_Vector[blockIdx.x] = p[0];
    }

}