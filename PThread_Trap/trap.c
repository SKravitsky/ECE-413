/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -o trap trap.c -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
#define NUM_TRAPEZOIDS 100000000
#define NUM_THREADS 16

double compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);
void pthread_compute(void* rank);

int thread_n;
double area = 0.0;
pthread_mutex_t mutex;



int main(void) 
{
	struct timeval start1,stop1,start2,stop2;

	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);
    
	gettimeofday(&start1,NULL);
		double reference = compute_gold(a, b, n, h);
    gettimeofday(&stop1,NULL);
    
	float time1 = (float) (stop1.tv_sec - start1.tv_sec +(stop1.tv_usec - start1.tv_usec) / (float) 1000000);
  	printf("Reference solution computed on the CPU = %f \nRun time: %f\n", reference,time1);

	/* Write this function to complete the trapezoidal on the GPU. */
    gettimeofday(&start2,NULL);
		double pthread_result = compute_using_pthreads(a, b, n, h);
    gettimeofday(&stop2,NULL);
    float time2 = (float) (stop2.tv_sec - start2.tv_sec +(stop2.tv_usec - start2.tv_usec) / (float) 1000000);
	
	printf("Solution computed using pthreads = %f \nRun time: %f\n", pthread_result,time2);
    printf("Speedup: %f times\n",time1/time2);

//	double reference = compute_gold(a, b, n, h);
  // printf("Reference solution computed on the CPU = %f \n", reference);

	/* Write this function to complete the trapezoidal on the GPU. */
//	double pthread_result = compute_using_pthreads(a, b, n, h);
//	printf("Solution computed using pthreads = %f \n", pthread_result);
} 


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 * Output: (x+1)/sqrt(x*x + x + 1)

 */
float f(float x) {
		  return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double compute_gold(float a, float b, int n, float h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_using_pthreads(float a, float b, int n, float h)
{
	int i;
	int rv;
	int thread_count = NUM_THREADS;
	pthread_t* pthreads;
	
	thread_n = n / thread_count;
	pthreads = malloc (thread_count*sizeof(pthread_t));
	
	pthread_mutex_init(&mutex, NULL);
	
	for(i = 0; i < NUM_THREADS; i++)
	{
		rv = pthread_create(&pthreads[i], NULL, pthread_compute, (void*) i);
		if(rv)
		{
			printf("Error creating Threads \n");
		}
		
	}
	
	for(i = 0; i < NUM_THREADS; i++)
	{
		rv = pthread_join(pthreads[i], NULL);
		if(rv)
		{
			printf("Error joining Threads \n");
		}
	}

	pthread_mutex_destroy(&mutex);
	free(pthreads);

	return area;
}

void pthread_compute(void* rank)
{
	double thread_a;
	double thread_b;
	double thread_area;
	double h;
	long thread_num = (long) rank;
	
	h = (RIGHT_ENDPOINT - LEFT_ENDPOINT) / (float) NUM_TRAPEZOIDS;
	
	thread_a = LEFT_ENDPOINT + thread_num * thread_n * h;
	thread_b = thread_a + thread_n * h;
	thread_area = compute_gold(thread_a, thread_b, thread_n, h);
	
	pthread_mutex_lock(&mutex);
	area += thread_area;
	pthread_mutex_unlock(&mutex);

}

