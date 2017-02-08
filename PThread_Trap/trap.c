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
#define NUM_THREADS 4

double compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);
void pthread_compute(void* rank);

double area = 0.0;
pthread_mutex_t mutex;
int local_n;


int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);

	double reference = compute_gold(a, b, n, h);
   printf("Reference solution computed on the CPU = %f \n", reference);

	/* Write this function to complete the trapezoidal on the GPU. */
	double pthread_result = compute_using_pthreads(a, b, n, h);
	printf("Solution computed using pthreads = %f \n", pthread_result);
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
	pthread_t* thread_handles;
	
	local_n = n / NUM_THREADS;
	thread_handles = malloc (NUM_THREADS*sizeof(pthread_t));
	
	pthread_mutex_init(&mutex, NULL);
	
	for(i = 0; i < NUM_THREADS; i++)
	{
		pthread_create(&thread_handles[i], NULL, pthread_compute, (void*) i);
	}
	
	for(i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(thread_handles[i], NULL);
	}

	pthread_mutex_destroy(&mutex);
	free(thread_handles);

	return area;
}

void pthread_compute(void* rank)
{
	double local_a;
	double local_b;
	double my_int;
	long my_rank = (long) rank;
	double h;
	
	h = (RIGHT_ENDPOINT - LEFT_ENDPOINT)/(float) NUM_TRAPEZOIDS;
	local_a = LEFT_ENDPOINT + my_rank*local_n*h;
	local_b = local_a + local_n*h;

	my_int = compute_gold(local_a, local_b, local_n, h);
	pthread_mutex_lock(&mutex);
	area += my_int;
	pthread_mutex_unlock(&mutex);

}

