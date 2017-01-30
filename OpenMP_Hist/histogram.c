/* Lab 2: Histrogram generation 
 * Author: Naga Kandasamy
 * Date: 01/25/2017
 *
 * compile as follows: gcc -o histogram histogram.c -std=c99 -fopenmp -lm
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <omp.h>

void run_test(int);
void compute_gold(int *, int *, int, int);
void compute_using_openmp(int *, int *, int, int);
void check_histogram(int *, int, int);

#define HISTOGRAM_SIZE 500
#define NUM_THREADS 8

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: histogram <num elements> \n");
		exit(0);	
	}
	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Generate the histogram in both single-threaded and multi-threaded fashion and compare results for correctness
////////////////////////////////////////////////////////////////////////////////
void run_test(int num_elements) 
{
	double diff;
	int i; 
	int *reference_histogram = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); // Space to store histogram generated by the CPU
	int *histogram_using_openmp = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); // Space to store histogram generated by the GPU

	// Allocate memory for the input data
	int size = sizeof(int) * num_elements;
	int *input_data = (int *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be integer values between 0 and (HISTOGRAM_SIZE - 1)
	for(i = 0; i < num_elements; i++)
		input_data[i] = floorf((HISTOGRAM_SIZE - 1) * (rand()/(float)RAND_MAX));

	printf("Creating the reference histogram. \n"); 
	// Compute the reference solution on the CPU
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	compute_gold(input_data, reference_histogram, num_elements, HISTOGRAM_SIZE);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	// check_histogram(reference_histogram, num_elements, HISTOGRAM_SIZE);
	
	// Compute the histogram using openmp. The result histogram should be stored on the histogram_using_openmp array
	printf("\n");
	printf("Creating histogram using OpenMP. \n");
	compute_using_openmp(input_data, histogram_using_openmp, num_elements, HISTOGRAM_SIZE);
	// check_histogram(histogram_using_openmp, num_elements, HISTOGRAM_SIZE);

	// Compute the differences between the reference and pthread results
	diff = 0.0;
   for(i = 0; i < HISTOGRAM_SIZE; i++)
		diff = diff + abs(reference_histogram[i] - histogram_using_openmp[i]);

	printf("Difference between the reference and OpenMP results: %f. \n", diff);
   
	// cleanup memory
	free(input_data);
	free(reference_histogram);
	free(histogram_using_openmp);
}

/* This function computes the reference solution. */
void compute_gold(int *input_data, int *histogram, int num_elements, int histogram_size)
{
  int i;
  
  // Initialize histogram
  for(i = 0; i < histogram_size; i++) 
			 histogram[i] = 0; 

  // Bin the elements in the input stream
  for(i = 0; i < num_elements; i++)
			 histogram[input_data[i]]++;
}


// Write the function to compute the histogram using openmp
void compute_using_openmp(int *input_data, int *histogram, int num_elements, int histogram_size)
{
	int i;
	// Initialize histogram
  for(i = 0; i < histogram_size; i++) 
			 histogram[i] = 0; 
}

void check_histogram(int *histogram, int num_elements, int histogram_size)
{
	int sum = 0;
	for(int i = 0; i < histogram_size; i++)
		sum += histogram[i];

	printf("Number of histogram entries = %d. \n", sum);
	if(sum == num_elements)
		printf("Histogram generated successfully. \n");
	else
		printf("Error generating histogram. \n");
}


