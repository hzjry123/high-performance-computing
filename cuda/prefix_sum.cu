/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * This is a skeleton for implementing prefix sum using GPU.
 */

#include <stdio.h>
#include "timer.h"
#include <math.h>
#include <string.h>

#define THREADS_PER_BLOCK 1024

/*
 * You should implement the Hillis/Steele inclusive scan here!
 */
__global__ void par_scan(const int n,
			 const float *g_idata,
			 float *g_odata) {
  // Use a flat array as in/out arrays.
  __shared__ float temp[THREADS_PER_BLOCK * 2];
  int thid = threadIdx.x;
  int pout = 0, pin = 1;
  // Load input into shared memory.
  temp[pout*n + thid] = g_idata[thid];
  temp[pin*n + thid] = g_idata[thid];
  __syncthreads();

//  // STUDENT: YOUR CODE GOES HERE.
  g_odata[thid] = temp[pout*n + thid];

}

/*
 * Fills an array a with n random floats.
 */
void random_floats(float* a, int n) {
  float d;
  // Comment out this line if you want consistent "random".
  srand(time(NULL));
  for (int i = 0; i < n; ++i) {
    d = rand() % 8;
    a[i] = ((rand() % 2) / (d > 0 ? d : 1));
    // Uncomment if you want to be sequential (helpful for debugging)
    //    a[i] = i+1;
  }
}

/*
 * Simple Serial implementation of inclusive scan.
 */
void serial_scan(int n, float* in, float* out) {
  // This is an inclusive scan, seed out array with first element.
  float total_sum = in[0];
  out[0] = in[0];
  for (int i = 1; i < n; i++) {
    total_sum += in[i];
    out[i] = out[i-1] + in[i];
  }
  if (total_sum != out[n-1]) {
    printf("Warning: exceeding accuracy of float.\n");
  }
}

/*
 * This is a simple function that confirms that the output of the scan
 * function matches that of a golden image (array).
 */
bool printError(int N, float *gold_out, float *test_out, bool show_all) {
  bool firstFail = true;
  bool error = false;
  float epislon = 0.1;
  float diff = 0.0;
  for (int i = 0; i < N; ++i) {
    diff = abs(gold_out[i] - test_out[i]);
    if ((diff > epislon) && firstFail) {
      printf("ERROR: gold_out[%d] = %f != test_out[%d] = %f // diff = %f \n", i, gold_out[i], i, test_out[i], diff);
      firstFail = show_all;
      error = true;
    }
  }
  return error;
}

int main(void) {
  const int kMaxThreadsPerBlock = THREADS_PER_BLOCK;
  const int kSimpleCount = kMaxThreadsPerBlock;
  const int kSimpleSize = sizeof(float) * kMaxThreadsPerBlock;

  float *in, *out, *gold_out; // host
  float *d_in, *d_out; // device
  timerStart();
  cudaMalloc((void **)&d_in, kSimpleSize);
  cudaMalloc((void **)&d_out, kSimpleSize);
  
  in = (float *)malloc(kSimpleSize);
  random_floats(in, kSimpleCount);
  out = (float *)malloc(kSimpleSize);
  gold_out = (float *)malloc(kSimpleSize);
  printf("TIME: Init took %d ms\n",  timerStop());
  // ***********
  // RUN SERIAL - SIMPLE SIZE (1024)
  // ***********
  timerStart();
  serial_scan(kSimpleCount, in, gold_out);
  printf("TIME: Serial (1024) took %d ms\n",  timerStop());

  timerStart();
  cudaMemcpy(d_in, in, kSimpleSize, cudaMemcpyHostToDevice);
  printf("TIME: Copy took %d ms\n",  timerStop());
  // ***********
  // RUN PARALLEL SCAN - SIMPLE SIZE (1024)
  // ***********
  timerStart();
  par_scan<<<1, kMaxThreadsPerBlock>>>(kSimpleCount, d_in, d_out);
  cudaDeviceSynchronize();
  printf("TIME: Parellel (1024) kernel took %d ms\n",  timerStop());
  timerStart();
  cudaMemcpy(out, d_out, kSimpleSize, cudaMemcpyDeviceToHost);
  printf("TIME: Copy back %d ms\n",  timerStop());

  if (printError(kSimpleCount, gold_out, out, true)) {
    printf("ERROR: The parallel scan function failed to produce proper output.\n");
  } else {
    printf("CONGRATS: The simple scan function produced proper output.\n");
  }

  return 0;
}
