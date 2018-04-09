#include <iostream>
#include <math.h>
#include <stdio.h>
#include "timer.h"

#define BLOCK_SIZE 1024
#define N 256*256*16

__global__ void reduce_interleaved(int *g_idata, int *g_odata) {
  __shared__ int sdata[BLOCK_SIZE*2]; 
  // each thread loads 1 element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  //  printf("Thread %d data %d\n", tid, g_idata[i]);
  __syncthreads(); 
  // do reduction in shared mem 
  for(unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  } 

  // write result for this block to global mem 
  if (tid == 0) {
    // printf("Found for block %d total %d\n", blockIdx.x, sdata[0]);
    g_odata[blockIdx.x] = sdata[0];
  }
} 


__global__ void reduce_lessbranch(int *g_idata, int *g_odata) {
  __shared__ int sdata[BLOCK_SIZE*2]; 
  // each thread loads 1 element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  //  printf("Thread %d data %d\n", tid, g_idata[i]);
  __syncthreads(); 

  for(unsigned int s=1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  } 

  // write result for this block to global mem 
  if (tid == 0) {
    // printf("Found for block %d total %d\n", blockIdx.x, sdata[0]);
    g_odata[blockIdx.x] = sdata[0];
  }
} 

void FillWithData(int n, int* x) {
  for (int i = 0; i < n; i++) {
    x[i] = 1;
  }
} 

void CheckCudaError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

int main(void) {
  //  int N = 1<<20;
  //  int N = 1024;
  int *x, *final;
  int *d_x, *d_out, *d_final;
  int size = N * sizeof(int);
  int out_size = N/BLOCK_SIZE * sizeof(int);

  x = (int*) malloc(size);
  //  out = (int*) malloc(out_size);
  final = (int*) malloc(sizeof(int));
  FillWithData(N, x);

  printf("Copying data to gpu.\n");

  printf("BLOCK_SIZE %d, N %d, OUT %d\n", BLOCK_SIZE, N, (N/BLOCK_SIZE));

  cudaMalloc(&d_x, size);
  cudaMalloc(&d_out, out_size);
  cudaMalloc(&d_final, sizeof(int));

  timerStart();

  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

  printf("TIME: Init took %d ms\n",  timerStop());

  timerStart();
  reduce_lessbranch<<<(N/BLOCK_SIZE), BLOCK_SIZE>>>(d_x, d_out);
  CheckCudaError();

  printf("TIME: First kernel took %d ms\n",  timerStop());

  timerStart();
  reduce_lessbranch<<<1, N/BLOCK_SIZE>>>(d_out, d_final);
  CheckCudaError();

  printf("TIME: Second kernel took %d ms\n",  timerStop());

  cudaMemcpy(final, d_final, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Final answer %d\n", final[0]);

  // Free memory
  free(x);
  cudaFree(d_x); cudaFree(d_out);
  
  return 0;
}
