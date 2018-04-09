#include <iostream>
#include <math.h>
#include <stdio.h>

__global__ void add(int n, float *x, float *y, float *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  c[idx] = x[idx] + y[idx];
}

void FillWithData(int n, float* x, float* y) {
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
} 

void FillWith(int n, float value, float* x) {
  for (int i = 0; i < n; i++) {
    x[i] = value;
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
  int N = 1<<20;
//  int N = 200000;
  //int N = 1024;
//  int N = 1 << 20;
  float *x, *y, *c;
  float *d_x, *d_y, *d_c;
  int size = N * sizeof(float);

  x = (float*) malloc(size);
  y = (float*) malloc(size);
  c = (float*) malloc(size);

  FillWithData(N, x, y);
  FillWith(N, 0.0f, c);

  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);
  cudaMalloc(&d_c, size);
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice); 

  add<<<N/(1024-1), 1024>>>(N, d_x, d_y, d_c);

  CheckCudaError();

  cudaDeviceSynchronize();
  //  cudaDeviceSync();
  //  cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); 

  int i = 0;
  int sample_rate = N / 100;
  for (i = 0; i < N; i=i+sample_rate) {
    printf("Value %d - %f + %f = %f\n" , i, x[i], y[i], c[i]);
  } 

  // Free memory
  free(x); free(y);
  cudaFree(d_x); cudaFree(d_y);
  
  return 0;
}
