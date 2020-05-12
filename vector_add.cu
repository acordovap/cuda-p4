#include <stdio.h>
#include <math.h>

#define N 512
#define MAX_ERR 1

__global__ void vector_add(float *out, float *a, float *b, int n){
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  //printf("index: %d\n", index);
  for(int i = index; i < n; i+=blockDim.x){
    printf("i=%d \n", i);
    out[i] = a[i] + b[i];
  }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
      a[i] = 1.0f; b[i] = 2.0f;
    }

    // Allocate device memory for a
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host t  o device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function

    vector_add<<</*1*/(N+256)/256,256>>>(d_out, d_a, d_b, N);

    // Transfer data from device to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Print results
    for(int i=0; i < N; i++){
      //if(fabs(out[i] - a[i] - b[i]) < MAX_ERR )
      //printf("failed");
      printf("%i.- %f = %f + %f \n", i, out[i], a[i], b[i]);
    }

    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);

}
