/*
 *  Ejercicio 5 Práctica 4: CUDA
 *  Mariana Hernández
 *  Alan Córdova
 */

#include <stdio.h>

#define STRIDE       32
#define OFFSET        0
#define GROUP_SIZE  512
#define SHARED_SIZE 256

// tamanio
#define n 8
//#define m 8

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N, int sec_size) {

    __shared__ float AS[SHARED_SIZE];
    __shared__ float BS[SHARED_SIZE];

    int ROW = blockIdx.y*sec_size+threadIdx.y;
    int COL = blockIdx.x*sec_size+threadIdx.x;
    float tmpSum = 0;

    for(int j = 0; j < N/sec_size; j++){
        AS[threadIdx.y * sec_size + threadIdx.x] = A[ROW * N + (j * sec_size + threadIdx.x)];
        BS[threadIdx.y * sec_size + threadIdx.x] = B[COL + (j * sec_size * N) + (threadIdx.y*N)];
        __syncthreads();

        for (int i = 0; i < sec_size; i++) {
            tmpSum += AS[threadIdx.y * sec_size + i] * BS[i * sec_size + threadIdx.x];
        }
        __syncthreads();
    }
    C[ROW * N + COL] = tmpSum;
}

// main routine that executes on the host
int main(void)
{
	float *a_h, *a_d;  // Pointer to host & device arrays
	float *mat1_h, *mat2_h, *mat_res_h, *mat1_d, *mat2_d, *mat_res_d;



	const int N = 1<<10;  // Make a big array with 2**N elements
	size_t size = N * sizeof(float);

	const int n_mat = n * n;
    size_t sz = n_mat * sizeof(float);
    /* Auxiliares para medir tiempos */

    cudaEvent_t start, stop;
    float time;

    a_h = (float *)malloc(size);        // Allocate array on host
	cudaMalloc((void **) &a_d, size);   // Allocate array on device

    mat1_h = (float *)malloc(sz);        // Allocate array on host
    mat2_h = (float *)malloc(sz);        // Allocate array on host
    mat_res_h = (float *)malloc(sz);        // Allocate array on host

	cudaMalloc((void **) &mat1_d, sz);   // Allocate array on device
	cudaMalloc((void **) &mat2_d, sz);   // Allocate array on device
	cudaMalloc((void **) &mat_res_d, sz);   // Allocate array on device


    // Initialize host array and copy it to CUDA device
	for (int i=0; i<N; i++){

        a_h[i] = (float)i;

    }
    for (int i = 0; i < n_mat; ++i){

    	mat1_h[i] = i % 8;
    	mat2_h[i] = i % 8;
    	mat_res_h[i] = 0;
    }


    printf("mats:\n");
    for (int i = 0; i < n_mat; ++i){

    	if(i%n == 0)
    		printf("\n");
    	printf("%.2f ", mat1_h[i] );
    }

	cudaMemcpy(mat1_d, mat1_h, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(mat2_d, mat2_h, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(mat_res_d, mat_res_h, sz, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy");

	// Create timer for timing CUDA calculation
	//PPunsigned int timer = 0;
	//PPcutCreateTimer( &timer );
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // Set number of threads and blocks
	//int n_threads_per_block = 128;//1<<9;  // 512 threads per block
	//int n_blocks = 256;//1<<10;  // 1024 blocks

	// Do calculation on device

	cudaEventRecord(start,0);
	//matrix_mult <<< n_blocks, n_threads_per_block >>> (mat1_d, mat2_d, mat_res_d, n);

	dim3 threadsPerBlock(n, n);
    dim3 blocksPerGrid(1, 1);

    if (n*n > 512){
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(n)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(n)/double(threadsPerBlock.y));
    }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(mat1_d, mat2_d, mat_res_d, n, threadsPerBlock.x);


	cudaDeviceSynchronize();  // Wait for matrix_mult to finish on CUDA

    checkCUDAError("kernel invocation");


	// Retrieve result from device and store it in host array
	cudaMemcpy(mat1_h, mat1_d, sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(mat2_h, mat2_d, sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_res_h, mat_res_d, sz, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime( &time, start, stop );

	// Print some of the results
	//for (int i=0; i<N; i+=N/50) printf("%d %f\n", i, a_h[i]);

    // Imprime tiempo de ejecución
    printf("\n\nTIEMPO DE EJECUCIÓN: %f mSeg\n\n", time);

	printf("res:\n");
    for (int i = 0; i < n_mat; ++i)
    {
    	if(i%n == 0)
    		printf("\n");
    	printf("%.2f ", mat_res_h[i] );
    }

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

	free(mat1_h);
	free(mat2_h);
	free(mat_res_h);

	cudaFree(mat1_d);
	cudaFree(mat2_d);
	cudaFree(mat_res_d);
}

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
