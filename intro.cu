/*
 *
 * Programa de Introducción a los conceptos de CUDA
 * Mariana Hernández
 * Alan Córdova
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>

/* Declaración de métodos/


/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

/* Kernel para sumar dos vectores en un sólo bloque de hilos */
__global__ void vect_add(int *d_a, int *d_b, int *d_c)
{
    /* Part 2B: Implementación del kernel para realizar la suma de los vectores en el GPU */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < N)
        d_c[i] = d_a[i] + d_b[i];
}

/* Versión de múltiples bloques de la suma de vectores */
__global__ void vect_add_multiblock(int *d_a, int *d_b, int *d_c)
{
    /* Part 2C: Implementación del kernel pero esta vez permitiendo múltiples bloques de hilos. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < N)
        d_c[i] = d_a[i] + d_b[i];
}

/* Numero de elementos en el vector */
#define ARRAY_SIZE 256

/*
 * Número de bloques e hilos
 * Su producto siempre debe ser el tamaño del vector (arreglo).
 */
#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

/* Main routine */
int main(int argc, char *argv[])
{
    int *a, *b, *c, *d; /* Arreglos del CPU */
    int *d_a, *d_b, *d_c, *d_d;/* Arreglos del GPU */

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    a = (int *) malloc(sz);
    b = (int *) malloc(sz);
    c = (int *) malloc(sz);
    d = (int *) malloc(sz);

    /*
     * Parte 1A:Reservar memoria en el GPU
     */
    cudaMalloc(&d_a, sz);
    cudaMalloc(&d_b, sz);
    cudaMalloc(&d_c, sz);
    cudaMalloc(&d_d, sz);

    /* inicialización */
    for (i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = ARRAY_SIZE - i;
        c[i] = 0;
        d[i] = 0;
    }

    /* Parte 1B: Copiar los vectores del CPU al GPU */
    cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, sz, cudaMemcpyHostToDevice);

    /* run the kernel on the GPU */
    /* Parte 2A: Configurar y llamar los kernels */
    /* dim3 dimGrid( ); */
    /* dim3 dimBlock( ); */
    /* vect_add<<< , >>>( ); */

    //invocamos kernel
    int threadsPerBlock = 64; // ARRAY_SIZE/NUM_BLOCKS
    int blocksPerGrid = 4; // nuevo NUM_BLOCKS

    // Para obtener tiempos de ejecucion del kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vect_add<<< NUM_BLOCKS , THREADS_PER_BLOCK >>> (d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de vect_add: \t %f \n", milliseconds);


    cudaEventRecord(start);
    vect_add_multiblock<<< blocksPerGrid , threadsPerBlock >>> (d_a, d_b, d_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de vect_add_multiblock (4 bloques): \t %f \n", milliseconds);

    /* Esperar a que todos los threads acaben y checar por errores */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(a, d_a, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d_d, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", c[i]);
    }

    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", d[i] );
    }
    printf("\n\n");

    /* Parte 1D: Liberar los arreglos */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    free(a);
    free(b);
    free(c);
    free(d);

    return 0;
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
