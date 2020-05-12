/*
 *  Ejercicio 4 Práctica 4: CUDA
 *  Mariana Hernández
 *  Alan Córdova
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

# define NPOINTS 2000
# define MAXITER 2000

#define ARRAY_SIZE 256
#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

struct complex{
  double real;
  double imag;
};

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

/* Kernel para sumar generar puntos muestra */
__global__
void maldel_gen_points(complex *d_out, int *numout_out)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < NPOINTS*NPOINTS; i += stride) {
        d_out[i].real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
        d_out[i].imag = 1.125*(double)(i)/(double)(NPOINTS)+1.0e-7;
        for (int iter=0; iter<MAXITER; iter++){
            double ztemp;
            struct complex z;
            z = d_out[i];
            ztemp=(z.real*z.real)-(z.imag*z.imag)+d_out[i].real;
            z.imag=z.real*z.imag*2+d_out[i].imag;
            z.real=ztemp;
            if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
                numout_out[i]++;
                break;
            }
        }
    }
}

/*
 *
 */
int main(int argc, char** argv) {

    complex *cnumbers;
    complex *d_cnumbers;
    int *numout;
    int *d_numout;

    size_t sz1 = NPOINTS*NPOINTS * sizeof(complex);
    size_t sz2 = NPOINTS*NPOINTS * sizeof(int);

    cnumbers = (complex *) malloc (sz1);
    cudaMalloc(&d_cnumbers, sz1);

    numout = (int *) malloc (sz2);
    cudaMalloc(&d_numout, sz2);

    // inicialización
    for (int i = 0; i < NPOINTS*NPOINTS; i++) {
        numout[i] = 0;
    }

    // copia host to device
    cudaMemcpy(d_numout, numout, sz2, cudaMemcpyHostToDevice);

    //invocamos kernel
    int blockSize = 256;
    int numBlocks = (NPOINTS*NPOINTS + blockSize - 1) / blockSize;

    // Para obtener tiempos de ejecucion del kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    maldel_gen_points<<< numBlocks, blockSize >>>(d_cnumbers, d_numout);
    cudaDeviceSynchronize();
    checkCUDAError("kernel invocation");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de maldel_gen_points: \t %f \n", milliseconds);

    cudaMemcpy(cnumbers, d_cnumbers, sz1, cudaMemcpyDeviceToHost);
    cudaMemcpy(numout, d_numout, sz2, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    int res = 0;
    for(int i = 0; i < NPOINTS*NPOINTS; i++){
        res += numout[i];
    }
    /* print out the result */
    double area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-res)/(double)(NPOINTS*NPOINTS);
    double error=area/(double)NPOINTS;

    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
    //printf("Tiempo de ejecución: %f segundos \n",difftime(t2,t1));

    cudaFree(d_cnumbers);
    free(cnumbers);
    cudaFree(d_numout);
    free(numout);


    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
