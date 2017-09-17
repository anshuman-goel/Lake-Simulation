/******************************************************************************
* FILE: lake.cu
* DESCRIPTION:
*
* Users will supply the functions
* i.) fn(x) - the polynomial function to be analyized
* ii.) dfn(x) - the true derivative of the function
* iii.) degreefn() - the degree of the polynomial
*
* The function fn(x) should be a polynomial.
*
* Group Info:
* agoel5 Anshuman Goel
* kgondha Kaustubh Gondhalekar
* ndas Neha Das
*
* LAST REVISED: 9/13/2017
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define VSQR 0.1
#define TSCALE 1.0

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/

extern int tpdt(double *t, double dt, double end_time);

__device__ double fn(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

__global__ static void evolve(double *un, double *uc, double *uo, double *pebbles, int *n, double *h, double *dt, double *t, int *n_blocks, int *n_threads)
{
  int i, j;
  const unsigned int tid = threadIdx.x;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int gid = idx;

  int leftover = (*n * *n) - (*n_threads * *n_blocks);

  i = idx / *n;
  j = idx % *n;

  if( i == 0 || i == *n - 1 || j == 0 || j == *n - 1 ||
      i == *n - 2 || i == 1 || j == *n - 2 || j == 1)
  {
    un[idx] = 0.;
  }
  else
  {
    // printf("%d\n", gid);

    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(*dt * *dt) *
    ((  uc[idx-1] // WEST
        + uc[idx+1] // EAST
        + uc[idx + *n] // SOUTH
        + uc[idx - *n] // NORTH
       + 0.25*( uc[idx - *n - 1 ] // NORTHWEST
              + uc[idx - *n + 1 ] // NORTHEAST
              + uc[idx + *n - 1 ] // SOUTHWEST
              + uc[idx + *n + 1 ] // SOUTHEAST
              )
      + 0.125*( uc[idx - 2 ]  // WESTWEST
              + uc[idx + 2 ] // EASTEAST
              + uc[idx - 2 * *n ] // NORTHNORTH
              + uc[idx + 2 * *n ] // SOUTHSOUTH
              )
      - 6 * uc[idx])/(*h * *h) + fn(pebbles[idx],*t));
    }

    if (blockIdx.x == *n_blocks - 1 && tid == 0 && leftover > 0)
    {
      for( idx = *n * *n - 1; idx>= *n * *n - leftover; idx--)
      {
        i = idx / *n;
        j = idx % *n;
        if( i == 0 || i == *n - 1 || j == 0 || j == *n - 1 ||
            i == *n - 2 || i == 1 || j == *n - 2 || j == 1)
        {
          un[idx] = 0.;
        }
        else
        {

          un[idx] = 2*uc[idx] - uo[idx] + VSQR *(*dt * *dt) *
          ((  uc[idx-1] // WEST
              + uc[idx+1] // EAST
              + uc[idx + *n] // SOUTH
              + uc[idx - *n] // NORTH
             + 0.25*( uc[idx - *n - 1 ] // NORTHWEST
                    + uc[idx - *n + 1 ] // NORTHEAST
                    + uc[idx + *n - 1 ] // SOUTHWEST
                    + uc[idx + *n + 1 ] // SOUTHEAST
                    )
            + 0.125*( uc[idx - 2 ]  // WESTWEST
                    + uc[idx + 2 ] // EASTEAST
                    + uc[idx - 2 * *n ] // NORTHNORTH
                    + uc[idx + 2 * *n ] // SOUTHSOUTH
                    )
            - 6 * uc[idx])/(*h * *h) + fn(pebbles[idx],*t));
          }

      }
    }
    __syncthreads();
    uo[gid] = uc[gid];
    uc[gid] = un[gid];
    if (blockIdx.x == *n_blocks - 1 && tid == 0 && leftover > 0)
    {
      for( idx = *n * *n - 1; idx>= *n * *n - leftover; idx--)
      {
        uo[idx] = uc[idx];
        uc[idx] = un[idx];
      }
    }

    (*t) = (*t) + *dt;
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */

  double t=0., dt = h / 2.;
  int blocks = (int)pow(n / nthreads, 2);
  int threads = nthreads * nthreads;

  int *blocks_d, *threads_d, *n_d;
  double *un_d, *uc_d, *uo_d, *pebs_d, *t_d, *dt_d, *h_d;

  if (nthreads > n)
  {
    printf("Choose threads less than grid dimension\n");
    return;
  }

  cudaMalloc( (void **) &un_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &uc_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &uo_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &pebs_d, sizeof(double) * n * n);
  cudaMalloc( (void **) &blocks_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &threads_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &n_d, sizeof(int) * 1 );
  cudaMalloc( (void **) &t_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &dt_d, sizeof(double) * 1 );
  cudaMalloc( (void **) &h_d, sizeof(double) * 1 );

        /* Set up device timers */
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */

  CUDA_CALL(cudaMemcpy( uc_d, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( un_d, u, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( uo_d, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( pebs_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( blocks_d, &blocks, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( threads_d, &threads, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( n_d, &n, sizeof(int) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( h_d, &h, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( dt_d, &dt, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  CUDA_CALL(cudaMemcpy( t_d, &t, sizeof(double) * 1, cudaMemcpyHostToDevice ));

  while(1)
  {

     evolve<<< blocks, threads >>>(un_d, uc_d, uo_d, pebs_d, n_d, h_d, dt_d, t_d, blocks_d, threads_d);

     if(!tpdt(&t,dt,end_time))
        break;

    CUDA_CALL(cudaMemcpy( t_d, &t, sizeof(double) * 1, cudaMemcpyHostToDevice ));
  }
  CUDA_CALL(cudaMemcpy( u, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost ));

        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */

  cudaFree(un_d);
  cudaFree(uc_d);
  cudaFree(uo_d);
  cudaFree(blocks_d);
  cudaFree(threads_d);
  cudaFree(pebs_d);
  cudaFree(n_d);
  cudaFree(t_d);
  cudaFree(h_d);
  cudaFree(dt_d);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
