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
#include <mpi.h>

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

__global__ static void evolve(double *un, double *uc, double *uo,
  double *pebbles, int *n, double *h, double *dt, double *t, int *n_blocks,
  int *n_threads, int *rank)
{
  int i, j;
  unsigned int bid = blockDim.x * blockIdx.x + threadIdx.x;

  if(*rank==0)
  {
    i = (bid/(*n/2));
    j = (bid%(*n/2));
  }
  else if (*rank==1)
  {
    i = (bid/(*n/2));
    j = (bid%(*n/2)+*n/2);
  }
  else if (*rank==2)
  {
    i = (bid/(*n/2)+*n/2);
    j = (bid%(*n/2));
  }
  else if (*rank==3)
  {
    i = (bid/(*n/2)+*n/2);
    j = (bid%(*n/2)+*n/2);
  }
//   if(tid==0 && *rank==3)
// printf("Rank = %d i=%d j= %d      \n",*rank,i,j);

  int idx = i * *n + j;
  // printf("%d\n", *n);
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

      // if(*rank ==2)
      //  printf("%f i = %d j= %d        .\n", un[idx], i, j);
    }

    __syncthreads();
    // if(*rank ==2)
    //  printf("%f i = %d j= %d        .\n", un[idx], i, j);

    uo[idx] = uc[idx];
    uc[idx] = un[idx];

    (*t) = (*t) + *dt;
}

void Send_Row(int start_row, int start_col, int n, int dest, double *u)
{
  double msg[2][n];
  for(int i=start_row; i<=start_row + 1; i++)
  {
    for(int j=0; j<=n-1; j++)
    {
      msg[i-start_row][j] = u[i*2*n + j + start_col];
    }
  }
  MPI_Send(msg, 2*n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}

void Send_Col(int start_row, int start_col, int n, int dest, double *u)
{
  double msg[n][2];
  for(int i=0; i<=n-1; i++)
  {
    for(int j=start_col; j<=start_col + 1; j++)
    {
      msg[i][j-start_col] = u[(i+start_row)*2*n + j ];
    }
  }
  MPI_Send(msg, 2*n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}

void Recv_Row(int start_row, int start_col, int n, int source, double *u)
{
  double msg[2][n];
  MPI_Recv(msg, 2*n, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(int i=start_row; i<=start_row + 1; i++)
  {
    for(int j=0; j<=n-1; j++)
    {
      u[i*2*n + j + start_col] = msg[i-start_row][j] ;
    }
  }
}

void Recv_Col(int start_row, int start_col, int n, int source, double *u)
{
  double msg[n][2];
  MPI_Recv(msg, 2*n, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(int i=0; i<=n - 1; i++)
  {
    for(int j=start_col; j<=start_col + 1; j++)
    {
       u[(i+start_row)*2*n + j ] = msg[i][j-start_col];
    }
  }
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n,
  double h, double end_time, int nthreads, int rank)
{
	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */

  double t=0., dt = h / 2.;
  int blocks = (int)pow(n / nthreads, 2) / 4;
  int threads = nthreads * nthreads;
  int r_min, r_max, c_min, c_max;

  int *blocks_d, *threads_d, *n_d, *rank_d;
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
  cudaMalloc( (void **) &rank_d, sizeof(int) * 1 );

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
  CUDA_CALL(cudaMemcpy( rank_d, &rank, sizeof(int) * 1, cudaMemcpyHostToDevice ));

  if(rank == 0)
  {
    r_min = 0;
    r_max = n / 2 + 1;
    c_min = 0;
    c_max = n / 2 + 1;
  }
  else if (rank == 1)
  {
    r_min = 0;
    r_max = n / 2 + 1;
    c_min = n / 2 - 2;
    c_max = n - 1;
  }
  else if (rank == 2)
  {
    r_min = n / 2 - 2;
    r_max = n - 1;
    c_min = 0;
    c_max = n / 2 + 1;
  }
  else if (rank == 3)
  {
    r_min = n / 2 - 2;
    r_max = n - 1;
    c_min = n / 2 - 2;
    c_max = n - 1;
  }

  while(1)
  {
     evolve<<< blocks, threads >>>(un_d, uc_d, uo_d, pebs_d, n_d, h_d, dt_d,
       t_d, blocks_d, threads_d, rank_d);

    CUDA_CALL(cudaMemcpy( u, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost ));

     if(!tpdt(&t,dt,end_time))
        break;

        // if (rank==0)
        // {
        //   Send_Row(r_max-3, 0, n/2, 2, u);
        //   Recv_Row(r_max-1, 0, n/2, 2, u);
        //   Send_Col(0, c_max-3, n/2, 1, u);
        //   Recv_Col(0, c_max-1, n/2, 1, u);
        //   MPI_Send(u+(r_max-2)*n+c_max-2, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
        //   MPI_Recv(u+(r_max-1)*n+c_max-1, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }
        // else if(rank==1)
        // {
        //   Recv_Col(0, c_min, n/2, 0, u);
        //   Send_Col(0, c_min+2, n/2, 0, u);
        //   Send_Row(r_max-3, n/2, n/2, 3, u);
        //   Recv_Row(r_max-1, n/2, n/2, 3, u);
        //   MPI_Send(u+(r_max-2)*n+c_min+2, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
        //   MPI_Recv(u+(r_max-1)*n+c_min+1, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }
        // else if(rank==2)
        // {
        //   Recv_Row(r_min, 0, n/2, 0, u);
        //   Send_Row(r_min+2, 0, n/2, 0, u);
        //   Send_Col(n/2, c_max-3, n/2, 3, u);
        //   Recv_Col(n/2, c_max-1, n/2, 3, u);
        //   MPI_Recv(u+(r_min+1)*n+c_max-1, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //   MPI_Send(u+(r_min+2)*n+c_max-2, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        // }
        // else if(rank==3)
        // {
        //   Recv_Row(r_min, n/2, n/2, 1, u);
        //   Send_Row(r_min+2, n/2, n/2, 1, u);
        //   Recv_Col(n/2, c_min, n/2, 2, u);
        //   Send_Col(n/2, c_min+2, n/2, 2, u);
        //   MPI_Recv(u+(r_min+1)*n+c_min+1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //   MPI_Send(u+(r_min+2)*n+c_min+2, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        // }
    if (rank==0)
    {
      Send_Col(0, c_max-3, n/2, 1, u);
      Send_Row(r_max-3, 0, n/2, 2, u);
      Recv_Col(0, c_max-1, n/2, 1, u);
      Recv_Row(r_max-1, 0, n/2, 2, u);
      MPI_Send(u+(r_max-2)*n+c_max-2, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
      MPI_Recv(u+(r_max-1)*n+c_max-1, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(rank==1)
    {
      Recv_Col(0, c_min, n/2, 0, u);
      Send_Col(0, c_min+2, n/2, 0, u);
      Send_Row(r_max-3, n/2, n/2, 3, u);
      Recv_Row(r_max-1, n/2, n/2, 3, u);
      MPI_Send(u+(r_max-2)*n+c_min+2, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
      MPI_Recv(u+(r_max-1)*n+c_min+1, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(rank==2)
    {
      Recv_Row(r_min, 0, n/2, 0, u);
      Send_Row(r_min+2, 0, n/2, 0, u);
      Send_Col(n/2, c_max-3, n/2, 3, u);
      Recv_Col(n/2, c_max-1, n/2, 3, u);
      MPI_Recv(u+(r_min+1)*n+c_max-1, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(u+(r_min+2)*n+c_max-2, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if(rank==3)
    {
      Recv_Row(r_min, n/2, n/2, 1, u);
      Recv_Col(n/2, c_min, n/2, 2, u);
      Send_Row(r_min+2, n/2, n/2, 1, u);
      Send_Col(n/2, c_min+2, n/2, 2, u);
      MPI_Recv(u+(r_min+1)*n+c_min+1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(u+(r_min+2)*n+c_min+2, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
  if(rank==0)
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
  cudaFree(rank_d);
	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
