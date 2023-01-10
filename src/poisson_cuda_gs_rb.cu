/*
  Created and Maintained by Teng Jiang @ Peking University.
  Date: 30/11/2021
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>
#define checkCudaErrors( a ) do { \
if (cudaSuccess != (a)) { \
fprintf(stderr, "Cuda runtime error in line %d of file %s \
: %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
exit(EXIT_FAILURE); \
} \
} while(0);


#define N 512
#define MAXITER 100
#define MINITER 34
#define RTOL 1e-6
#define PI 3.14159265358979323846


void init_sol_h(double *__restrict__ b, double *__restrict__ u_exact, double *__restrict__ u)
{
    double a = N / 4.;
    double h = 1. / (N + 1);
#pragma omp parallel for
    for (int i = 0; i < N + 2; i++)
        for (int j = 0; j < N + 2; j++)
            for (int k = 0; k < N + 2; k++)
            {
                u_exact[i * (N + 2) * (N + 2) + j * (N + 2) + k] = sin(a * PI * i * h) * sin(a * PI * j * h) * sin(a * PI * k * h);
                u[i * (N + 2) * (N + 2) + j * (N + 2) + k] = 0.;
            }
            
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                b[i * N * N + j * N + k] = 3. * a * a * PI * PI * sin(a * PI * (i + 1) * h) * sin(a * PI * (j + 1) * h) * sin(a * PI * (k + 1) * h) * h * h;
            }
}

double error(double *__restrict__ u, double *__restrict__ u_exact)
{
    double tmp = 0;
#pragma omp parallel for reduction(+:tmp)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] - u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    double tmp2 = 0;
#pragma omp parallel for reduction(+:tmp2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp2 += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    return pow(tmp, 0.5) / pow(tmp2, 0.5);
}

__global__
void gauss_seidel_kernel(double *__restrict__ d_u, double *__restrict__ d_b,int num){
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int i = blockDim.z * blockIdx.z + threadIdx.z;
   
    if((i+j+k+num)%2==0){ // 判断颜色
    d_u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] = 
                                (d_b[i * N * N + j * N + k] 
                                + d_u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]  // from last iter
                                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1]
                                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0] // from last iter
                                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2]
                                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1]  // from last iter
                                + d_u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                                ) / 6.0;}
    //__syncthreads();
}



__global__
void residual_norm_kernel(double *__restrict__ u, double *__restrict__ b,double *__restrict__ rn){
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int i = blockDim.z * blockIdx.z + threadIdx.z;
    double r = b[i * N * N + j * N + k] + 
                    + u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1]
                    + u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    - 6.0 * u[(i + 1) * ((N + 2) * (N + 2)) + (j + 1) * (N + 2) + (k + 1)];
    rn[i * N * N + j * N + k]=r*r;
    __syncthreads();
}

double residual_norm(double *__restrict__ u, double *__restrict__ b)
{
    double norm2 = 0;

    #pragma omp parallel for collapse(3) reduction(+:norm2)
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k++)
            {
                double r = b[i * N * N + j * N + k] + 
                    + u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1]
                    + u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    - 6.0 * u[(i + 1) * ((N + 2) * (N + 2)) + (j + 1) * (N + 2) + (k + 1)];
                norm2 += r * r;
            }
        }
    }
    return sqrt(norm2);
}

__global__ void reduction(double *g_odata, double *g_idata)
{
    // dynamically allocated shared memory

    __shared__  double temp[N];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid+blockIdx.x*blockDim.x];

    // next, we perform binary tree reduction

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[blockIdx.x] = temp[0];
}

int main(int argc, char **argv)
{
    double * u = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
    double * u_exact = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
    double * b = (double *)malloc(sizeof(double) * N * N * N);
    double * h_recv = (double *)malloc(sizeof(double));

    init_sol_h(b, u_exact, u);
    double normr0 = residual_norm(u, b); // Please ensure that this residual_norm is exact.
    double normr = normr0;

    printf("Method: Gauss Seidel with red and black coloring\n");
    
    // Allocate device memory and transfer data
    double  *d_u_exact, *d_u, *d_b,*d_u_new, *d_rn, *d_rn_out; // d_u_new for jacobi, d_rn for residual norms, d_rn_out for results after reduction
    cudaMalloc((void**)&d_u,sizeof(double)* (N + 2) * (N + 2) * (N + 2));
    cudaMalloc((void**)&d_u_new,sizeof(double)* (N + 2) * (N + 2) * (N + 2));
    cudaMalloc((void**)&d_b,sizeof(double)*N * N * N);
    cudaMalloc((void**)&d_u_exact,sizeof(double)* (N + 2) * (N + 2) * (N + 2));
    cudaMalloc((void**)&d_rn,sizeof(double)*  N*N*N);
    cudaMalloc((void**)&d_rn_out,sizeof(double)*  N*N*N);

    printf("cuda malloc successful\n");
    cudaMemset(d_rn,0,sizeof(double)* N*N*N); 
    cudaMemset(d_rn_out,0,sizeof(double)*  N*N*N); 
    cudaMemset(d_u,0,sizeof(double)* (N + 2) * (N + 2) * (N + 2)); 
    cudaMemset(d_u_new,0,sizeof(double)* (N + 2) * (N + 2) * (N + 2)); 
    cudaMemcpy(d_u_exact, u_exact,sizeof(double)* (N + 2) * (N + 2) * (N + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b,sizeof(double)*N* N* N, cudaMemcpyHostToDevice);
    /*
    checkCudaErrors(cudaMemset(d_rn,0,sizeof(double)* N*N*N)); 
    checkCudaErrors(cudaMemset(d_rn_out,0,sizeof(double)*  N*N*N)); 
    checkCudaErrors(cudaMemset(d_u,0,sizeof(double)* (N + 2) * (N + 2) * (N + 2))); 
    checkCudaErrors(cudaMemset(d_u_new,0,sizeof(double)* (N + 2) * (N + 2) * (N + 2))); 
    checkCudaErrors(cudaMemcpy(d_u_exact, u_exact,sizeof(double)* (N + 2) * (N + 2) * (N + 2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b,sizeof(double)*N* N* N, cudaMemcpyHostToDevice));
    */

    printf("cuda copy and set successful\n");

    dim3 grid_dim(N/32, N/8, N/4);
    dim3 block_dim(32,8,4);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    
    
    int tsteps = MAXITER;
    cudaEventRecord(start, 0);
    for (int k = 0; k < MAXITER; k++)
    {   
        printf("Iteration %d, normr/normr0=%g\n", k,  normr/ normr0);

        gauss_seidel_kernel<<<grid_dim, block_dim>>>(d_u, d_b,1);
        gauss_seidel_kernel<<<grid_dim, block_dim>>>(d_u, d_b,0);

        residual_norm_kernel<<<grid_dim, block_dim>>>(d_u, d_b, d_rn);
        //checkCudaErrors(cudaDeviceSynchronize());
        reduction<<<N*N,N>>>(d_rn_out,d_rn);
        //checkCudaErrors(cudaDeviceSynchronize());
        reduction<<<N,N>>>(d_rn,d_rn_out);
        //checkCudaErrors(cudaDeviceSynchronize());
        reduction<<<1,N>>>(d_rn_out,d_rn);
        
        cudaMemcpy(h_recv, d_rn_out ,sizeof(double), cudaMemcpyDeviceToHost);
        
        normr = sqrt(h_recv[0]);
        if (normr < RTOL * normr0 && k+1>=MINITER)
        {
            printf("Iteration %d, normr/normr0=%g\n", k + 1, normr/ normr0);
            tsteps = k + 1;
            printf("Converged with %d iterations.\n", tsteps);
            break;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //printf("time: %g\n", elapsedTime);
    printf("time: %g\n", elapsedTime / 1000.);

    cudaMemcpy(u, d_u ,sizeof(double)* (N + 2) * (N + 2) * (N + 2), cudaMemcpyDeviceToHost);
    printf("  grid  dim:  %d, %d, %d.\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("  block dim: %d, %d, %d.\n", block_dim.x, block_dim.y, block_dim.z);
    cudaFree(d_u_exact); cudaFree(d_u); cudaFree(d_b); cudaFree(d_rn); cudaFree(d_rn_out); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Residual norm: %g\n", normr);

    double final_normr = residual_norm(u, b); // Please ensure that this residual_norm is exact.
    printf("Final residual norm: %g\n", final_normr);
    printf("|r_n|/|r_0| = %g\n", final_normr / normr0);

    long long residual_norm_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + (N * N * N)) * tsteps;
    long long gs_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + 2 * (N * N * N)) * tsteps;

    long long total_bytes = residual_norm_bytes + gs_bytes;
    double bandwidth = total_bytes / elapsedTime * 1000.;

    printf("total bandwidth: %g GB/s\n", bandwidth / (double)(1 << 30));

    double relative_err = error(u, u_exact);
    printf("relative error: %g\n", relative_err);
    
    //int num_threads = omp_get_max_threads();
    //printf("openmp max num threads: %d\n", num_threads);

    free(u);
    free(u_exact);
    free(b);

    return 0;
}
