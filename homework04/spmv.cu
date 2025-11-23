#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"

template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals,
                int m, int n, int n_new,
                const double* x, double* b)
{
    extern __shared__ double store[];
    // ... initializations ...
    unsigned int block_size = blockDim.x;
    unsigned int row  = blockIdx.x;
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid % warpSize;
    unsigned int row_start = row * n_new;

    double contrib = 0.0; // Correctly initialized

    for(int i = 0; i < n_new; i += block_size){
        
        if (tid + i < n_new){
            unsigned col = col_ind[row_start + tid + i];
            double v = static_cast<double>(vals[row_start + tid + i]); 

            if (col < n) {
                contrib += x[col] * v;
            } 
        }   
    }

    unsigned mask = 0xffffffff;

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        contrib += __shfl_down_sync(mask, contrib, offset);
    }
    
    if (lane == 0) {
        store[tid / warpSize] = contrib;
    }
    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            sum += store[i];
        }
        b[row] = sum; 
    }
}



void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int n_new, 
                  double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    unsigned int blocks = m; 
    unsigned int threads = 128; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, n_new, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}

void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
    int n_new, double* x, unsigned int** dev_col_ind, 
    double** dev_vals, double** dev_x, double** dev_b)
{
    cudaMalloc(dev_col_ind, sizeof(unsigned int) * m * n_new);  // Fixed
    cudaMalloc(dev_vals, sizeof(double) * m * n_new);
    cudaMalloc(dev_x, sizeof(double) * m);
    cudaMalloc(dev_b, sizeof(double) * m);
    double* copy_arr = (double*)calloc(m, sizeof(double));

    cudaMemcpy(*dev_col_ind, col_ind, sizeof(unsigned int) * m * n_new, cudaMemcpyHostToDevice);  // Fixed
    cudaMemcpy(*dev_vals, vals, sizeof(double) * m * n_new, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_x, x, sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_b, copy_arr, sizeof(double) * m, cudaMemcpyHostToDevice);

    free(copy_arr);
}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
    double* vals, int m, int n, int nnz, double* x, 
    unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
    double** dev_vals, double** dev_x, double** dev_b)
{
    cudaMalloc(dev_row_ptr, sizeof(unsigned int) * (m + 1));
    cudaMalloc(dev_col_ind, sizeof(unsigned int) * nnz);
    cudaMalloc(dev_vals, sizeof(double) * nnz);
    cudaMalloc(dev_x, sizeof(double) * m);
    cudaMalloc(dev_b, sizeof(double) * m);
    double* copy_arr = (double*)calloc(m, sizeof(double));

    cudaMemcpy(*dev_row_ptr, row_ptr, sizeof(unsigned int) * (m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_col_ind, col_ind, sizeof(unsigned int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_vals, vals, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_x, x, sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_b, copy_arr, sizeof(double) * m, cudaMemcpyHostToDevice);

    free(copy_arr);
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
            int m, int n, int nnz, double* x, double* b)
{
    extern __shared__ double store[];

    unsigned int block_size = blockDim.x;
    unsigned int row  = blockIdx.x;
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid % warpSize;
    unsigned int row_start = row_ptr[row];
    unsigned int row_end = row_ptr[row + 1];
    
    unsigned int row_nnz = row_end - row_start; 

    double contrib = 0.0; 

    for(int i = 0; i < row_nnz; i += block_size)
    {
        unsigned int local_idx = tid + i;
        
        if (local_idx < row_nnz)
        {
            unsigned int global_idx = row_start + local_idx;
            
            unsigned int col = col_ind[global_idx];
            double v = static_cast<double>(vals[global_idx]); 

            if (col < n) {
                contrib += x[col] * v;
            }
        }
    }

    unsigned mask = 0xffffffff;

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        contrib += __shfl_down_sync(mask, contrib, offset);
    }

    if (lane == 0) {
        store[tid / warpSize] = contrib;
    }
    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            sum += store[i];
        }
        b[row] = sum; 
    }
}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 thread block per row
    // 64 threads working on the non-zeros on the same row
    unsigned int blocks = m; 
    unsigned int threads = 128; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}
