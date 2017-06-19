#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include "../cuda/shared.h"

#define NDBLOCKS 3
#define NVARIABLES 2
#define GB (1024LLU*1024LLU*1024LLU)

__global__ void init(const size_t n, double* x, double* y);
__global__ void compute(const size_t n, double* x, double* y);
__global__ void validate(const size_t n, double* x, double* y);

int main()
{
  // Calculate the space on host
  const size_t pages = sysconf(_SC_PHYS_PAGES);
  const size_t page_size = sysconf(_SC_PAGE_SIZE);
  size_t free_dram_memory = pages*page_size;
  printf("DRAM Memory Capacity Available. Free = %llu\n", free_dram_memory);

  // Fudging a value in for testing
  const size_t ablock_bytes = free_dram_memory*0.8;
  const size_t ablock_space_len = ablock_bytes/sizeof(double);
  double* ablock_space = (double*)malloc(ablock_bytes);
  gpu_check(cudaHostRegister(ablock_space, ablock_bytes, 0));

  double* ablock[NVARIABLES];
  const size_t ablock_len = ablock_space_len/NVARIABLES;
  for(size_t vv = 0; vv < NVARIABLES; ++vv) {
    ablock[vv] = &ablock_space[vv*ablock_len];
  }
  printf("Application Data Block Length %llu\n", ablock_len);

  for(size_t ii = 0; ii < NVARIABLES*ablock_len; ++ii) {
    ablock_space[ii] = 1.0;
  }

  // Work out how large data staging blocks are available
  size_t free_gpu_mem, total_gpu_mem;
  cudaMemGetInfo(&free_gpu_mem, &total_gpu_mem);

  // Fudging a value in for testing
  printf("GPU Memory Capacity Available. Free = %llu\n", free_gpu_mem, total_gpu_mem);
  const size_t dblock_bytes = (free_gpu_mem*0.8)/(NVARIABLES*NDBLOCKS);
  const size_t max_dblock_len = dblock_bytes/sizeof(double);
  printf("Max Data Block Length %llu\n", max_dblock_len);

  // Allocate a validation bit for the validation routine
  int* validation_bit;
  cudaMalloc((void**)&validation_bit, 1);
  gpu_check(cudaDeviceSynchronize());

  // Initialise all of the data staging blocks
  double* dblocks[NVARIABLES][NDBLOCKS];
  for(size_t vv = 0; vv < NVARIABLES; ++vv) {
    for(size_t dd = 0; dd < NDBLOCKS; ++dd) {
      cudaMalloc((void**)&dblocks[vv][dd], dblock_bytes);
      gpu_check(cudaDeviceSynchronize());
    }
  }

  // Create streams for asynchronous copies
  cudaStream_t in_stream;
  cudaStream_t work_stream;
  cudaStream_t out_stream;
  gpu_check(cudaStreamCreateWithFlags(&in_stream,cudaStreamNonBlocking));
  gpu_check(cudaStreamCreateWithFlags(&work_stream,cudaStreamNonBlocking));
  gpu_check(cudaStreamCreateWithFlags(&out_stream,cudaStreamNonBlocking));

  /* BEGIN STAGIN ROUTINE */
  const size_t ndblocks_reqd = ceil(ablock_len/(double)max_dblock_len);
  printf("Requiring %llu Data Blocks\n", ndblocks_reqd);

  for(size_t ii = 0; ii < ndblocks_reqd; ++ii) {
    const size_t in_id =  (ii+2)%NDBLOCKS;
    const size_t on_id =  (ii+1)%NDBLOCKS;
    const size_t out_id = (ii+0)%NDBLOCKS;
    const size_t dblock_len = ((ii+1)*max_dblock_len > ablock_len) 
      ? ablock_len-ii*max_dblock_len : max_dblock_len;

    printf("Stagin: in %llu on %llu out %llu dblock_len %llu\n", 
        in_id, on_id, out_id, dblock_len);

    // Copy on the first block into the 'on' data staging block
    if(ii == 0) {
      for(size_t vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            dblocks[vv][on_id], 
            &ablock[vv][0],
            max_dblock_len*sizeof(double), cudaMemcpyHostToDevice, in_stream);
      }

      // Have to sync here to prepare initial data
      gpu_check(cudaStreamSynchronize(in_stream));
    }

    // If not last iteration, asynchronously stage new blocks
    if(ii < ndblocks_reqd-1) {
      const size_t next_dblock_len = ((ii+2)*max_dblock_len > ablock_len) 
        ? ablock_len-ii*max_dblock_len : max_dblock_len;
      for(size_t vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            dblocks[vv][in_id],
            &ablock[vv][(ii+1)*max_dblock_len],
            next_dblock_len*sizeof(double), cudaMemcpyHostToDevice, in_stream);
      }
    }

    // Perform the operation
    const size_t nblocks = ceil(dblock_len/(double)NTHREADS);
    compute<<<nblocks, NTHREADS, 0, work_stream>>>(
        dblock_len, dblocks[0][on_id], dblocks[1][on_id]);

    // After first iteration, begin copying blocks back
    if(ii > 0) {
      for(size_t vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            &ablock[vv][(ii-1)*max_dblock_len],
            dblocks[vv][out_id],
            max_dblock_len*sizeof(double), cudaMemcpyDeviceToHost, out_stream);
      }
    }

    // Copy back the last computed block
    if(ii == ndblocks_reqd-1) {
      for(size_t vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            &ablock[vv][ii*max_dblock_len],
            dblocks[vv][on_id],
            dblock_len*sizeof(double), cudaMemcpyDeviceToHost, out_stream);
      }
    }
    gpu_check(cudaDeviceSynchronize());
  }


  cudaHostUnregister(ablock);
  gpu_check(cudaStreamDestroy(in_stream));
  gpu_check(cudaStreamDestroy(out_stream));
}

__global__ void init(const size_t n, double* x, double* y)
{
  const size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid < n) {
    x[gid] = 1.0;
    y[gid] = 2.0;
  }
}

__global__ void compute(const size_t n, double* x, double* y)
{
  const size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid < n) {
    x[gid] = x[gid]*y[gid];
  }
}

__global__ void validate(const size_t n, double* x, double* y)
{
  const size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid < n) {
    if(x[gid] != 2) {
      y[0] = 99;
    }
  }
}

#if 0
  for(int ii = 0; ii < ndblocks_reqd; ++ii) {
    const size_t in_id =  (ii+2)%NDBLOCKS;
    const size_t on_id =  (ii+1)%NDBLOCKS;
    const size_t out_id = (ii+0)%NDBLOCKS;
    const size_t dblock_len = ((ii+1)*max_dblock_len > ablock_len) 
      ? ablock_len-ii*max_dblock_len : max_dblock_len;

    printf("Stagin: in %llu on %llu out %llu dblock_len %llu\n", 
        in_id, on_id, out_id, dblock_len);

    // Copy on the first block into the 'on' data staging block
    if(ii == 0) {
      for(int vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            dblocks[vv][on_id], 
            &ablock[vv][0],
            max_dblock_len*sizeof(int), cudaMemcpyHostToDevice, in_stream);
      }

      // Have to sync here to prepare initial data
      gpu_check(cudaStreamSynchronize(in_stream));
    }
    gpu_check(cudaDeviceSynchronize());

    // If not last iteration, asynchronously stage new blocks
    if(ii < ndblocks_reqd-1) {
      const size_t next_dblock_len = ((ii+2)*max_dblock_len > ablock_len) 
        ? ablock_len-ii*max_dblock_len : max_dblock_len;
      for(int vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            dblocks[vv][in_id],
            &ablock[vv][(ii+1)*max_dblock_len],
            next_dblock_len*sizeof(int), cudaMemcpyHostToDevice, in_stream);
      }
    }
    gpu_check(cudaDeviceSynchronize());

    // Perform the operation
    const size_t nblocks = ceil(dblock_len/(double)NTHREADS);
    compute<<<nblocks, NTHREADS>>>(
        dblock_len, dblocks[0][on_id], dblocks[1][on_id]);
    gpu_check(cudaDeviceSynchronize());

    // After first iteration, begin copying blocks back
    if(ii > 0) {
      for(int vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            &ablock[vv][(ii-1)*max_dblock_len],
            dblocks[vv][out_id],
            max_dblock_len*sizeof(int), cudaMemcpyDeviceToHost, out_stream);
      }
    }
    gpu_check(cudaDeviceSynchronize());

    // Copy back the last computed block
    if(ii == ndblocks_reqd-1) {
      for(int vv = 0; vv < NVARIABLES; ++vv) {
        cudaMemcpyAsync(
            &ablock[vv][ii*max_dblock_len],
            dblocks[vv][on_id],
            dblock_len*sizeof(int), cudaMemcpyDeviceToHost, out_stream);
      }
    }
    gpu_check(cudaDeviceSynchronize());
  }
#endif // if 0

#if 0
  for(size_t ii = 0; ii < ablock_len; ++ii) {
    if(ablock[0][ii] != 3) {
      printf("0 %llu unsuccessful initialisation %d\n", ii, ablock[0][ii]);
    }
    else {
      printf("%d successful initialisation %d\n", ii, ablock[0][ii]);
    }
  }
#endif // if 0
