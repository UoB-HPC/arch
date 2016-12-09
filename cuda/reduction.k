#include <float.h>

  template <unsigned int nthreads>
__device__ void min_reduce_in_shared(
    const int tid, double* sdata)
{
  if (nthreads >= 512) { 
    if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } 
    __syncthreads(); 
  }
  if (nthreads >= 256) { 
    if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } 
    __syncthreads(); 
  }
  if (nthreads >= 128) { 
    if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } 
    __syncthreads(); 
  }
  if (tid < 32) {
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
  }
}

  template <unsigned int nthreads>
__device__ void sum_reduce_in_shared(
    const int tid, double* sdata)
{
  if (nthreads >= 512) { 
    if (tid < 256) { sdata[tid] += sdata[tid + 256]; } 
    __syncthreads(); 
  }
  if (nthreads >= 256) { 
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } 
    __syncthreads(); 
  }
  if (nthreads >= 128) { 
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } 
    __syncthreads(); 
  }
  if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    __syncthreads();
    sdata[tid] += sdata[tid + 16];
    __syncthreads();
    sdata[tid] += sdata[tid + 8];
    __syncthreads();
    sdata[tid] += sdata[tid + 4];
    __syncthreads();
    sdata[tid] += sdata[tid + 2];
    __syncthreads();
    sdata[tid] += sdata[tid + 1];
    __syncthreads();
  }
}

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
  template <unsigned int nthreads>
__global__ void min_reduce(
    double* data, double* result, int nblocks)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double sdata[nthreads];
  sdata[threadIdx.x] = (gid < nblocks) ? data[gid] : DBL_MAX;
  __syncthreads();

  min_reduce_in_shared<nthreads>(threadIdx.x, sdata);
  if (threadIdx.x == 0) 
    result[blockIdx.x] = sdata[0];
}

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
  template <unsigned int nthreads>
__global__ void sum_reduce(
    double* data, double* result, int nblocks)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double sdata[nthreads];
  sdata[threadIdx.x] = (gid < nblocks) ? data[gid] : 0.0;
  __syncthreads();

  sum_reduce_in_shared<nthreads>(threadIdx.x, sdata);
  if (threadIdx.x == 0) 
    result[blockIdx.x] = sdata[0];
}

