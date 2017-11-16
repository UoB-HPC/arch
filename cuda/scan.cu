#include "../shared.h"
#include "shared.h"
#include <math.h>

// Perform the up-sweep and down-sweep of the exclusive scan.
__global__ void exclusive_scan(const int n, int* buffer,
                               int* global_scan_sums) {

  const int local_n = 2*NTHREADS_SCAN;
  const int tid = threadIdx.x;
  const int block_off = blockIdx.x * blockDim.x;

  // Fill the shared buffer
  __shared__ int local_scan_array[NTHREADS_SCAN*2];
  local_scan_array[(2 * tid)] = buffer[(block_off+2*tid)];
  local_scan_array[(2 * tid + 1)] = buffer[(block_off+2*tid + 1)];

  int off = 1;
  for (int d = local_n / 2; d > 0; d /= 2) {
    __syncthreads();

    if (tid < d) {
      const int ai = off * (2 * tid + 1) - 1;
      const int bi = off * (2 * tid + 2) - 1;
      local_scan_array[(bi)] += local_scan_array[(ai)];
    }

    off *= 2;
  }

  if (tid == 0) {
    local_scan_array[(local_n-1)] = 0;
  }

  for (int d = 1; d < local_n; d *= 2) {
    off >>= 1;

    __syncthreads();

    if (tid < d) {
      const int ai = off * (2 * tid + 1) - 1;
      const int bi = off * (2 * tid + 2) - 1;
      const int temp = local_scan_array[(ai)];
      local_scan_array[(ai)] = local_scan_array[(bi)];
      local_scan_array[(bi)] += temp;
    }
  }

  __syncthreads();
  buffer[(block_off+2*tid)] = local_scan_array[(2 * tid)];
  buffer[(block_off+2*tid + 1)] = local_scan_array[(2 * tid + 1)];
}

// Perform the final step of the exclusive scan where the block local exclusive
// buckets are incrememnted by global scan sums.
__global__ void finalise_exclusive_scan(int* buffer,
    int* global_scan_sums) {

  const int bid = blockIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: do we need to handle the padding of the block size here?
  buffer[(gid)] = global_scan_sums[(bid)];
}

// Performs an exclusive scan for the provided buffer
void perform_exclusive_scan(const int n, int* buffer,
    int* global_scan_sums) {

  const int ln = (int)log2((double)n) - 1;
  const int nblocks = ceil(n / (double)NTHREADS_SCAN);
  exclusive_scan<<<nblocks, NTHREADS_SCAN>>>(n, buffer, global_scan_sums);
  gpu_check(cudaDeviceSynchronize());

#if 0
  if (nblocks > 1) {
    perform_exclusive_scan<<<nblocks, NTHREADS_SCAN>>>(global_scan_sums);
    finalise_exclusive_scan<<<nblocks, NTHREADS_SCAN>>>(buffer,
        global_scan_sums);
  }
#endif // if 0
}
