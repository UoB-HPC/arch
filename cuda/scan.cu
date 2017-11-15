#include "../shared.h"
#include "shared.h"
#include <math.h>

// Perform the up-sweep and down-sweep of the exclusive scan.
__global__ void exclusive_scan(const int n, double* buffer,
                               double* global_scan_sums) {

  const int tid = threadIdx.x;
  const int gid0 = blockIdx.x * blockDim.x + 2 * threadIdx.x;

  // Fill the shared buffer
  __shared__ double local_scan_array[NTHREADS_SCAN];
  local_scan_array[(2 * tid)] = buffer[(gid0)];
  local_scan_array[(2 * tid + 1)] = buffer[(gid0 + 1)];

  int offset = 1;
  for (int d = NTHREADS_SCAN / 2; d > 0; d /= 2) {
    __syncthreads();

    const int ai = offset * (2 * tid + 1) - 1;
    const int bi = offset * (2 * tid + 2) - 1;

    if (tid < d) {
      local_scan_array[(bi)] += local_scan_array[(ai)];
    }

    offset *= 2;
  }

  if (tid == NTHREADS_SCAN - 1) {
    local_scan_array[(tid)] = 0;
  }

  for (int d = 1; d < NTHREADS_SCAN; d *= 2) {
    offset >>= 1;

    __syncthreads();

    const int ai = offset * (2 * tid + 1) - 1;
    const int bi = offset * (2 * tid + 2) - 1;

    if (tid < d) {
      double temp = local_scan_array[(ai)];
      local_scan_array[(ai)] = local_scan_array[(bi)];
      local_scan_array[(bi)] = t;
    }
  }

  __syncthreads();
  buffer[(gid0)] = local_scan_array[(2 * tid)];
  buffer[(gid0 + 1)] = local_scan_array[(2 * tid + 1)];
}

// Perform the final step of the exclusive scan where the block local exclusive
// buckets are incrememnted by global scan sums.
__global__ void finalise_exclusive_scan(double* buffer,
                                        double* global_scan_sums) {

  const int bid = blockIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadsIdx.x;

  // TODO: do we need to handle the padding of the block size here?
  buffer[(gid)] = global_scan_sums[(bid)];
}

// Performs an exclusive scan for the provided buffer
void perform_exclusive_scan(const int n, double* buffer,
                            double* global_scan_sums) {

  const int ln = log2(n) - 1;

  const int nblocks = ceil(n / (double)NTHREADS_SCAN);

  exclusive_scan<<<nblocks, NTHREADS_SCAN>>>(buffer);

#if 0
  if (nblocks > 1) {
    perform_exclusive_scan<<<nblocks, NTHREADS_SCAN>>>(global_scan_sums);
    finalise_exclusive_scan<<<nblocks, NTHREADS_SCAN>>>(buffer,
                                                        global_scan_sums);
  }
#endif // if 0
}
