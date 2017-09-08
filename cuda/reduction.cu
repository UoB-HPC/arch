#include "../shared.h"
#include "reduction.k"
#include "shared.h"

void finish_min_reduce(int nblocks1, double* reduce_array, double* result) {
  while (nblocks1 > 1) {
    int nblocks0 = nblocks1;
    nblocks1 = max(1, (int)ceil(nblocks1 / (double)NTHREADS));
    min_reduce<double, NTHREADS><<<nblocks1, NTHREADS>>>(
        reduce_array, reduce_array, nblocks0);
  }
  gpu_check(cudaDeviceSynchronize());

  copy_buffer(1, &reduce_array, &result, RECV);
}

void finish_sum_reduce(int nblocks1, double* reduce_array, double* result) {
  while (nblocks1 > 1) {
    int nblocks0 = nblocks1;
    nblocks1 = max(1, (int)ceil(nblocks1 / (double)NTHREADS));
    sum_reduce<double, NTHREADS><<<nblocks1, NTHREADS>>>(
        reduce_array, reduce_array, nblocks0);
  }
  gpu_check(cudaDeviceSynchronize());

  copy_buffer(1, &reduce_array, &result, RECV);
}

void finish_sum_int_reduce(int nblocks1, int* reduce_array, int* result) {
  while (nblocks1 > 1) {
    int nblocks0 = nblocks1;
    nblocks1 = max(1, (int)ceil(nblocks1 / (double)NTHREADS));
    sum_reduce<int, NTHREADS><<<nblocks1, NTHREADS>>>(reduce_array,
                                                      reduce_array, nblocks0);
  }
  gpu_check(cudaDeviceSynchronize());

  copy_int_buffer(1, &reduce_array, &result, RECV);
}

void finish_sum_uint64_reduce(int nblocks1, uint64_t* reduce_array, uint64_t* result) {
  while (nblocks1 > 1) {
    int nblocks0 = nblocks1;
    nblocks1 = max(1, (int)ceil(nblocks1 / (double)NTHREADS));
    sum_reduce<uint64_t, NTHREADS><<<nblocks1, NTHREADS>>>(reduce_array,
                                                      reduce_array, nblocks0);
  }
  gpu_check(cudaDeviceSynchronize());

  copy_uint64_buffer(1, &reduce_array, &result, RECV);
}
