#pragma once

#include "cuda_fp16.h"
#include <stdint.h>
#include <stdio.h>

#define NTHREADS 128
#define NTHREADS_SCAN 1024
#define NTHREADS_REDUCE 1024

// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpu_check(ans)                                                         \
  { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char* file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Error Returned: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

void finish_min_reduce(int nblocks1, double* reduce_array, double* result);
void finish_sum_reduce(int nblocks1, double* reduce_array, double* result);
void finish_sum_int_reduce(int nblocks1, int* reduce_array, int* result);
void finish_sum_uint64_reduce(int nblocks1, uint64_t* reduce_array,
                              uint64_t* result);

// TODO: At some point the half precision type could be refactored and this can
// be added back to the global shared.h
size_t allocate_half_data(__half** buf, const size_t len);
