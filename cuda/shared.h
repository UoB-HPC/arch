#pragma once

#include <stdio.h>

#define NTHREADS 128

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)


// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpu_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPU Error Returned: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

void finish_min_reduce(
    int nblocks1, double* reduce_array, double* result);
void finish_sum_reduce(
    int nblocks1, double* reduce_array, double* result);

