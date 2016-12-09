#pragma once

#include <stdio.h>

#define NTHREADS 128

#define set_cuda_indices(padx) \
  const int gid = threadIdx.x+blockIdx.x*blockDim.x; \
const int jj = (gid % (nx+padx));\
const int ii = (gid / (nx+padx));

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

extern "C"
void state_data_init(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y, double* reduce_array);

void finish_min_reduce(
    int nblocks1, double* reduce_array, double* result);
void finish_sum_reduce(
    int nblocks1, double* reduce_array, double* result);

