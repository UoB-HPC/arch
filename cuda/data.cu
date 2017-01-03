#include <stdlib.h>
#include "../cuda/shared.h"
#include "../shared.h"
#include "../mesh.h"
#include "../state.h"
#include "data.k"

// Allocates some double precision data
void allocate_data(double** buf, const size_t len)
{
  gpu_check(
      cudaMalloc((void**)buf, sizeof(double)*len));

  const int nblocks = ceil(len/(double)NTHREADS);
  zero_array<<<nblocks, NTHREADS>>>(len, *buf);
  gpu_check(cudaDeviceSynchronize());
}

// Allocates some double precision data
void allocate_host_data(double** buf, const size_t len)
{
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double)*len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double)*len);
#endif

#pragma omp parallel for
  for(size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }
}

// Allocates a data array
void deallocate_data(double* buf)
{
  gpu_check(
      cudaFree(buf));
}

// Allocates a data array
void deallocate_host_data(double* buf)
{
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Synchronise data
void sync_data(const size_t len, double** src, double** dst, int send)
{
  if(send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(double)*len, cudaMemcpyHostToDevice));
  }
  else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(double)*len, cudaMemcpyDeviceToHost));
  }
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
  int nblocks = ceil((nx+1)/(double)NTHREADS);
  mesh_data_init_dx<<<nblocks, NTHREADS>>>(
      nx, ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil((ny+1)/(double)NTHREADS);
  mesh_data_init_dy<<<nblocks, NTHREADS>>>(
      nx, ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());
}

// Initialise state data in device specific manner
void state_data_init_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y, double* reduce_array)
{
  // TODO: Improve what follows, make it a somewhat more general problem 
  // selection mechanism for some important stock problems

  // WET STATE INITIALISATION
  // Initialise a default state for the energy and density on the mesh
  nblocks = ceil(nx*ny/(double)NTHREADS);
  initialise_default_state<<<nblocks, NTHREADS>>>(
      nx, ny, rho, e, rho_old, x);
  gpu_check(cudaDeviceSynchronize());

  // Introduce a problem
  nblocks = ceil(nx*ny/(double)NTHREADS);
  initialise_problem_state<<<nblocks, NTHREADS>>>(
      nx, ny, global_nx, global_ny, x_off, y_off, rho, e, rho_old, x);
  gpu_check(cudaDeviceSynchronize());
}

