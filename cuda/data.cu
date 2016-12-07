#include <stdlib.h>
#include "../cuda/config.h"
#include "../shared.h"
#include "../mesh.h"
#include "data.k"

// Allocates some double precision data
void allocate_data(double** buf, const size_t len)
{
  gpu_check(cudaMalloc((void**)buf, sizeof(double)*len));
}

// Allocates a data array
void deallocate_data(double* buf)
{
  gpu_check(cudaFree(buf));
}

// Synchronise data
void sync_data(const size_t len, double** src, double** dst, int send)
{
  if(send) {
    gpu_check(cudaMemcpy(*dst, *src, sizeof(double)*len, cudaMemcpyHostToDevice));
  }
  else {
    gpu_check(cudaMemcpy(*dst, *src, sizeof(double)*len, cudaMemcpyDeviceToHost));
  }
}

// Initialises mesh data in device specific manner
void mesh_data_init(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
  int nthreads_per_block = ceil((nx+1)/(double)NBLOCKS);
  mesh_data_init_dx<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());

  nthreads_per_block = ceil((ny+1)/(double)NBLOCKS);
  mesh_data_init_dy<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());
}

// Initialise state data in device specific manner
void state_data_init(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y)
{
  // TODO: Improve what follows, make it a somewhat more general problem 
  // selection mechanism for some important stock problems

  int nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  zero_cell_arrays<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, x_off, y_off, rho, e, rho_old, P);
  gpu_check(cudaDeviceSynchronize());

  nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
  zero_edge_arrays<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, Qxx, Qyy, x, p, rho_u, rho_v, F_x, F_y,
      uF_x, uF_y, vF_x, vF_y);
  gpu_check(cudaDeviceSynchronize());

  // WET STATE INITIALISATION
  // Initialise a default state for the energy and density on the mesh
  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  initialise_default_state<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, rho, e, rho_old, x);
  gpu_check(cudaDeviceSynchronize());

  // Introduce a problem
  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  initialise_problem_state<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, global_nx, global_ny, x_off, y_off, rho, e, rho_old, x);
  gpu_check(cudaDeviceSynchronize());
}

