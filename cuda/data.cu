#include <stdlib.h>
#include "../cuda/config.h"
#include "../shared.h"
#include "../mesh.h"

// Allocates some double precision data
void allocate_data(double** buf, size_t len)
{
  cudaMalloc((void**)buf, sizeof(double)*len);

  // TODO: Add error checking
}

// Allocates a data array
void deallocate_data(double* buf)
{
  cudaFree(buf);

}

// Synchronise data
void sync_data(const int len, double* src, double* dst, int send)
{
  cudaMemcpy(src, dst, sizeof(double)*len, 
      (send ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));

  // TODO: Add error checking
}

__global__ void mesh_data_init_dx(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  edgedx[threadIdx.x+blockIdx.x*blockDim.x] = WIDTH / (global_nx);
  celldx[threadIdx.x+blockIdx.x*blockDim.x] = WIDTH / (global_nx);
}

__global__ void mesh_data_init_dy(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  edgedy[gid] = HEIGHT / (global_ny);
  celldy[gid] = HEIGHT / (global_ny);
}

__global__ void zero_cell_arrays(
    const int nx, const int ny, const int x_off, const int y_off, 
    double* rho, double* e, double* rho_old, double* P)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= nx*ny) return;

  rho[gid] = 0.0;
  e[gid] = 0.0;
  rho_old[gid] = 0.0;
  P[gid] = 0.0;
}

__global__ void zero_edge_arrays(
    const int nx, const int ny, double* Qxx, double* Qyy, double* x, 
    double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= (nx+1)*(ny+1)) return;

  Qxx[gid] = 0.0;
  Qyy[gid] = 0.0;
  x[gid] = 0.0;
  p[gid] = 0.0;
  rho_u[gid] = 0.0;
  rho_v[gid] = 0.0;
  F_x[gid] = 0.0;
  F_y[gid] = 0.0;
  uF_x[gid] = 0.0;
  uF_y[gid] = 0.0;
  vF_x[gid] = 0.0;
  vF_y[gid] = 0.0;
}

__global__ void initialise_default_state(
    const int nx, const int ny, 
    double* rho, double* e, double* rho_old, double* x)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= nx*ny) return;

  rho[gid] = 0.125;
  e[gid] = 2.0;
  x[gid] = rho[gid]*0.1;
}

__global__ void initialise_problem_state(
    const int nx, const int ny, const int global_nx, 
    const int global_ny, const int x_off, const int y_off, 
    double* rho, double* e, double* rho_old, double* x)
{
  set_cuda_indices(0);

  if(gid >= nx*ny) return;

  // CENTER SQUARE TEST
  if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
      jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
      ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
      ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
    rho[gid] = 1.0;
    e[gid] = 2.5;
    x[gid] = rho[ii*nx+jj]*0.1;
  }

#if 0
  // OFF CENTER SQUARE TEST
  const int dist = 100;
  if(jj+x_off-PAD >= global_nx/4-dist && 
      jj+x_off-PAD < global_nx/4+dist && 
      ii+y_off-PAD >= global_ny/2-dist && 
      ii+y_off-PAD < global_ny/2+dist) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
    x[ii*nx+jj] = rho[ii*nx+jj]*e[ii*nx+jj];
  }
#endif // if 0

#if 0
  if(jj+x_off < ((global_nx+2*PAD)/2)) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
    x[ii*nx+jj] = rho[ii*nx+jj]*0.1;
  }
#endif // if 0

#if 0
  if(ii+y_off < (global_ny+2*PAD)/2) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
  }
#endif // if 0

#if 0
  if(ii+y_off > (global_ny+2*PAD)/2) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
  }
#endif // if 0

#if 0
  if(jj+x_off > (global_nx+2*PAD)/2) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
  }
#endif // if 0
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

  nthreads_per_block = ceil((ny+1)/(double)NBLOCKS);
  mesh_data_init_dy<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);
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

  nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
  zero_edge_arrays<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, Qxx, Qyy, x, p, rho_u, rho_v, F_x, F_y,
      uF_x, uF_y, vF_x, vF_y);

  // WET STATE INITIALISATION
  // Initialise a default state for the energy and density on the mesh
  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  initialise_default_state<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, rho, e, rho_old, x);

  // Introduce a problem
  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  initialise_problem_state<<<nthreads_per_block, NBLOCKS>>>(
    nx, ny, global_nx, global_ny, x_off, y_off, rho, e, rho_old, x);
}

