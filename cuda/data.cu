#include <stdlib.h>
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
  cudaMemcpy(src, dst, nx*ny, (send ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));

  // TODO: Add error checking
}

// Initialises mesh data in device specific manner
void mesh_data_init(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
  int nthreads_per_block = ceil((local_nx+1)/(double)NBLOCKS);
  mesh_data_init_dx<<<nthreads_per_block, NBLOCKS>>>(
      local_nx, local_ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);

  nthreads_per_block = ceil((local_ny+1)/(double)NBLOCKS);
  mesh_data_init_dy<<<nthreads_per_block, NBLOCKS>>>(
      local_nx, local_ny, global_nx, global_ny,
      edgedx, edgedy, celldx, celldy);
}

__global__ void mesh_data_init_dx(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  edgedx[threadIdx.x+blockIdx.x*blockDim.x] = WIDTH / (global_nx);
  celldx[threadIdx.x+blockIdx.x*blockDim.x] = WIDTH / (global_nx);
}

__global__ void mesh_data_init_dy(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  edgedy[gid] = HEIGHT / (global_ny);
  celldy[gid] = HEIGHT / (global_ny);
}

__global__ void zero_cell_arrays(
    const int local_nx, const int local_ny, const int x_off, const int y_off, 
    double* rho, double* e, double* rho_old, double* P)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= local_nx*local_ny) return;

  rho[gid] = 0.0;
  e[gid] = 0.0;
  rho_old[gid] = 0.0;
  P[gid] = 0.0;
}

__global__ void zero_edge_arrays(
    const int local_nx, const int local_ny, double* Qxx, double* Qyy, double* x, 
    double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= (local_nx+1)*(local_ny+1)) return;

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
    const int local_nx, const int local_ny, 
    double* rho, double* e, double* rho_old, double* x)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= local_nx*local_ny) return;

  rho[gid] = 0.125;
  e[gid] = 2.0;
  x[gid] = rho[gid]*0.1;
}

__global__ void initialise_problem_state(
    const int local_nx, const int local_ny, const int global_nx, 
    const int global_ny, double* rho, double* e, double* rho_old, double* x)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= local_nx*local_ny) return;

  // CENTER SQUARE TEST
  if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
      jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
      ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
      ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
    rho[ii*local_nx+jj] = 1.0;
    e[ii*local_nx+jj] = 2.5;
    x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
  }

#if 0
  // OFF CENTER SQUARE TEST
  const int dist = 100;
  if(jj+x_off-PAD >= global_nx/4-dist && 
      jj+x_off-PAD < global_nx/4+dist && 
      ii+y_off-PAD >= global_ny/2-dist && 
      ii+y_off-PAD < global_ny/2+dist) {
    rho[ii*local_nx+jj] = 1.0;
    e[ii*local_nx+jj] = 2.5;
    x[ii*local_nx+jj] = rho[ii*local_nx+jj]*e[ii*local_nx+jj];
  }
#endif // if 0

#if 0
  if(jj+x_off < ((global_nx+2*PAD)/2)) {
    rho[ii*local_nx+jj] = 1.0;
    e[ii*local_nx+jj] = 2.5;
    x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
  }
#endif // if 0

#if 0
  if(ii+y_off < (global_ny+2*PAD)/2) {
    rho[ii*local_nx+jj] = 1.0;
    e[ii*local_nx+jj] = 2.5;
  }
#endif // if 0

#if 0
  if(ii+y_off > (global_ny+2*PAD)/2) {
    rho[ii*local_nx+jj] = 1.0;
    e[ii*local_nx+jj] = 2.5;
  }
#endif // if 0

#if 0
  if(jj+x_off > (global_nx+2*PAD)/2) {
    rho[ii*local_nx+jj] = 1.0;
    e[ii*local_nx+jj] = 2.5;
  }
#endif // if 0
}

// Initialise state data in device specific manner
void state_data_init(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y)
{
  // TODO: Improve what follows, make it a somewhat more general problem 
  // selection mechanism for some important stock problems

  int nthreads_per_block = ceil(local_nx*local_ny/(double)NBLOCKS);
  zero_cell_arrays<<<nthreads_per_block, NBLOCKS>>>(
      local_nx, local_ny, x_off, y_off, rho, e, rho_old, P);

  nthreads_per_block = ceil((local_nx+1)*(local_ny+1)/(double)NBLOCKS);
  zero_edge_arrays<<<nthreads_per_block, NBLOCKS>>>(
      local_nx, local_ny, Qxx, Qyy, x, p, rho_u, rho_v, F_x, F_y,
      uF_x, uF_y, vF_x, vF_y);

  // WET STATE INITIALISATION
  // Initialise a default state for the energy and density on the mesh
  nthreads_per_block = ceil(local_nx*local_ny/(double)NBLOCKS);
  initialise_default_state<<<nthreads_per_block, NBLOCKS>>>(
      local_nx, local_ny, rho, e, rho_old, x);

  // Introduce a problem
  nthreads_per_block = ceil(local_nx*local_ny/(double)NBLOCKS);
  initialise_problem_state<<<nthreads_per_block, NBLOCKS>>>(
      local_nx, local_ny, rho, e, rho_old, x);
}

