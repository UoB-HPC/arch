#include <stdlib.h>
#include "../shared.h"
#include "../mesh.h"

// Allocates some double precision data
void allocate_data(double** buf, size_t len)
{
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double)*len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double)*len);
#endif
}

// Allocates a host copy of some buffer
void allocate_host_data(double** buf, size_t len)
{
  // Not necessary as host-only
}

// Allocates a data array
void deallocate_data(double* buf)
{
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Allocates a data array
void deallocate_host_data(double* buf)
{
  // Not necessary as host-only
}

// Synchronise data
void sync_data(const size_t len, double** src, double** dst, int send)
{
  // Don't need to move data with shared memory
  *dst = *src;
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_ny+1; ++ii) {
    edgedy[ii] = HEIGHT / (global_ny);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
    celldy[ii] = HEIGHT / (global_ny);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nx+1; ++ii) {
    edgedx[ii] = WIDTH / (global_nx);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = WIDTH / (global_nx);
  }
}

// Initialises mesh data in device specific manner
void mesh_data_init_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    double* edgedx, double* edgedy, double* edgedz, 
    double* celldx, double* celldy, double* celldz)
{
  // Initialise as in the 2d case
  mesh_data_init_2d(local_nx, local_ny, global_nx, global_ny, edgedx, edgedy, celldx, celldy);

  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_nz+1; ++ii) {
    edgedz[ii] = DEPTH / (global_nz);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nz; ++ii) {
    celldz[ii] = DEPTH / (global_nz);
  }
}

// Set the cell arrays to 0
void zero_cell_arrays(
    const int len, double* rho, double* e, double* rho_old, double* P)
{
  // Initialise all of the state to 0, but is this best for NUMA?
#pragma omp parallel for
  for(int ii = 0; ii < len; ++ii) {
    rho[ii] = 0.0;
    e[ii] = 0.0;
    rho_old[ii] = 0.0;
    P[ii] = 0.0;
  }
}

// Set the edge arrays to 0
void zero_edge_arrays_2d(
    const int len, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y, double* reduce_array)
{
#pragma omp parallel for
  for(int ii = 0; ii < len; ++ii) {
    Qxx[ii] = 0.0;
    Qyy[ii] = 0.0;
    x[ii] = 0.0;
    p[ii] = 0.0;
    rho_u[ii] = 0.0;
    rho_v[ii] = 0.0;
    F_x[ii] = 0.0;
    F_y[ii] = 0.0;
    uF_x[ii] = 0.0;
    uF_y[ii] = 0.0;
    vF_x[ii] = 0.0;
    vF_y[ii] = 0.0;
    reduce_array[ii] = 0.0;
  }
}

// Set the edge arrays to 0
void zero_edge_arrays_3d(
    const int local_nx, const int local_ny, const int local_nz, double* Qxx, 
    double* Qyy, double* Qzz, double* x, double* p, double* rho_u, double* rho_v, 
    double* rho_w, double* F_x, double* F_y, double* F_z, double* uF_x, 
    double* uF_y, double* uF_z, double* vF_x, double* vF_y, double* vF_z, 
    double* wF_x, double* wF_y, double* wF_z, double* reduce_array)
{
  zero_edge_arrays_2d( 
      (local_nx+1)*(local_ny+1)*(local_nz+1), Qxx, Qyy, x, p, rho_u, 
      rho_v, F_x, F_y, uF_x, uF_y, vF_x, vF_y, reduce_array);

#pragma omp parallel for
  for(int ii = 0; ii < (local_nx+1)*(local_ny+1)*(local_nz+1); ++ii) {
    Qzz[ii] = 0.0;
    rho_w[ii] = 0.0;
    F_z[ii] = 0.0;
    uF_z[ii] = 0.0; 
    vF_z[ii] = 0.0;
    wF_x[ii] = 0.0;
    wF_y[ii] = 0.0;
    wF_z[ii] = 0.0;
  }
}

void set_default_state(
    const int len, double* rho, double* e, double* x)
{
  // Initialise a default state for the energy and density on the mesh
#pragma omp parallel for
  for(int ii = 0; ii < len; ++ii) {
    rho[ii] = 0.125;
    e[ii] = 2.0;
    x[ii] = rho[ii]*0.1;
  }
}

// Initialise state data in device specific manner
void state_data_init_2d(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off, double* rho, double* e, double* rho_old, 
    double* P, double* Qxx, double* Qyy, double* x, double* p, double* rho_u, 
    double* rho_v, double* F_x, double* F_y, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* reduce_array, double* celldx, double* celldy)
{
  zero_cell_arrays(
      local_nx*local_ny, rho, e, rho_old, P);
  zero_edge_arrays_2d( 
      (local_nx+1)*(local_ny+1), Qxx, Qyy, x, p, rho_u, rho_v, F_x, F_y,
      uF_x, uF_y, vF_x, vF_y, reduce_array);
  set_default_state(
      local_nx*local_ny, rho, e, x);

  // Introduce a problem
#pragma omp parallel for 
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {

#if 0
      // POINT CHARGE PROBLEM
      if(jj+x_off == global_nx/2 && ii+y_off == global_ny/2)
        e[(ii*local_nx)+(jj)] = 10.0/(WIDTH/global_nx*HEIGHT/global_ny);
      else 
        e[(ii*local_nx)+(jj)] = 0.0;
#endif // if 0

      // CENTER SQUARE TEST
      if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
          jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
          ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
          ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
        x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
      }
    }
  }
}

// Initialise state data in device specific manner
void state_data_init_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    const int x_off, const int y_off, const int z_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy, 
    double* Qzz, double* x, double* p, double* rho_u, double* rho_v, double* rho_w, 
    double* F_x, double* F_y, double* F_z, double* uF_x, double* uF_y, double* uF_z, 
    double* vF_x, double* vF_y, double* vF_z, double* wF_x, double* wF_y, 
    double* wF_z, double* reduce_array)
{
  zero_cell_arrays(
      local_nx*local_ny*local_nz, rho, e, rho_old, P);
  zero_edge_arrays_3d( 
      local_nx, local_ny, local_nz, Qxx, Qyy, Qzz, x, p, rho_u, rho_v, rho_w, 
      F_x, F_y, F_z, uF_x, uF_y, uF_z, vF_x, vF_y, vF_z, wF_x, wF_y, wF_z, reduce_array);
  set_default_state(
      local_nx*local_ny*local_nz, rho, e, x);

  // Introduce a problem
#pragma omp parallel for 
  for(int ii = 0; ii < local_nz; ++ii) {
    for(int jj = 0; jj < local_ny; ++jj) {
      for(int kk = 0; kk < local_nx; ++kk) {
        const int ind = ii*local_nx*local_ny+jj*local_nx+kk;

        // CENTER TUBE TEST
        if(kk+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
            kk+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
            jj+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
            jj+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
          rho[ind] = 1.0;
          e[ind] = 2.5;
          x[ind] = rho[ind]*0.1;
        }
      }
    }
  }
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
