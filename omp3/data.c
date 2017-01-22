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

  // Perform first-touch
#pragma omp parallel for
  for(size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }
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
    const int local_nx, const int local_ny, 
    const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* edgex, double* edgey, 
    double* edgedx, double* edgedy, 
    double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_nx+1; ++ii) {
    edgedx[ii] = WIDTH / (global_nx);

    // Note: correcting for padding
    edgex[ii] = edgedx[ii]*(x_off+ii-PAD);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = WIDTH / (global_nx);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_ny+1; ++ii) {
    edgedy[ii] = HEIGHT / (global_ny);

    // Note: correcting for padding
    edgey[ii] = edgedy[ii]*(y_off+ii-PAD);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
    celldy[ii] = HEIGHT / (global_ny);
  }
}

// Initialises mesh data in device specific manner
void mesh_data_init_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    const int x_off, const int y_off, const int z_off,
    double* edgex, double* edgey, double* edgez, 
    double* edgedx, double* edgedy, double* edgedz, 
    double* celldx, double* celldy, double* celldz)
{
  // Initialise as in the 2d case
  mesh_data_init_2d(
      local_nx, local_ny, global_nx, global_ny, x_off, y_off,
      edgex, edgey, edgedx, edgedy, celldx, celldy);

  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_nz+1; ++ii) {
    edgedz[ii] = DEPTH / (global_nz);
    edgez[ii] = edgedz[ii]*(z_off+ii-PAD);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nz; ++ii) {
    celldz[ii] = DEPTH / (global_nz);
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
void set_problem_2d(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off, double* rho, double* e, double* x)
{
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
        rho[ii*local_nx+jj] = 100.0;
        e[ii*local_nx+jj] = 2.5;
        x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
      }
    }
  }
}

// Initialise state data in device specific manner
void set_problem_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    const int x_off, const int y_off, const int z_off,
    double* rho, double* e, double* x)
{
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
