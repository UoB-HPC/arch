#include <stdlib.h>
#include "../shared.h"
#include "../mesh.h"
#include "../params.h"

// Allocates some double precision data
void allocate_data(double** buf, size_t len)
{
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double)*len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double)*len);
#endif

  if(*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

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
    const double width, const double height,
    double* edgex, double* edgey, 
    double* edgedx, double* edgedy, 
    double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_nx+1; ++ii) {
    edgedx[ii] = width / (global_nx);

    // Note: correcting for padding
    edgex[ii] = edgedx[ii]*(x_off+ii-PAD);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = width / (global_nx);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_ny+1; ++ii) {
    edgedy[ii] = height / (global_ny);

    // Note: correcting for padding
    edgey[ii] = edgedy[ii]*(y_off+ii-PAD);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
    celldy[ii] = height / (global_ny);
  }
}

// Initialises mesh data in device specific manner
void mesh_data_init_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    const int x_off, const int y_off, const int z_off,
    const double width, const double height, const double depth,
    double* edgex, double* edgey, double* edgez, 
    double* edgedx, double* edgedy, double* edgedz, 
    double* celldx, double* celldy, double* celldz)
{
  // Initialise as in the 2d case
  mesh_data_init_2d(
      local_nx, local_ny, global_nx, global_ny, x_off, y_off, width, height,
      edgex, edgey, edgedx, edgedy, celldx, celldy);

  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_nz+1; ++ii) {
    edgedz[ii] = depth / (global_nz);
    edgez[ii] = edgedz[ii]*(z_off+ii-PAD);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nz; ++ii) {
    celldz[ii] = depth / (global_nz);
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
    const int local_nx, const int local_ny, const int x_off, const int y_off,
    const double* edgex, const double* edgey, const int ndims,
    const char* problem_def_filename, double* rho, double* e, double* x)
{
  set_default_state(
      local_nx*local_ny, rho, e, x);

  printf("Loading problem from %s.\n", problem_def_filename);

  char* keys_space = (char*)malloc(sizeof(char)*MAX_KEYS*(MAX_STR_LEN+1));
  char** keys = (char**)malloc(sizeof(char*)*MAX_KEYS);
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);
  for(int ii = 0; ii < MAX_KEYS; ++ii) {
    keys[ii] = &keys_space[ii*(MAX_STR_LEN+1)];
  }

  int nkeys = 0;
  int nproblem_entries = 0;
  while(get_problem_parameter(
        nproblem_entries++, problem_def_filename, keys, values, &nkeys)) {

    // The last four keys are the bound specification
    double xpos = values[nkeys-4];
    double ypos = values[nkeys-3];
    double width = values[nkeys-2];
    double height = values[nkeys-1];

    // Loop through the mesh and set the problem
    for(int ii = 0; ii < local_ny; ++ii) {
      for(int jj = 0; jj < local_nx; ++jj) {
        double global_xpos = edgex[jj+x_off];
        double global_ypos = edgey[ii+y_off];

        // Check we are in bounds of the problem entry
        if(global_xpos >= xpos && global_ypos >= ypos && 
           global_xpos < xpos+width && global_ypos < ypos+height)
        {
          // The upper bound excludes the bounding box for the entry
          for(int kk = 0; kk < nkeys-(2*ndims); ++kk) {
            if(strmatch(keys[kk], "density")) {
              rho[ii*local_nx+jj] = values[kk];
            }
            else if(strmatch(keys[kk], "energy")) {
              e[ii*local_nx+jj] = values[ii];
            }
            else if(strmatch(keys[kk], "temperature")) {
              x[ii*local_nx+jj] = values[kk];
            }
            else {
              TERMINATE("Found unrecognised key in %s : %s.\n", 
                  problem_def_filename, keys[kk]);
            }
          }
        }
      }
    }

    nkeys = 0;
  }

  free(keys_space);
  free(keys);
  free(values);
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
      }
    }
  }
}


