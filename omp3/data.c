#include <stdlib.h>
#include "../shared.h"
#include "../mesh.h"
#include "../params.h"

// Allocates some double precision data
size_t allocate_data(double** buf, size_t len)
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

  return sizeof(double)*len;
}

// Allocates some int precision data
size_t allocate_int_data(int** buf, size_t len)
{
#ifdef INTEL
  *buf = (int*)_mm_malloc(sizeof(int)*len, VEC_ALIGN);
#else
  *buf = (int*)malloc(sizeof(int)*len);
#endif

  if(*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

  // Perform first-touch
#pragma omp parallel for
  for(size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0;
  }

  return sizeof(int)*len;
}

// Allocates a host copy of some buffer
void allocate_host_data(double** buf, size_t len)
{
  allocate_data(buf, len);
}

// Allocates a host copy of some buffer
void allocate_host_int_data(int** buf, size_t len)
{
  allocate_int_data(buf, len);
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

// Allocates a data array
void deallocate_int_data(int* buf)
{
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Allocates a data array
void deallocate_host_int_data(int* buf)
{
  // Not necessary as host-only
}


// Just swaps the buffers on the host
void copy_buffer(const size_t len, double** src, double** dst, int send)
{
  double* temp = *src;
  *src = *dst;
  *dst = temp;
}

// Just swaps the buffers on the host
void copy_int_buffer(const size_t len, int** src, int** dst, int send)
{
  int* temp = *src;
  *src = *dst;
  *dst = temp;
}

// Move a host buffer onto the device
void move_host_buffer_to_device(const size_t len, double** src, double** dst)
{
  copy_buffer(len, src, dst, SEND);
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(
    const int local_nx, const int local_ny, 
    const int global_nx, const int global_ny,
    const int pad, const int x_off, const int y_off,
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
    edgex[ii] = edgedx[ii]*(x_off+ii-pad);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = width / (global_nx);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_ny+1; ++ii) {
    edgedy[ii] = height / (global_ny);

    // Note: correcting for padding
    edgey[ii] = edgedy[ii]*(y_off+ii-pad);
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
    const int pad, const int x_off, const int y_off, const int z_off,
    const double width, const double height, const double depth,
    double* edgex, double* edgey, double* edgez, 
    double* edgedx, double* edgedy, double* edgedz, 
    double* celldx, double* celldy, double* celldz)
{
  // Initialise as in the 2d case
  mesh_data_init_2d(
      local_nx, local_ny, global_nx, global_ny, pad, x_off, y_off, width, height,
      edgex, edgey, edgedx, edgedy, celldx, celldy);

  // Simple uniform rectilinear initialisation
#pragma omp parallel for
  for(int ii = 0; ii < local_nz+1; ++ii) {
    edgedz[ii] = depth / (global_nz);
    edgez[ii] = edgedz[ii]*(z_off+ii-pad);
  }
#pragma omp parallel for
  for(int ii = 0; ii < local_nz; ++ii) {
    celldz[ii] = depth / (global_nz);
  }
}

// Initialise state data in device specific manner
void set_problem_2d(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const int pad, const int x_off, const int y_off, 
    const double mesh_width, const double mesh_height, const double* edgex, 
    const double* edgey, const int ndims, const char* problem_def_filename, 
    double* rho, double* e, double* x)
{
  char* keys = (char*)malloc(sizeof(char)*MAX_KEYS*MAX_STR_LEN);
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);

  int nentries = 0;
  while(1) {
    char specifier[MAX_STR_LEN];
    sprintf(specifier, "problem_%d", nentries++);

    int nkeys = 0;
    if(!get_key_value_parameter(
          specifier, problem_def_filename, keys, values, &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = values[nkeys-4]*mesh_width;
    double ypos = values[nkeys-3]*mesh_height;
    double width = values[nkeys-2]*mesh_width;
    double height = values[nkeys-1]*mesh_height;

    // Loop through the mesh and set the problem
    for(int ii = pad; ii < local_ny-pad; ++ii) {
      for(int jj = pad; jj < local_nx-pad; ++jj) {
        double global_xpos = edgex[jj];
        double global_ypos = edgey[ii];

        // Check we are in bounds of the problem entry
        if(global_xpos >= xpos && global_ypos >= ypos && 
            global_xpos < xpos+width && global_ypos < ypos+height)
        {
          // The upper bound excludes the bounding box for the entry
          for(int kk = 0; kk < nkeys-(2*ndims); ++kk) {
            const char* key = &keys[kk*MAX_STR_LEN];
            if(strmatch(key, "density")) {
              rho[ii*local_nx+jj] = values[kk];
            }
            else if(strmatch(key, "energy")) {
              e[ii*local_nx+jj] = values[kk];
            }
            else if(strmatch(key, "temperature")) {
              x[ii*local_nx+jj] = values[kk];
            }
            else {
              TERMINATE("Found unrecognised key in %s : %s.\n", 
                  problem_def_filename, key);
            }
          }
        }
      }
    }
  }

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
  TERMINATE("set_problem_3d not implemented yet.");
}


