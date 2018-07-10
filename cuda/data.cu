#include "../cuda/shared.h"
#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include "../shared_data.h"
#include "../umesh.h"
#include "data.k"
#include <stdlib.h>

// Allocates some double precision data
size_t allocate_data(double** buf, const size_t len) {
  gpu_check(cudaMalloc((void**)buf, sizeof(double) * len));

  const int nblocks = ceil(len / (double)NTHREADS);
  zero_array<double><<<nblocks, NTHREADS>>>(len, *buf);
  gpu_check(cudaDeviceSynchronize());
  return sizeof(double) * len;
}

// Allocates some single precision data
size_t allocate_float_data(float** buf, const size_t len) {
  gpu_check(cudaMalloc((void**)buf, sizeof(double) * len));

  const int nblocks = ceil(len / (double)NTHREADS);
  zero_array<float><<<nblocks, NTHREADS>>>(len, *buf);
  gpu_check(cudaDeviceSynchronize());
  return sizeof(float) * len;
}

// Allocates some integer data
size_t allocate_int_data(int** buf, const size_t len) {
  gpu_check(cudaMalloc((void**)buf, sizeof(int) * len));

  const int nblocks = ceil(len / (double)NTHREADS);
  zero_array<int><<<nblocks, NTHREADS>>>(len, *buf);
  gpu_check(cudaDeviceSynchronize());
  return sizeof(int) * len;
}

// Allocates half precision floating point data
size_t allocate_half_data(__half** buf, const size_t len) {
  gpu_check(cudaMalloc((void**)buf, sizeof(__half) * len));
  const int nblocks = ceil(len / (double)NTHREADS);
  gpu_check(cudaDeviceSynchronize());
  return sizeof(__half) * len;
}

// Allocates some 64bit integer data
size_t allocate_uint64_data(uint64_t** buf, const size_t len) {
  gpu_check(cudaMalloc((void**)buf, sizeof(uint64_t) * len));

  const int nblocks = ceil(len / (double)NTHREADS);
  zero_array<uint64_t><<<nblocks, NTHREADS>>>(len, *buf);
  gpu_check(cudaDeviceSynchronize());
  return len;
}

// Allocates some double precision data
void allocate_host_data(double** buf, const size_t len) {
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double) * len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double) * len);
#endif
  if (!*buf) {
    TERMINATE("Could not allocate host int data.\n");
  }

#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }
}

// Allocates some single precision data
void allocate_host_float_data(float** buf, const size_t len) {
#ifdef INTEL
  *buf = (float*)_mm_malloc(sizeof(float) * len, VEC_ALIGN);
#else
  *buf = (float*)malloc(sizeof(float) * len);
#endif
  if (!*buf) {
    TERMINATE("Could not allocate host int data.\n");
  }

#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }
}

// Allocates some double precision data
void allocate_host_int_data(int** buf, const size_t len) {
#ifdef INTEL
  *buf = (int*)_mm_malloc(sizeof(int) * len, VEC_ALIGN);
#else
  *buf = (int*)malloc(sizeof(int) * len);
#endif
  if (!*buf) {
    TERMINATE("Could not allocate host int data.\n");
  }

#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }
}

// Allocates a data array
void deallocate_data(double* buf) { gpu_check(cudaFree(buf)); }

// Allocates a data array
void deallocate_host_data(double* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Allocates a data array
void deallocate_host_float_data(float* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Allocates a data array
void deallocate_int_data(int* buf) { gpu_check(cudaFree(buf)); }

// Allocates a data array
void deallocate_host_int_data(int* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Copy a buffer to/from the device
void copy_buffer(const size_t len, double** src, double** dst, int send) {
  if (send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(double) * len, cudaMemcpyHostToDevice));
  } else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(double) * len, cudaMemcpyDeviceToHost));
  }
}

// Copy a buffer to/from the device
void copy_float_buffer(const size_t len, float** src, float** dst, int send) {
  if (send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(float) * len, cudaMemcpyHostToDevice));
  } else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(float) * len, cudaMemcpyDeviceToHost));
  }
}

// Copy a buffer to/from the device
void copy_int_buffer(const size_t len, int** src, int** dst, int send) {
  if (send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(int) * len, cudaMemcpyHostToDevice));
  } else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(int) * len, cudaMemcpyDeviceToHost));
  }
}

// Copy a buffer to/from the device
void copy_uint64_buffer(const size_t len, uint64_t** src, uint64_t** dst,
                        int send) {
  if (send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(uint64_t) * len, cudaMemcpyHostToDevice));
  } else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(uint64_t) * len, cudaMemcpyDeviceToHost));
  }
}

// Move a host buffer onto the device
void move_host_buffer_to_device(const size_t len, double** src, double** dst) {
  allocate_data(dst, len);
  copy_buffer(len, src, dst, SEND);
  deallocate_host_data(*src);
}

// Move a host buffer onto the device
void move_host_float_buffer_to_device(const size_t len, float** src, float** dst) {
  allocate_float_data(dst, len);
  copy_float_buffer(len, src, dst, SEND);
  deallocate_host_float_data(*src);
}


// Initialises mesh data in device specific manner
void mesh_data_init_2d(const int local_nx, const int local_ny,
                       const int global_nx, const int global_ny, const int pad,
                       const int x_off, const int y_off, const double width,
                       const double height, double* edgex, double* edgey,
                       double* edgedx, double* edgedy, double* celldx,
                       double* celldy) {
  // Simple uniform rectilinear initialisation
  int nblocks = ceil((local_nx + 1) / (double)NTHREADS);
  mesh_data_init_dx<<<nblocks, NTHREADS>>>(
      local_nx, local_ny, global_nx, global_ny, pad, x_off, width, edgex, edgey,
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil((local_ny + 1) / (double)NTHREADS);
  mesh_data_init_dy<<<nblocks, NTHREADS>>>(
      local_nx, local_ny, global_nx, global_ny, pad, y_off, height, edgex,
      edgey, edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());
}

// Initialise state data in device specific manner
void set_problem_2d(const int local_nx, const int local_ny, const int pad,
                    const double mesh_width, const double mesh_height,
                    const double* edgex, const double* edgey, const int ndims,
                    const char* problem_def_filename, double* density, double* energy,
                    double* temperature) {
  int* h_keys;
  int* d_keys;
  allocate_int_data(&d_keys, MAX_KEYS);
  allocate_host_int_data(&h_keys, MAX_KEYS);

  double* h_values;
  double* d_values;
  allocate_data(&d_values, MAX_KEYS);
  allocate_host_data(&h_values, MAX_KEYS);

  int nentries = 0;
  while (1) {
    char specifier[MAX_STR_LEN];
    char keys[MAX_STR_LEN * MAX_KEYS];
    sprintf(specifier, "problem_%d", nentries++);

    int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys,
                                 h_values, &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = h_values[nkeys - 4] * mesh_width;
    double ypos = h_values[nkeys - 3] * mesh_height;
    double width = h_values[nkeys - 2] * mesh_width;
    double height = h_values[nkeys - 1] * mesh_height;

    for (int kk = 0; kk < nkeys - (2 * ndims); ++kk) {
      const char* key = &keys[kk * MAX_STR_LEN];
      if (strmatch(key, "density")) {
        h_keys[kk] = DENSITY_KEY;
      } else if (strmatch(key, "energy")) {
        h_keys[kk] = ENERGY_KEY;
      } else if (strmatch(key, "temperature")) {
        h_keys[kk] = TEMPERATURE_KEY;
      } else {
        TERMINATE("Found unrecognised key in %s : %s.\n", problem_def_filename,
                  key);
      }
    }

    copy_int_buffer(MAX_KEYS, &h_keys, &d_keys, SEND);
    copy_buffer(MAX_KEYS, &h_values, &d_values, SEND);

    // Introduce a problem
    const int nblocks = ceil(local_nx * local_ny / (double)NTHREADS);
    initialise_problem_state<<<nblocks, NTHREADS>>>(
        local_nx, local_ny, nkeys, ndims, xpos, ypos, width, height, edgey,
        edgex, density, energy, temperature, d_keys, d_values);
    gpu_check(cudaDeviceSynchronize());
  }

  deallocate_host_int_data(h_keys);
  deallocate_host_data(h_values);
}

void mesh_data_init_3d(const int local_nx, const int local_ny,
                       const int local_nz, const int global_nx,
                       const int global_ny, const int global_nz, const int pad,
                       const int x_off, const int y_off, const int z_off,
                       const double width, const double height,
                       const double depth, double* edgex, double* edgey,
                       double* edgez, double* edgedx, double* edgedy,
                       double* edgedz, double* celldx, double* celldy,
                       double* celldz) {

  // Initialise as in the 2d case
  mesh_data_init_2d(local_nx, local_ny, global_nx, global_ny, pad, x_off, y_off,
      width, height, edgex, edgey, edgedx, edgedy, celldx,
      celldy);

  // Initialises mesh data for the y dimension
  const int nblocks = ceil((local_nz+1)/(double)NTHREADS);
  mesh_data_init_dz<<<nblocks, NTHREADS>>>(
      local_nz, global_nz, pad, z_off, depth, edgez, edgedz, celldz);
}

void state_data_init_3d(const int local_nx, const int local_ny,
    const int local_nz, const int global_nx,
    const int global_ny, const int global_nz, const int pad,
    const int x_off, const int y_off, const int z_off,
    double* density, double* energy, double* rho_old, double* P,
    double* Qxx, double* Qyy, double* Qzz, double* temperature,
    double* p, double* rho_u, double* rho_v, double* rho_w,
    double* F_x, double* F_y, double* F_z, double* uF_x,
    double* uF_y, double* uF_z, double* vF_x, double* vF_y,
    double* vF_z, double* wF_x, double* wF_y, double* wF_z,
    double* reduce_array) {
  TERMINATE("CUDA 3d INCOMPLETE");
}

void set_problem_3d(const int local_nx, const int local_ny, const int local_nz,
    const int pad, const double mesh_width,
    const double mesh_height, const double mesh_depth,
    const double* edgex, const double* edgey,
    const double* edgez, const int ndims,
    const char* problem_def_filename, double* density, double* energy,
    double* temperature) {

  int* h_keys;
  int* d_keys;
  allocate_int_data(&d_keys, MAX_KEYS);
  allocate_host_int_data(&h_keys, MAX_KEYS);

  double* h_values;
  double* d_values;
  allocate_data(&d_values, MAX_KEYS);
  allocate_host_data(&h_values, MAX_KEYS);

  int nentries = 0;
  while (1) {
    char specifier[MAX_STR_LEN];
    char keys[MAX_STR_LEN * MAX_KEYS];
    sprintf(specifier, "problem_%d", nentries++);

    int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys, h_values,
          &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = h_values[nkeys - 6] * mesh_width;
    double ypos = h_values[nkeys - 5] * mesh_height;
    double zpos = h_values[nkeys - 4] * mesh_depth;
    double width = h_values[nkeys - 3] * mesh_width;
    double height = h_values[nkeys - 2] * mesh_height;
    double depth = h_values[nkeys - 1] * mesh_depth;

    copy_int_buffer(MAX_KEYS, &h_keys, &d_keys, SEND);
    copy_buffer(MAX_KEYS, &h_values, &d_values, SEND);

    const int nblocks = ceil(local_nx*local_ny*local_nz/(double)NTHREADS);
    initialise_problem_state_3d<<<nblocks, NTHREADS>>>(
        local_nx, local_ny, local_nz, nkeys, ndims, xpos,  ypos,  zpos,  width,
        height,  depth,  edgex, edgey,  edgez, density, energy, temperature, d_keys, d_values);
    gpu_check(cudaDeviceSynchronize());
  }

  deallocate_host_int_data(h_keys);
  deallocate_host_data(h_values);
}

// Finds the normals for all boundary cells
void find_boundary_normals(UnstructuredMesh* umesh, int* boundary_edge_list) {
  TERMINATE("find_boundary_normals needs implementing.");
}
