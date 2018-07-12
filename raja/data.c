#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include "../umesh.h"
#include "../shared_data.h"
#include "shared.h"
#include <math.h>
#include <stdlib.h>

// Allocates a double precision array
size_t allocate_data(double** buf, size_t len) {
  if(len == 0) {
    return 0;
  }

#ifdef RAJA_USE_CUDA

#ifdef CUDA_MANAGED_MEM
  gpu_check(cudaMallocManaged((void**)buf, sizeof(double) * len));
#else
  gpu_check(cudaMalloc((void**)buf, sizeof(double) * len));
#endif // CUDA_MANAGED_MEM

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0.0;
      });

#else 

  allocate_host_data(buf, len);

#endif // RAJA_USE_CUDA

  return sizeof(double) * len;
}

// Allocates a single precision array
size_t allocate_float_data(float** buf, size_t len) {
  if(len == 0) {
    return 0;
  }

#ifdef RAJA_USE_CUDA

#ifdef CUDA_MANAGED_MEM
  gpu_check(cudaMallocManaged((void**)buf, sizeof(double) * len));
#else
  gpu_check(cudaMalloc((void**)buf, sizeof(double) * len));
#endif // CUDA_MANAGED_MEM

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0.0f;
      });

#else

  allocate_host_float_data(buf, len);

#endif // RAJA_USE_CUDA

  return sizeof(double) * len;
}

// Allocates a 32-bit integer array
size_t allocate_int_data(int** buf, size_t len) {
  if(len == 0) {
    return 0;
  }

#ifdef RAJA_USE_CUDA

#ifdef CUDA_MANAGED_MEM
  gpu_check(cudaMallocManaged((void**)buf, sizeof(int) * len));
#else
  gpu_check(cudaMalloc((void**)buf, sizeof(int) * len));
#endif // CUDA_MANAGED_MEM

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0;
      });

#else

  allocate_host_int_data(buf, len);

#endif // RAJA_USE_CUDA

  return sizeof(int) * len;
}

// Allocates a 64-bit integer array
size_t allocate_uint64_data(uint64_t** buf, const size_t len) {
  if(len == 0) {
    return 0;
  }
#ifdef RAJA_USE_CUDA

#ifdef CUDA_MANAGED_MEM
  gpu_check(cudaMallocManaged((void**)buf, sizeof(uint64_t) * len));
#else
  gpu_check(cudaMalloc((void**)buf, sizeof(uint64_t) * len));
#endif // CUDA_MANAGED_MEM

#else

#ifdef INTEL
  *buf = (uint64_t*)_mm_malloc(sizeof(uint64_t) * len, VEC_ALIGN);
#else
  *buf = (uint64_t*)malloc(sizeof(uint64_t) * len);
#endif // INTEL

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

#endif // RAJA_USE_CUDA

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0;
      });

  return sizeof(uint64_t) * len;
}

// Allocates a complex double array
size_t allocate_complex_double_data(_Complex double** buf, const size_t len) {
  TERMINATE("Not implemented\n");
}

// Allocates a host copy of some buffer
void allocate_host_data(double** buf, size_t len) { 
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double) * len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double) * len);
#endif // INTEL

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0.0;
      });
}

// Allocates a host copy of some buffer
void allocate_host_int_data(int** buf, size_t len) {
#ifdef INTEL
  *buf = (int*)_mm_malloc(sizeof(int) * len, VEC_ALIGN);
#else
  *buf = (int*)malloc(sizeof(int) * len);
#endif // INTEL

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0;
      });
}

void allocate_host_float_data(float** buf, size_t len) {
#ifdef INTEL
  *buf = (float*)_mm_malloc(sizeof(float) * len, VEC_ALIGN);
#else
  *buf = (float*)malloc(sizeof(float) * len);
#endif // INTEL

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, len), [=] (int i) {
      (*buf)[i] = 0.0f;
      });
}

// Deallocate a double array
void deallocate_data(double* buf) {
#ifdef RAJA_USE_CUDA

  gpu_check(cudaFree(buf));

#else

#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif

#endif // RAJA_USE CUDA
}

// Deallocates a float array
void deallocate_float_data(float* buf) {
#ifdef RAJA_USE_CUDA

  gpu_check(cudaFree(buf));

#else

#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif

#endif // RAJA_USE CUDA
}

// Deallocation of host data
void deallocate_host_data(double* buf) {
#ifdef RAJA_USE_CUDA
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
#endif
}

// Deallocates a 32-bit integer array
void deallocate_int_data(int* buf) {
#ifdef RAJA_USE_CUDA

  gpu_check(cudaFree(buf));

#else

#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif

#endif // RAJA_USE CUDA
}

// Deallocates a 64-bit integer array
void deallocate_uint64_t_data(uint64_t* buf) {
#ifdef RAJA_USE_CUDA

  gpu_check(cudaFree(buf));

#else

#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif

#endif // RAJA_USE CUDA
}

// Deallocates complex double data
void deallocate_complex_double_data(_Complex double* buf) {
#ifdef RAJA_USE_CUDA

  gpu_check(cudaFree(buf));

#else

#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif

#endif // RAJA_USE CUDA
}

// Allocates a data array
void deallocate_host_int_data(int* buf) {
#ifdef RAJA_USE_CUDA

  gpu_check(cudaFree(buf));

#else

#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif

#endif // RAJA_USE CUDA
}

// Just swaps the buffers on the host
void copy_buffer(const size_t len, double** src, double** dst, int send) {
#ifdef RAJA_USE_CUDA

  if (send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(double) * len, cudaMemcpyHostToDevice));
  } else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(double) * len, cudaMemcpyDeviceToHost));
  }
  gpu_check(cudaDeviceSynchronize());

#else

  double* temp = *src;
  *src = *dst;
  *dst = temp;

#endif
}

// Just swaps the buffers on the host
void copy_int_buffer(const size_t len, int** src, int** dst, int send) {
#ifdef RAJA_USE_CUDA

  if (send) {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(int) * len, cudaMemcpyHostToDevice));
  } else {
    gpu_check(
        cudaMemcpy(*dst, *src, sizeof(int) * len, cudaMemcpyDeviceToHost));
  }
  gpu_check(cudaDeviceSynchronize());

#else

  int* temp = *src;
  *src = *dst;
  *dst = temp;

#endif
}

// Move a host buffer onto the device
void move_host_buffer_to_device(const size_t len, double** src, double** dst) {
#ifdef RAJA_USE_CUDA

  allocate_data(dst, len);
  copy_buffer(len, src, dst, SEND);
  deallocate_host_data(*src);

#else

  copy_buffer(len, src, dst, SEND);

#endif
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(const int local_nx, const int local_ny,
    const int global_nx, const int global_ny, const int pad,
    const int x_off, const int y_off, const double width,
    const double height, double* edgex, double* edgey,
    double* edgedx, double* edgedy, double* celldx,
    double* celldy) {

  // Simple uniform rectilinear initialisation
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_nx+1), [=] (int ii) {
    edgedx[ii] = width / (global_nx);

    // Note: correcting for padding
    edgex[ii] = edgedx[ii] * (x_off + ii - pad);
  });

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_nx), [=] (int ii) {
    celldx[ii] = width / (global_nx);
  });

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_ny+1), [=] (int ii) {
    edgedy[ii] = height / (global_ny);

    // Note: correcting for padding
    edgey[ii] = edgedy[ii] * (y_off + ii - pad);
  });

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_ny), [=] (int ii) {
    celldy[ii] = height / (global_ny);
  });
}

// Initialises mesh data in device specific manner
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

  // Simple uniform rectilinear initialisation
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_nz+1), [=] (int ii) {
    edgedz[ii] = depth / (global_nz);
    edgez[ii] = edgedz[ii] * (z_off + ii - pad);
  });

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_nz), [=] (int ii) {
    celldz[ii] = depth / (global_nz);
  });
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

    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_nx*local_ny), [=] (int i) {
        const int ii = i / local_nx;
        const int jj = i % local_nx;
        double global_xpos = edgex[jj];
        double global_ypos = edgey[ii];

        // Check we are in bounds of the problem entry
        if (global_xpos >= xpos && 
          global_ypos >= ypos && 
          global_xpos < xpos + width && 
          global_ypos < ypos + height) {

        // The upper bound excludes the bounding box for the entry
        for (int nn = 0; nn < nkeys - (2 * ndims); ++nn) {
        const int key = d_keys[nn];
        if (key == DENSITY_KEY) {
        density[i] = d_values[nn];
        } else if (key == ENERGY_KEY) {
          energy[i] = d_values[nn];
        } else if (key == TEMPERATURE_KEY) {
          temperature[i] = d_values[nn];
        }
        }
        }
    });
  }

  deallocate_host_int_data(h_keys);
  deallocate_host_data(h_values);
}

// Initialise state data in device specific manner
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

    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, local_nx*local_ny*local_nz), [=] (int i) {
        const int ii = (i / local_nx*local_ny);
        const int jj = (i / local_nx) % local_ny;
        const int kk = (i % local_nx);
        double global_xpos = edgex[kk];
        double global_ypos = edgey[jj];
        double global_zpos = edgez[ii];

        // Check we are in bounds of the problem entry
        if (global_xpos >= xpos && 
          global_ypos >= ypos && 
          global_zpos >= zpos &&
          global_xpos < xpos + width && 
          global_ypos < ypos + height && 
          global_zpos < zpos + depth) {

        // The upper bound excludes the bounding box for the entry
        for (int nn = 0; nn < nkeys - (2 * ndims); ++nn) {
        const int key = d_keys[nn];
        if (key == DENSITY_KEY) {
        density[i] = d_values[nn];
        } else if (key == ENERGY_KEY) {
          energy[i] = d_values[nn];
        } else if (key == TEMPERATURE_KEY) {
          temperature[i] = d_values[nn];
        }
        }
        }
    });
  }

  deallocate_host_int_data(h_keys);
  deallocate_host_data(h_values);
}

// Finds the normals for all boundary cells
void find_boundary_normals(UnstructuredMesh* umesh, int* boundary_face_list) {

  TERMINATE("%s not yet implemented.", __func__);
}

// Finds the normals for all boundary cells
void find_boundary_normals_3d(UnstructuredMesh* umesh,
    int* boundary_face_list) {

  TERMINATE("%s not yet implemented.", __func__);
}
