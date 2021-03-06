#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include "../umesh.h"
#include <math.h>
#include <stdlib.h>

// Allocates a double precision array
size_t allocate_data(double** buf, size_t len) {
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double) * len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

// Perform first-touch
#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }

  return sizeof(double) * len;
}

// Allocates a single precision array
size_t allocate_float_data(float** buf, size_t len) {
#ifdef INTEL
  *buf = (float*)_mm_malloc(sizeof(float) * len, VEC_ALIGN);
#else
  *buf = (float*)malloc(sizeof(float) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

// Perform first-touch
#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }

  return sizeof(double) * len;
}

// Allocates a 32-bit integer array
size_t allocate_int_data(int** buf, size_t len) {
#ifdef INTEL
  *buf = (int*)_mm_malloc(sizeof(int) * len, VEC_ALIGN);
#else
  *buf = (int*)malloc(sizeof(int) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

// Perform first-touch
#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0;
  }

  return sizeof(int) * len;
}

// Allocates a 64-bit integer array
size_t allocate_uint64_data(uint64_t** buf, const size_t len) {
#ifdef INTEL
  *buf = (uint64_t*)_mm_malloc(sizeof(uint64_t) * len, VEC_ALIGN);
#else
  *buf = (uint64_t*)malloc(sizeof(uint64_t) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

// Perform first-touch
#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0;
  }

  return sizeof(uint64_t) * len;
}

// Allocates a complex double array
size_t allocate_complex_double_data(_Complex double** buf, const size_t len) {
#ifdef INTEL
  *buf = (_Complex double*)_mm_malloc(sizeof(_Complex double) * len, VEC_ALIGN);
#else
  *buf = (_Complex double*)malloc(sizeof(_Complex double) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }

// Perform first-touch
#pragma omp parallel for
  for (size_t ii = 0; ii < len; ++ii) {
    (*buf)[ii] = 0.0;
  }

  return sizeof(_Complex double) * len;
}

// Allocates a host copy of some buffer
void allocate_host_data(double** buf, size_t len) { allocate_data(buf, len); }

// Allocates a host copy of some buffer
void allocate_host_int_data(int** buf, size_t len) {
  allocate_int_data(buf, len);
}

// Deallocate a double array
void deallocate_data(double* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Deallocates a float array
void deallocate_float_data(float* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Deallocation of host data
void deallocate_host_data(double* buf) {
  // Not necessary as host-only
}

// Deallocates a 32-bit integer array
void deallocate_int_data(int* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Deallocates a 64-bit integer array
void deallocate_uint64_t_data(uint64_t* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Deallocates complex double data
void deallocate_complex_double_data(_Complex double* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Allocates a data array
void deallocate_host_int_data(int* buf) {
  // Not necessary as host-only
}

// Just swaps the buffers on the host
void copy_buffer(const size_t len, double** src, double** dst, int send) {
  double* temp = *src;
  *src = *dst;
  *dst = temp;
}

// Just swaps the buffers on the host
void copy_int_buffer(const size_t len, int** src, int** dst, int send) {
  int* temp = *src;
  *src = *dst;
  *dst = temp;
}

// Move a host buffer onto the device
void move_host_buffer_to_device(const size_t len, double** src, double** dst) {
  copy_buffer(len, src, dst, SEND);
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(const int local_nx, const int local_ny,
                       const int global_nx, const int global_ny, const int pad,
                       const int x_off, const int y_off, const double width,
                       const double height, double* edgex, double* edgey,
                       double* edgedx, double* edgedy, double* celldx,
                       double* celldy) {
// Simple uniform rectilinear initialisation
#pragma omp parallel for
  for (int ii = 0; ii < local_nx + 1; ++ii) {
    edgedx[ii] = width / (global_nx);

    // Note: correcting for padding
    edgex[ii] = edgedx[ii] * (x_off + ii - pad);
  }
#pragma omp parallel for
  for (int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = width / (global_nx);
  }
#pragma omp parallel for
  for (int ii = 0; ii < local_ny + 1; ++ii) {
    edgedy[ii] = height / (global_ny);

    // Note: correcting for padding
    edgey[ii] = edgedy[ii] * (y_off + ii - pad);
  }
#pragma omp parallel for
  for (int ii = 0; ii < local_ny; ++ii) {
    celldy[ii] = height / (global_ny);
  }
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
#pragma omp parallel for
  for (int ii = 0; ii < local_nz + 1; ++ii) {
    edgedz[ii] = depth / (global_nz);
    edgez[ii] = edgedz[ii] * (z_off + ii - pad);
  }
#pragma omp parallel for
  for (int ii = 0; ii < local_nz; ++ii) {
    celldz[ii] = depth / (global_nz);
  }
}

// Initialise state data in device specific manner
void set_problem_2d(const int local_nx, const int local_ny, const int pad,
                    const double mesh_width, const double mesh_height,
                    const double* edgex, const double* edgey, const int ndims,
                    const char* problem_def_filename, double* rho, double* e,
                    double* x) {
  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * MAX_STR_LEN);
  double* values = (double*)malloc(sizeof(double) * MAX_KEYS);

  int nentries = 0;
  while (1) {
    char specifier[MAX_STR_LEN];
    sprintf(specifier, "problem_%d", nentries++);

    int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys, values,
                                 &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = values[nkeys - 4] * mesh_width;
    double ypos = values[nkeys - 3] * mesh_height;
    double width = values[nkeys - 2] * mesh_width;
    double height = values[nkeys - 1] * mesh_height;

    // Loop through the mesh and set the problem
    for (int ii = pad; ii < local_ny - pad; ++ii) {
      for (int jj = pad; jj < local_nx - pad; ++jj) {
        double global_xpos = edgex[jj];
        double global_ypos = edgey[ii];

        // Check we are in bounds of the problem entry
        if (global_xpos >= xpos && global_ypos >= ypos &&
            global_xpos < xpos + width && global_ypos < ypos + height) {
          // The upper bound excludes the bounding box for the entry
          for (int kk = 0; kk < nkeys - (2 * ndims); ++kk) {
            const char* key = &keys[kk * MAX_STR_LEN];
            if (strmatch(key, "density")) {
              rho[ii * local_nx + jj] = values[kk];
            } else if (strmatch(key, "energy")) {
              e[ii * local_nx + jj] = values[kk];
            } else if (strmatch(key, "temperature")) {
              x[ii * local_nx + jj] = values[kk];
            } else {
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
void set_problem_3d(const int local_nx, const int local_ny, const int local_nz,
                    const int pad, const double mesh_width,
                    const double mesh_height, const double mesh_depth,
                    const double* edgex, const double* edgey,
                    const double* edgez, const int ndims,
                    const char* problem_def_filename, double* rho, double* e,
                    double* x) {
  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * MAX_STR_LEN);
  double* values = (double*)malloc(sizeof(double) * MAX_KEYS);

  int nentries = 0;
  while (1) {
    char specifier[MAX_STR_LEN];
    sprintf(specifier, "problem_%d", nentries++);

    int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys, values,
                                 &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = values[nkeys - 6] * mesh_width;
    double ypos = values[nkeys - 5] * mesh_height;
    double zpos = values[nkeys - 4] * mesh_depth;
    double width = values[nkeys - 3] * mesh_width;
    double height = values[nkeys - 2] * mesh_height;
    double depth = values[nkeys - 1] * mesh_depth;

    // Loop through the mesh and set the problem
    for (int ii = pad; ii < local_nz - pad; ++ii) {
      for (int jj = pad; jj < local_ny - pad; ++jj) {
        for (int kk = pad; kk < local_nx - pad; ++kk) {
          double global_xpos = edgex[kk];
          double global_ypos = edgey[jj];
          double global_zpos = edgez[ii];

          // Check we are in bounds of the problem entry
          if (global_xpos >= xpos && global_ypos >= ypos &&
              global_zpos >= zpos && global_xpos < xpos + width &&
              global_ypos < ypos + height && global_zpos < zpos + depth) {
            // The upper bound excludes the bounding box for the entry
            for (int ee = 0; ee < nkeys - (2 * ndims); ++ee) {
              const int index =
                  (ii * local_nx * local_ny) + (jj * local_nx) + (kk);
              const char* key = &keys[ee * MAX_STR_LEN];
              if (strmatch(key, "density")) {
                rho[(index)] = values[ee];
              } else if (strmatch(key, "energy")) {
                e[(index)] = values[ee];
              } else if (strmatch(key, "temperature")) {
                x[(index)] = values[ee];
              } else {
                TERMINATE("Found unrecognised key in %s : %s.\n",
                          problem_def_filename, key);
              }
            }
          }
        }
      }
    }
  }

  free(keys);
  free(values);
}

// Finds the normals for all boundary cells
void find_boundary_normals(UnstructuredMesh* umesh, int* boundary_face_list) {
// Loop through all of the boundary cells and find their normals
#pragma omp parallel for
  for (int nn = 0; nn < umesh->nnodes; ++nn) {
    const int boundary_index = umesh->boundary_index[(nn)];
    if (boundary_index == IS_INTERIOR) {
      continue;
    }

    double normal_x = 0.0;
    double normal_y = 0.0;

    for (int bb1 = 0; bb1 < umesh->nboundary_nodes; ++bb1) {
      const int node0 = boundary_face_list[bb1 * 2];
      const int node1 = boundary_face_list[bb1 * 2 + 1];

      if (node0 == nn || node1 == nn) {
        const double node0_x = umesh->nodes_x0[(node0)];
        const double node0_y = umesh->nodes_y0[(node0)];
        const double node1_x = umesh->nodes_x0[(node1)];
        const double node1_y = umesh->nodes_y0[(node1)];

        normal_x += node0_y - node1_y;
        normal_y += -(node0_x - node1_x);
      }
    }

    // We are fixed if we are one of the four corners
    if ((umesh->nodes_x0[(nn)] == 0.0 || umesh->nodes_x0[(nn)] == 1.0) &&
        (umesh->nodes_y0[(nn)] == 0.0 || umesh->nodes_y0[(nn)] == 1.0)) {
      umesh->boundary_type[(boundary_index)] = IS_CORNER;
    } else {
      umesh->boundary_type[(boundary_index)] = IS_BOUNDARY;
    }

    const double normal_mag = sqrt(normal_x * normal_x + normal_y * normal_y);
    umesh->boundary_normal_x[(boundary_index)] = normal_x / normal_mag;
    umesh->boundary_normal_y[(boundary_index)] = normal_y / normal_mag;
  }
}

// Finds the normals for all boundary cells
void find_boundary_normals_3d(UnstructuredMesh* umesh,
                              int* boundary_face_list) {

  TERMINATE("%s not yet implemented.", __func__);
#if 0
// Loop through all of the boundary cells and find their normals
#pragma omp parallel for
  for (int nn = 0; nn < umesh->nnodes; ++nn) {
    const int boundary_index = umesh->boundary_index[(nn)];
    if (boundary_index == IS_INTERIOR) {
      continue;
    }

    double normal_x = 0.0;
    double normal_y = 0.0;

    for (int bb1 = 0; bb1 < umesh->nboundary_faces; ++bb1) {
      const int node0 = boundary_face_list[bb1 * 2];
      const int node1 = boundary_face_list[bb1 * 2 + 1];

      if (node0 == nn || node1 == nn) {
        const double node0_x = umesh->nodes_x0[(node0)];
        const double node0_y = umesh->nodes_y0[(node0)];
        const double node1_x = umesh->nodes_x0[(node1)];
        const double node1_y = umesh->nodes_y0[(node1)];

        normal_x += node0_y - node1_y;
        normal_y += -(node0_x - node1_x);
      }
    }

    // We are fixed if we are one of the four corners
    if ((umesh->nodes_x0[(nn)] == 0.0 || umesh->nodes_x0[(nn)] == 1.0) &&
        (umesh->nodes_y0[(nn)] == 0.0 || umesh->nodes_y0[(nn)] == 1.0)) {
      umesh->boundary_type[(boundary_index)] = IS_CORNER;
    } else {
      umesh->boundary_type[(boundary_index)] = IS_BOUNDARY;
    }

    const double normal_mag = sqrt(normal_x * normal_x + normal_y * normal_y);
    umesh->boundary_normal_x[(boundary_index)] = normal_x / normal_mag;
    umesh->boundary_normal_y[(boundary_index)] = normal_y / normal_mag;
  }
#endif // if 0
}
