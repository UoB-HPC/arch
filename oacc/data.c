#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include "../umesh.h"
#include <math.h>
#include <stdlib.h>

// Checks if two strings match
#pragma acc routine seq
int device_strmatch(const char* str1, const char* str2) {
  int ii = 0;
  for (ii = 0; str1[ii] != '\0'; ++ii) {
    if (str1[ii] != str2[ii]) {
      return 0;
    }
  }
  return str1[ii] == str2[ii];
}

// Allocates some double precision data
size_t allocate_data(double** buf, size_t len) {
  allocate_host_data(buf, len);

  double* local_buf = *buf;
#pragma acc enter data copyin(local_buf[:len])

#pragma acc kernels
#pragma acc loop independent
  for (size_t ii = 0; ii < len; ++ii) {
    local_buf[ii] = 0.0;
  }

  return sizeof(double) * len;
}

// Allocates some int precision data
size_t allocate_int_data(int** buf, size_t len) {
  allocate_host_int_data(buf, len);

  int* local_buf = *buf;
#pragma acc enter data copyin(local_buf[ : len])

#pragma acc kernels
#pragma acc loop independent
  for (size_t ii = 0; ii < len; ++ii) {
    local_buf[ii] = 0;
  }

  return sizeof(int) * len;
}

// Allocates some int precision data
size_t allocate_uint64_data(uint64_t** buf, size_t len) {
  allocate_host_uint64_data(buf, len);

  uint64_t* local_buf = *buf;
#pragma acc enter data copyin(local_buf[ : len])

#pragma acc kernels
#pragma acc loop independent
  for (size_t ii = 0; ii < len; ++ii) {
    local_buf[ii] = 0;
  }

  return sizeof(uint64_t) * len;
}

// Allocates a host copy of some buffer
void allocate_host_data(double** buf, size_t len) {
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double) * len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }
}

// Allocates a host copy of some integer buffer
void allocate_host_int_data(int** buf, size_t len) {
#ifdef INTEL
  *buf = (int*)_mm_malloc(sizeof(int) * len, VEC_ALIGN);
#else
  *buf = (int*)malloc(sizeof(int) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }
}

void allocate_host_uint64_data(uint64_t** buf, const size_t len) {
#ifdef INTEL
  *buf = (uint64_t*)_mm_malloc(sizeof(uint64_t) * len, VEC_ALIGN);
#else
  *buf = (uint64_t*)malloc(sizeof(uint64_t) * len);
#endif

  if (*buf == NULL) {
    TERMINATE("Failed to allocate a data array.\n");
  }
}

// Allocates a data array
void deallocate_data(double* buf) {
#pragma acc exit data delete(buf)
}

// Allocates a data array
void deallocate_int_data(int* buf) {
#pragma acc exit data delete(buf)
}

// Allocates a data array
void deallocate_host_data(double* buf) {
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Synchronise data
void copy_buffer(const size_t len, double** src, double** dst, int send) {
  double* local_src = *src;
  if (send == SEND) {
#pragma acc update device(local_src[ : len])
  } else {
#pragma acc update self(local_src[ : len])
  }
  *dst = *src;
}

// Move a host buffer onto the device
void move_host_buffer_to_device(const size_t len, double** src, double** dst) {
  double* local_src = *src;

#pragma acc enter data copyin(local_src[:len])

  *dst = local_src;
}

// Initialises mesh data in device specific manner
void mesh_data_init_2d(const int local_nx, const int local_ny,
    const int global_nx, const int global_ny, const int pad,
    const int x_off, const int y_off, const double width,
    const double height, double* edgex, double* edgey,
    double* edgedx, double* edgedy, double* celldx,
    double* celldy) {

  // Simple uniform rectilinear initialisation
#pragma acc kernels
#pragma acc loop independent
  for (int ii = 0; ii < local_nx + 1; ++ii) {
    edgedx[ii] = width / (global_nx);

    // Note: correcting for padding
    edgex[ii] = edgedx[ii] * (x_off + ii - pad);
  }

#pragma acc kernels
#pragma acc loop independent
  for (int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = width / (global_nx);
  }

#pragma acc kernels
#pragma acc loop independent
  for (int ii = 0; ii < local_ny + 1; ++ii) {
    edgedy[ii] = height / (global_ny);

    // Note: correcting for padding
    edgey[ii] = edgedy[ii] * (y_off + ii - pad);
  }

#pragma acc kernels
#pragma acc loop independent
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
#pragma acc kernels
#pragma acc loop independent
  for (int ii = 0; ii < local_nz + 1; ++ii) {
    edgedz[ii] = depth / (global_nz);
    edgez[ii] = edgedz[ii] * (z_off + ii - pad);
  }

#pragma acc kernels
#pragma acc loop independent
  for (int ii = 0; ii < local_nz; ++ii) {
    celldz[ii] = depth / (global_nz);
  }
}

// Initialise state data in device specific manner
void set_problem_2d(const int local_nx, const int local_ny, const int pad,
    const double mesh_width, const double mesh_height,
    const double* edgex, const double* edgey, const int ndims,
    const char* problem_def_filename, double* density, double* energy,
    double* temperature) {

  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * MAX_STR_LEN);
  double* values;
  allocate_data(&values, MAX_KEYS);

#pragma acc update host(edgex[:local_nx+1], edgey[:local_ny+1])

  int nentries = 0;
  while (1) {
    char specifier[MAX_STR_LEN];
    sprintf(specifier, "problem_%d", nentries++);

    int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys, values,
          &nkeys)) {
      break;
    }

    copy_buffer(MAX_KEYS, &values, &values, SEND);

    // The last four keys are the bound specification
    double xpos = values[nkeys - 4] * mesh_width;
    double ypos = values[nkeys - 3] * mesh_height;
    double width = values[nkeys - 2] * mesh_width;
    double height = values[nkeys - 1] * mesh_height;

    int failed = 0;

    // Loop through the mesh and set the problem
#pragma omp parallel for
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
              density[ii * local_nx + jj] = values[kk];
            } else if (strmatch(key, "energy")) {
              energy[ii * local_nx + jj] = values[kk];
            } else if (strmatch(key, "temperature")) {
              temperature[ii * local_nx + jj] = values[kk];
            } else {
              TERMINATE("Found unrecognised key in %s.\n", problem_def_filename);
            }
          }
        }
      }
    }
  }

#pragma acc update device(density[:local_nx*local_ny], energy[:local_nx*local_ny], \
    temperature[:local_nx*local_ny])

  free(keys);
  deallocate_data(values);
}

// Initialise state data in device specific manner
void set_problem_3d(const int local_nx, const int local_ny, const int local_nz,
    const int pad, const double mesh_width,
    const double mesh_height, const double mesh_depth,
    const double* edgex, const double* edgey,
    const double* edgez, const int ndims,
    const char* problem_def_filename, double* density, double* energy,
    double* temperature) {

  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * MAX_STR_LEN);
  double* values;
  allocate_data(&values, MAX_KEYS);

#pragma acc update host(edgex[:local_nx+1], edgey[:local_ny+1], edgez[:local_nz+1])

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

    int failed = 0;

    // Loop through the mesh and set the problem
#pragma omp parallel for
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
                ii * local_nx * local_ny + jj * local_nx + kk;
              const char* key = &keys[ee * MAX_STR_LEN];
              if (device_strmatch(key, "density")) {
                density[index] = values[ee];
              } else if (device_strmatch(key, "energy")) {
                energy[index] = values[ee];
              } else if (device_strmatch(key, "temperature")) {
                temperature[index] = values[ee];
              } else {
                failed++;
              }
            }
          }
        }
      }
    }

    if(failed) {
      TERMINATE("Found unrecognised key in %s.\n", problem_def_filename);
    }
  }

#pragma acc update device(density[:local_nx*local_ny*local_nz])
#pragma acc update device(energy[:local_nx*local_ny*local_nz])
#pragma acc update device(temperature[:local_nx*local_ny*local_nz])

  free(keys);
  deallocate_data(values);
}

// Finds the normals for all boundary cells
void find_boundary_normals(UnstructuredMesh* umesh, int* boundary_edge_list) {

  const int nnodes = umesh->nnodes;
  const int nboundary_nodes = umesh->nboundary_nodes;
  const int* boundary_index = umesh->boundary_index;
  const double* nodes_x0 = umesh->nodes_x0;
  const double* nodes_y0 = umesh->nodes_y0;
  const double* nodes_z0 = umesh->nodes_z0;
  int* boundary_type = umesh->boundary_type;
  double* boundary_normal_x = umesh->boundary_normal_x;
  double* boundary_normal_y = umesh->boundary_normal_y;

  // Loop through all of the boundary cells and find their normals
#pragma acc kernels
#pragma acc loop independent
  for (int nn = 0; nn < nnodes; ++nn) {
    const int bi = boundary_index[(nn)];
    if (bi == IS_INTERIOR) {
      continue;
    }

    double normal_x = 0.0;
    double normal_y = 0.0;

    for (int bb1 = 0; bb1 < nboundary_nodes; ++bb1) {
      const int node0 = boundary_edge_list[bb1 * 2];
      const int node1 = boundary_edge_list[bb1 * 2 + 1];

      if (node0 == nn || node1 == nn) {
        const double node0_x = nodes_x0[(node0)];
        const double node0_y = nodes_y0[(node0)];
        const double node1_x = nodes_x0[(node1)];
        const double node1_y = nodes_y0[(node1)];

        normal_x += node0_y - node1_y;
        normal_y += -(node0_x - node1_x);
      }
    }

    // We are fixed if we are one of the four corners
    if ((nodes_x0[(nn)] == 0.0 || nodes_x0[(nn)] == 1.0) &&
        (nodes_y0[(nn)] == 0.0 || nodes_y0[(nn)] == 1.0)) {
      boundary_type[(bi)] = IS_CORNER;
    } else {
      boundary_type[(bi)] = IS_BOUNDARY;
    }

    const double normal_mag = sqrt(normal_x * normal_x + normal_y * normal_y);
    boundary_normal_x[(bi)] = normal_x / normal_mag;
    boundary_normal_y[(bi)] = normal_y / normal_mag;
  }
}
