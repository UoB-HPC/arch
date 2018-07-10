
// Matches a string on the device
__device__ int cuda_strmatch(const char* str, const char* compare) {
  while (*compare != '\0') {
    if (*compare != *str) {
      return 0;
    }
    compare++; // Move the pointers along
    str++;     // Move the pointers along
  }
  return 1;
}

// Generic routine to zero an array
template <typename T> __global__ void zero_array(const int len, T* array) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= len)
    return;
  array[gid] = 0.0;
}

// Problem initialisation on target device
__global__ void initialise_problem_state(
    const int nx, const int ny, const int nkeys, const int ndims,
    const double xpos, const double ypos, const double width,
    const double height, const double* edgex, const double* edgey, double* density,
    double* energy, double* temperature, int* keys, double* values) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  if (gid >= nx * ny)
    return;

  double global_xpos = edgex[jj];
  double global_ypos = edgey[ii];

  // Check we are in bounds of the problem entry
  if (global_xpos >= xpos && global_ypos >= ypos &&
      global_xpos < xpos + width && global_ypos < ypos + height) {
    // The upper bound excludes the bounding box for the entry
    for (int kk = 0; kk < nkeys - (2 * ndims); ++kk) {
      const int key = keys[kk];
      if (key == DENSITY_KEY) {
        density[ii * nx + jj] = values[kk];
      } else if (key == ENERGY_KEY) {
        energy[ii * nx + jj] = values[kk];
      } else if (key == TEMPERATURE_KEY) {
        temperature[ii * nx + jj] = values[kk];
      }
    }
  }
}


// Problem initialisation on target device
__global__ void initialise_problem_state_3d(
    const int nx, const int ny, const int nz, const int nkeys, const int ndims,
    const double xpos, const double ypos, const double zpos, const double width,
    const double height, const double depth, const double* edgex, 
    const double* edgey, const double* edgez, double* density,
    double* energy, double* temperature, int* keys, double* values) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;

  const int ii = (gid / (nx*ny));
  const int jj = (gid / (nx)) % ny;
  const int kk = (gid % (nx));

  if (gid >= nx * ny * nz)
    return;

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
      const int key = keys[nn];
      if (key == DENSITY_KEY) {
        density[gid] = values[nn];
      } else if (key == ENERGY_KEY) {
        energy[gid] = values[nn];
      } else if (key == TEMPERATURE_KEY) {
        temperature[gid] = values[nn];
      }
    }
  }
}

// Initialises mesh data for the x dimension
__global__ void mesh_data_init_dx(const int nx, const int ny,
                                  const int global_nx, const int global_ny,
                                  const int pad, const int x_off,
                                  const double width, double* edgex,
                                  double* edgey, double* edgedx, double* edgedy,
                                  double* celldx, double* celldy) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= (nx + 1))
    return;
  edgedx[gid] = width / (global_nx);
  celldx[gid] = width / (global_nx);
  edgex[gid] = edgedx[gid] * (x_off + gid - pad);
}

// Initialises mesh data for the y dimension
__global__ void mesh_data_init_dy(const int nx, const int ny,
                                  const int global_nx, const int global_ny,
                                  const int pad, const int y_off,
                                  const double height, double* edgex,
                                  double* edgey, double* edgedx, double* edgedy,
                                  double* celldx, double* celldy) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= (ny + 1))
    return;
  edgedy[gid] = height / (global_ny);
  celldy[gid] = height / (global_ny);
  edgey[gid] = edgedy[gid] * (y_off + gid - pad);
}

// Initialises mesh data for the y dimension
__global__ void mesh_data_init_dz(const int nz, const int global_nz,
                                  const int pad, const int z_off,
                                  const double depth, double* edgez,
                                  double* edgedz, double* celldz) {

  const int ii = threadIdx.x + blockIdx.x * blockDim.x;
  if (ii >= (nz + 1))
    return;

  edgedz[ii] = depth / (global_nz);
  edgez[ii] = edgedz[ii] * (z_off + ii - pad);
  celldz[ii] = depth / (global_nz);
}

