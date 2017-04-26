
__device__ int cuda_strmatch(const char* str, const char* compare) 
{
  while(*compare != '\0') { \
    if(*compare != *str) {
      return 0;
    }
    compare++; // Move the pointers along
    str++; // Move the pointers along
  }
  return 1;
}

__global__ void zero_array(
    const int len, double* array)
{
  const int gid = threadIdx.x + blockIdx.x*blockDim.x;
  if(gid >= len) return;
  array[gid] = 0.0;
}

__global__ void zero_int_array(
    const int len, int* array)
{
  const int gid = threadIdx.x + blockIdx.x*blockDim.x;
  if(gid >= len) return;
  array[gid] = 0;
}

__global__ void initialise_problem_state(
    const int nx, const int ny, const int global_nx, 
    const int global_ny, const int x_off, const int y_off, const int nkeys,
    const int ndims, const double xpos, const double ypos, const double width, 
    const double height, const double* edgex, const double* edgey,
    double* rho, double* e, double* x, int* keys, double* values)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  if(gid >= nx*ny) return;

  double global_xpos = edgex[jj];
  double global_ypos = edgey[ii];

  // Check we are in bounds of the problem entry
  if(global_xpos >= xpos && global_ypos >= ypos && 
     global_xpos < xpos+width && global_ypos < ypos+height)
  {
    // The upper bound excludes the bounding box for the entry
    for(int kk = 0; kk < nkeys-(2*ndims); ++kk) {
      const int key = keys[kk];
      if(key == DENSITY_KEY) {
        rho[ii*nx+jj] = values[kk];
      }
      else if(key == ENERGY_KEY) {
        e[ii*nx+jj] = values[kk];
      }
      else if(key == TEMPERATURE_KEY) {
        x[ii*nx+jj] = values[kk];
      }
    }
  }
}

__global__ void mesh_data_init_dx(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int x_off, const double width, double* edgex, double* edgey, 
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= (nx+1)) return;
  edgedx[gid] = width / (global_nx);
  celldx[gid] = width / (global_nx);
  edgex[gid] = edgedx[gid] * (x_off+gid-PAD);
}

__global__ void mesh_data_init_dy(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int y_off, const double height, double* edgex, double* edgey, 
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= (ny+1)) return;
  edgedy[gid] = height / (global_ny);
  celldy[gid] = height / (global_ny);
  edgey[gid] = edgedy[gid] * (y_off+gid-PAD);
}

