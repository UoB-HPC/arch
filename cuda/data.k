
__global__ void mesh_data_init_dx(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgex, double* edgey, double* edgedx, double* edgedy, 
    double* celldx, double* celldy)
{
  set_cuda_indices(0);
  if(ind0 >= (nx+1)) return;
  edgedx[ind0] = WIDTH / (global_nx);
  celldx[ind0] = WIDTH / (global_nx);
  edgex[ind0] = edgedx[ind0] * ind0;
}

__global__ void mesh_data_init_dy(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgex, double* edgey, double* edgedx, double* edgedy, 
    double* celldx, double* celldy)
{
  set_cuda_indices(0);
  if(ind0 >= (ny+1)) return;
  edgedy[ind0] = HEIGHT / (global_ny);
  celldy[ind0] = HEIGHT / (global_ny);
  edgey[ind0] = edgedy[ind0] * ind0;
}

__global__ void zero_array(
    const int len, double* array)
{
  const int gid = threadIdx.x + blockIdx.x*blockDim.x;
  if(gid > len) return;
  array[gid] = 0.0;
}

__global__ void initialise_default_state(
    const int nx, const int ny, 
    double* rho, double* e, double* rho_old, double* x)
{
  set_cuda_indices(0);
  if(ind0 >= nx*ny) return;

  rho[ind0] = 0.125;
  e[ind0] = 2.0;
  x[ind0] = rho[ind0]*0.1;
}

__global__ void initialise_problem_state(
    const int nx, const int ny, const int global_nx, 
    const int global_ny, const int x_off, const int y_off, 
    double* rho, double* e, double* rho_old, double* x)
{
  set_cuda_indices(0);
  if(ind0 >= nx*ny) return;

  // CENTER SQUARE TEST
  if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
      jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
      ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
      ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
    rho[ind0] = 1.0;
    e[ind0] = 2.5;
    x[ind0] = rho[ii*nx+jj]*0.1;
  }
}
