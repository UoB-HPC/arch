
__global__ void mesh_data_init_dx(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  edgedx[threadIdx.x+blockIdx.x*blockDim.x] = WIDTH / (global_nx);
  celldx[threadIdx.x+blockIdx.x*blockDim.x] = WIDTH / (global_nx);
}

__global__ void mesh_data_init_dy(
    const int nx, const int ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  edgedy[gid] = HEIGHT / (global_ny);
  celldy[gid] = HEIGHT / (global_ny);
}

__global__ void zero_cell_arrays(
    const int nx, const int ny, const int x_off, const int y_off, 
    double* rho, double* e, double* rho_old, double* P)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= nx*ny) return;

  rho[gid] = 0.0;
  e[gid] = 0.0;
  rho_old[gid] = 0.0;
  P[gid] = 0.0;
}

__global__ void zero_edge_arrays(
    const int nx, const int ny, double* Qxx, double* Qyy, double* x, 
    double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= (nx+1)*(ny+1)) return;

  Qxx[gid] = 0.0;
  Qyy[gid] = 0.0;
  x[gid] = 0.0;
  p[gid] = 0.0;
  rho_u[gid] = 0.0;
  rho_v[gid] = 0.0;
  F_x[gid] = 0.0;
  F_y[gid] = 0.0;
  uF_x[gid] = 0.0;
  uF_y[gid] = 0.0;
  vF_x[gid] = 0.0;
  vF_y[gid] = 0.0;
}

__global__ void initialise_default_state(
    const int nx, const int ny, 
    double* rho, double* e, double* rho_old, double* x)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= nx*ny) return;

  rho[gid] = 0.125;
  e[gid] = 2.0;
  x[gid] = rho[gid]*0.1;
}

__global__ void initialise_problem_state(
    const int nx, const int ny, const int global_nx, 
    const int global_ny, const int x_off, const int y_off, 
    double* rho, double* e, double* rho_old, double* x)
{
  set_cuda_indices(0);

  if(gid >= nx*ny) return;

  // CENTER SQUARE TEST
  if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
      jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
      ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
      ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
    rho[gid] = 1.0;
    e[gid] = 2.5;
    x[gid] = rho[ii*nx+jj]*0.1;
  }

#if 0
  // OFF CENTER SQUARE TEST
  const int dist = 100;
  if(jj+x_off-PAD >= global_nx/4-dist && 
      jj+x_off-PAD < global_nx/4+dist && 
      ii+y_off-PAD >= global_ny/2-dist && 
      ii+y_off-PAD < global_ny/2+dist) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
    x[ii*nx+jj] = rho[ii*nx+jj]*e[ii*nx+jj];
  }
#endif // if 0

#if 0
  if(jj+x_off < ((global_nx+2*PAD)/2)) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
    x[ii*nx+jj] = rho[ii*nx+jj]*0.1;
  }
#endif // if 0

#if 0
  if(ii+y_off < (global_ny+2*PAD)/2) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
  }
#endif // if 0

#if 0
  if(ii+y_off > (global_ny+2*PAD)/2) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
  }
#endif // if 0

#if 0
  if(jj+x_off > (global_nx+2*PAD)/2) {
    rho[ii*nx+jj] = 1.0;
    e[ii*nx+jj] = 2.5;
  }
#endif // if 0
}

