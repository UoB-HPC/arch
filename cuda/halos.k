
__global__ void prepare_east(
    const int nx, const int ny, double* east_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii >= PAD && ii < ny-PAD) {
    east_buffer_out[gid] = arr[(ii*nx)+(nx-2*PAD+dd)];
  }
}

__global__ void retrieve_east(
    const int nx, const int ny, double* east_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii >= PAD && ii < ny-PAD) {
    arr[ii*nx + (nx-PAD+dd)] = east_buffer_in[gid];
  }
}

__global__ void prepare_west(
    const int nx, const int ny, double* west_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii >= PAD && ii < ny-PAD) {
    west_buffer_out[gid] = arr[(ii*nx)+(PAD+dd)];
  }
}

__global__ void retrieve_west(
    const int nx, const int ny, double* west_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii >= PAD && ii < ny-PAD) {
    arr[ii*nx + dd] = west_buffer_in[gid];
  }
}

__global__ void prepare_north(
    const int nx, const int ny, double* north_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / (nx-2*PAD);
  const int jj = gid % (nx-2*PAD);

  if(jj >= PAD && jj < nx-PAD && dd < PAD) {
    north_buffer_out[gid] = arr[(ny-2*PAD+dd)*nx+jj];
  }
}

__global__ void retrieve_north(
    const int nx, const int ny, double* north_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / (nx-2*PAD);
  const int jj = gid % (nx-2*PAD);

  if(jj >= PAD && jj < nx-PAD && dd < PAD) {
    arr[(ny-PAD+dd)*nx+jj] = north_buffer_in[gid];
  }
}

__global__ void prepare_south(
    const int nx, const int ny, double* south_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / (nx-2*PAD);
  const int jj = gid % (nx-2*PAD);

  if(jj >= PAD && jj < nx-PAD && dd < PAD) {
    south_buffer_out[gid] = arr[(PAD+dd)*nx+jj];
  }
}

__global__ void retrieve_south(
    const int nx, const int ny, double* south_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / (nx-2*PAD);
  const int jj = gid % (nx-2*PAD);

  if(jj >= PAD && jj < nx-PAD && dd < PAD) {
    arr[dd*nx + jj] = south_buffer_in[gid];
  }
}

__global__ void north_boundary(
    const int nx, const int ny, const int y_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / (nx-2*PAD);
  const int jj = gid % (nx-2*PAD);

  if(jj >= PAD && jj < nx-PAD && dd < PAD) {
    arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
  }
}

__global__ void south_boundary(
    const int nx, const int ny, const int y_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / (nx-2*PAD);
  const int jj = gid % (nx-2*PAD);

  if(jj >= PAD && jj < nx-PAD && dd < PAD) {
    arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
  }
}

__global__ void east_boundary(
    const int nx, const int ny, const int x_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii >= PAD && ii < ny-PAD) {
    arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
  }
}

__global__ void west_boundary(
    const int nx, const int ny, const int x_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii >= PAD && ii < ny-PAD) {
    arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
  }
}

