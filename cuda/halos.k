
__global__ void fill_east(
    const int nx, const int ny, double* east_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii < PAD || ii >= ny-PAD) return;

  east_buffer_out[(ii-PAD)*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
}

__global__ void retrieve_east(
    const int nx, const int ny, double* east_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii < PAD || ii >= ny-PAD) return;

  arr[ii*nx + (nx-PAD+dd)] = east_buffer_in[(ii-PAD)*PAD+dd];
}

__global__ void fill_west(
    const int nx, const int ny, double* west_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii < PAD || ii >= ny-PAD) return;

  west_buffer_out[(ii-PAD)*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
}

__global__ void retrieve_west(
    const int nx, const int ny, double* west_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii < PAD || ii >= ny-PAD) return;

  arr[ii*nx + dd] = west_buffer_in[(ii-PAD)*PAD+dd];
}

__global__ void fill_north(
    const int nx, const int ny, double* north_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if(jj < PAD || jj >= nx-PAD) return;

  north_buffer_out[dd*(nx-2*PAD)+(jj-PAD)] = arr[(ny-2*PAD+dd)*nx+jj];
}

__global__ void retrieve_north(
    const int nx, const int ny, double* north_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if(jj < PAD || jj >= nx-PAD) return;

  arr[(ny-PAD+dd)*nx+jj] = north_buffer_in[dd*(nx-2*PAD)+(jj-PAD)];
}

__global__ void fill_south(
    const int nx, const int ny, double* south_buffer_out, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if(jj < PAD || jj >= nx-PAD) return;

  south_buffer_out[dd*(nx-2*PAD)+(jj-PAD)] = arr[(PAD+dd)*nx+jj];
}

__global__ void retrieve_south(
    const int nx, const int ny, double* south_buffer_in, double* arr)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if(jj < PAD || jj >= nx-PAD) return;

  arr[dd*nx + jj] = south_buffer_in[dd*(nx-2*PAD)+(jj-PAD)];
}

__global__ void north_boundary(
    const int nx, const int ny, const int y_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if(jj < PAD || jj >= nx-PAD) return;

  arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
}

__global__ void south_boundary(
    const int nx, const int ny, const int y_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if(jj < PAD || jj >= nx-PAD) return;

  arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
}

__global__ void east_boundary(
    const int nx, const int ny, const int x_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii < PAD || ii >= ny-PAD) return;

  arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
}

__global__ void west_boundary(
    const int nx, const int ny, const int x_inversion_coeff, double* arr) 
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int ii = gid / PAD;
  const int dd = gid % PAD;

  if(ii < PAD || ii >= ny-PAD) return;

  arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
}

