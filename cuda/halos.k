
__global__ void prepare_east(const int nx, const int ny, const int pad,
                             double* east_buffer_out, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int ii = gid / pad;
  const int dd = gid % pad;

  if (gid < ny * pad) {
    east_buffer_out[gid] = arr[(ii * nx) + (nx - 2 * pad + dd)];
  }
}

__global__ void retrieve_east(const int nx, const int ny, const int pad,
                              double* east_buffer_in, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int ii = gid / pad;
  const int dd = gid % pad;

  if (gid < ny * pad) {
    arr[ii * nx + (nx - pad + dd)] = east_buffer_in[gid];
  }
}

__global__ void prepare_west(const int nx, const int ny, const int pad,
                             double* west_buffer_out, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int ii = gid / pad;
  const int dd = gid % pad;

  if (gid < ny * pad) {
    west_buffer_out[gid] = arr[(ii * nx) + (pad + dd)];
  }
}

__global__ void retrieve_west(const int nx, const int ny, const int pad,
                              double* west_buffer_in, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int ii = gid / pad;
  const int dd = gid % pad;

  if (gid < ny * pad) {
    arr[ii * nx + dd] = west_buffer_in[gid];
  }
}

__global__ void prepare_north(const int nx, const int ny, const int pad,
                              double* north_buffer_out, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if (gid < nx * pad) {
    north_buffer_out[gid] = arr[(ny - 2 * pad + dd) * nx + jj];
  }
}

__global__ void retrieve_north(const int nx, const int ny, const int pad,
                               double* north_buffer_in, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if (gid < nx * pad) {
    arr[(ny - pad + dd) * nx + jj] = north_buffer_in[gid];
  }
}

__global__ void prepare_south(const int nx, const int ny, const int pad,
                              double* south_buffer_out, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if (gid < nx * pad) {
    south_buffer_out[gid] = arr[(pad + dd) * nx + jj];
  }
}

__global__ void retrieve_south(const int nx, const int ny, const int pad,
                               double* south_buffer_in, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if (gid < nx * pad) {
    arr[dd * nx + jj] = south_buffer_in[gid];
  }
}

__global__ void north_boundary(const int nx, const int ny, const int pad,
                               const int y_inversion_coeff, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if (gid < nx * pad) {
    arr[(ny - pad + dd) * nx + jj] =
        y_inversion_coeff * arr[(ny - 1 - pad - dd) * nx + jj];
  }
}

__global__ void south_boundary(const int nx, const int ny, const int pad,
                               const int y_inversion_coeff, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int dd = gid / nx;
  const int jj = gid % nx;

  if (gid < nx * pad) {
    arr[(pad - 1 - dd) * nx + jj] =
        y_inversion_coeff * arr[(pad + dd) * nx + jj];
  }
}

__global__ void east_boundary(const int nx, const int ny, const int pad,
                              const int x_inversion_coeff, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int ii = gid / pad;
  const int dd = gid % pad;

  if (gid < ny * pad) {
    arr[ii * nx + (nx - pad + dd)] =
        x_inversion_coeff * arr[ii * nx + (nx - 1 - pad - dd)];
  }
}

__global__ void west_boundary(const int nx, const int ny, const int pad,
                              const int x_inversion_coeff, double* arr) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int ii = gid / pad;
  const int dd = gid % pad;

  if (gid < ny * pad) {
    arr[ii * nx + (pad - 1 - dd)] =
        x_inversion_coeff * arr[ii * nx + (pad + dd)];
  }
}
