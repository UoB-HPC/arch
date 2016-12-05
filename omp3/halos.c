#include "../comms.h"
#include "../mesh.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int pack)
{
  START_PROFILING(&comms_profile);

  double* north_buffer_out = mesh->north_buffer_out;
  double* east_buffer_out = mesh->east_buffer_out;
  double* south_buffer_out = mesh->south_buffer_out;
  double* west_buffer_out = mesh->west_buffer_out;
  double* north_buffer_in = mesh->north_buffer_in;
  double* east_buffer_in = mesh->east_buffer_in;
  double* south_buffer_in = mesh->south_buffer_in;
  double* west_buffer_in = mesh->west_buffer_in;
  int* neighbours = mesh->neighbours;

#ifdef MPI
  int nmessages = 0;
  MPI_Request req[2*NNEIGHBOURS];
#endif

  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;

  if(neighbours[WEST] == EDGE) {
    // reflect at the west
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        west_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
      }
    }

    MPI_Isend(west_buffer_out, ny*PAD, MPI_DOUBLE,
        neighbours[WEST], 3, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(west_buffer_in, ny*PAD, MPI_DOUBLE, 
        neighbours[WEST], 2, MPI_COMM_WORLD, &req[nmessages++]);
  }
#endif

  // Reflect at the east
  if(neighbours[EAST] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        east_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
      }
    }

    MPI_Isend(east_buffer_out, ny*PAD, MPI_DOUBLE, 
        neighbours[EAST], 2, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(east_buffer_in, ny*PAD, MPI_DOUBLE,
        neighbours[EAST], 3, MPI_COMM_WORLD, &req[nmessages++]);
  }
#endif

  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if(neighbours[NORTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        north_buffer_out[dd*nx+jj] = arr[(ny-2*PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(north_buffer_out, nx*PAD, MPI_DOUBLE, 
        neighbours[NORTH], 1, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(north_buffer_in, nx*PAD, MPI_DOUBLE,
        neighbours[NORTH], 0, MPI_COMM_WORLD, &req[nmessages++]);
  }
#endif

  // reflect at the south
  if(neighbours[SOUTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
      }
    }
  }
#ifdef MPI
  else if (pack) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        south_buffer_out[dd*nx+jj] = arr[(PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(south_buffer_out, nx*PAD, MPI_DOUBLE, 
        neighbours[SOUTH], 0, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(south_buffer_in, nx*PAD, MPI_DOUBLE,
        neighbours[SOUTH], 1, MPI_COMM_WORLD, &req[nmessages++]);
  }
#endif

  // Unpack the buffers
#ifdef MPI
  if(pack) {
    MPI_Waitall(nmessages, req, MPI_STATUSES_IGNORE);

    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[(ny-PAD+dd)*nx+jj] = north_buffer_in[dd*nx+jj];
        }
      }
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[dd*nx + jj] = south_buffer_in[dd*nx+jj];
        }
      }
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + dd] = west_buffer_in[ii*PAD+dd];
        }
      }
    }

    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + (nx-PAD+dd)] = east_buffer_in[ii*PAD+dd];
        }
      }
    }
  }
#endif

  STOP_PROFILING(&comms_profile, __func__);
}

