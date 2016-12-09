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
#endif

#ifdef MPI
  if(pack) {
    // Pack east and west
    if(neighbours[EAST] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          east_buffer_out[(ii-PAD)*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
        }
      }

      sync_data(PAD*ny, &east_buffer_out, &east_buffer_out, RECV);
      non_block_send(east_buffer_out, (ny-2*PAD)*PAD, neighbours[EAST], 2, nmessages++);
      non_block_recv(east_buffer_in, (ny-2*PAD)*PAD, neighbours[EAST], 3, nmessages++);
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          west_buffer_out[(ii-PAD)*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
        }
      }

      sync_data(PAD*ny, &west_buffer_out, &west_buffer_out, RECV);
      non_block_send(west_buffer_out, (ny-2*PAD)*PAD, neighbours[WEST], 3, nmessages++);
      non_block_recv(west_buffer_in, (ny-2*PAD)*PAD, neighbours[WEST], 2, nmessages++);
    }

    // Pack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          north_buffer_out[dd*(nx-2*PAD)+(jj-PAD)] = arr[(ny-2*PAD+dd)*nx+jj];
        }
      }

      sync_data(nx*PAD, &north_buffer_out, &north_buffer_out, RECV);
      non_block_send(north_buffer_out, (nx-2*PAD)*PAD, neighbours[NORTH], 1, nmessages++);
      non_block_recv(north_buffer_in, (nx-2*PAD)*PAD, neighbours[NORTH], 0, nmessages++);
    }


    if(neighbours[SOUTH] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          south_buffer_out[dd*(nx-2*PAD)+(jj-PAD)] = arr[(PAD+dd)*nx+jj];
        }
      }

      sync_data(nx*PAD, &south_buffer_out, &south_buffer_out, RECV);
      non_block_send(south_buffer_out, (nx-2*PAD)*PAD, neighbours[SOUTH], 0, nmessages++);
      non_block_recv(south_buffer_in, (nx-2*PAD)*PAD, neighbours[SOUTH], 1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if(neighbours[WEST] != EDGE) {
      sync_data(PAD*ny, &west_buffer_in, &west_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + dd] = west_buffer_in[(ii-PAD)*PAD+dd];
        }
      }
    }

    if(neighbours[EAST] != EDGE) {
      sync_data(PAD*ny, &east_buffer_in, &east_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + (nx-PAD+dd)] = east_buffer_in[(ii-PAD)*PAD+dd];
        }
      }
    }

    // Unpack north and south
    if(neighbours[NORTH] != EDGE) {
      sync_data(nx*PAD, &north_buffer_in, &north_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          arr[(ny-PAD+dd)*nx+jj] = north_buffer_in[dd*(nx-2*PAD)+(jj-PAD)];
        }
      }
    }

    if(neighbours[SOUTH] != EDGE) {
      sync_data(nx*PAD, &south_buffer_in, &south_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          arr[dd*nx + jj] = south_buffer_in[dd*(nx-2*PAD)+(jj-PAD)];
        }
      }
    }
  }
#endif

  // Perform the boundary reflections, potentially with the data updated from neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if(neighbours[NORTH] == EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = PAD; jj < nx-PAD; ++jj) {
        arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
      }
    }
  }
  // reflect at the south
  if(neighbours[SOUTH] == EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = PAD; jj < nx-PAD; ++jj) {
        arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
      }
    }
  }
  // reflect at the east
  if(neighbours[EAST] == EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
    for(int ii = PAD; ii < ny-PAD; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
      }
    }
  }
  if(neighbours[WEST] == EDGE) {
    // reflect at the west
#pragma omp target teams distribute parallel for collapse(2)
    for(int ii = PAD; ii < ny-PAD; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
      }
    }
  }
  STOP_PROFILING(&comms_profile, __func__);
}

