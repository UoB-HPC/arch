#include "../comms.h"
#include "../mesh.h"
#include "halos.k"
#include "shared.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int fill)
{
  START_PROFILING(&comms_profile);

#ifdef MPI
  double* north_buffer_out = mesh->north_buffer_out;
  double* east_buffer_out = mesh->east_buffer_out;
  double* south_buffer_out = mesh->south_buffer_out;
  double* west_buffer_out = mesh->west_buffer_out;
  double* north_buffer_in = mesh->north_buffer_in;
  double* east_buffer_in = mesh->east_buffer_in;
  double* south_buffer_in = mesh->south_buffer_in;
  double* west_buffer_in = mesh->west_buffer_in;

  int nmessages = 0;
#endif

  int* neighbours = mesh->neighbours;

#ifdef MPI
  if(fill) {
    // fill east and west
    if(neighbours[EAST] != EDGE) {
      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      fill_east<<<nblocks, NTHREADS>>>(
          nx, ny, east_buffer_out, arr);

      sync_data(PAD*ny, &east_buffer_out, &east_buffer_out, RECV);
      non_block_send(east_buffer_out, (ny-2*PAD)*PAD, neighbours[EAST], 2, nmessages++);
      non_block_recv(east_buffer_in, (ny-2*PAD)*PAD, neighbours[EAST], 3, nmessages++);
    }

    if(neighbours[WEST] != EDGE) {
      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      fill_west<<<nblocks, NTHREADS>>>(
          nx, ny, west_buffer_out, arr);

      sync_data(PAD*ny, &west_buffer_out, &west_buffer_out, RECV);
      non_block_send(west_buffer_out, (ny-2*PAD)*PAD, neighbours[WEST], 3, nmessages++);
      non_block_recv(west_buffer_in, (ny-2*PAD)*PAD, neighbours[WEST], 2, nmessages++);
    }

    // fill north and south
    if(neighbours[NORTH] != EDGE) {
      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      fill_north<<<nblocks, NTHREADS>>>(
          nx, ny, north_buffer_out, arr);

      sync_data(nx*PAD, &north_buffer_out, &north_buffer_out, RECV);
      non_block_send(north_buffer_out, (nx-2*PAD)*PAD, neighbours[NORTH], 1, nmessages++);
      non_block_recv(north_buffer_in, (nx-2*PAD)*PAD, neighbours[NORTH], 0, nmessages++);
    }

    if(neighbours[SOUTH] != EDGE) {
      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      fill_south<<<nblocks, NTHREADS>>>(
          nx, ny, south_buffer_out, arr);

      sync_data(nx*PAD, &south_buffer_out, &south_buffer_out, RECV);
      non_block_send(south_buffer_out, (nx-2*PAD)*PAD, neighbours[SOUTH], 0, nmessages++);
      non_block_recv(south_buffer_in, (nx-2*PAD)*PAD, neighbours[SOUTH], 1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unfill east and west
    if(neighbours[WEST] != EDGE) {
      sync_data(PAD*ny, &west_buffer_in, &west_buffer_in, SEND);

      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      retrieve_west<<<nblocks, NTHREADS>>>(
          nx, ny, west_buffer_in, arr);
    }

    if(neighbours[EAST] != EDGE) {
      sync_data(PAD*ny, &east_buffer_in, &east_buffer_in, SEND);

      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      retrieve_east<<<nblocks, NTHREADS>>>(
          nx, ny, east_buffer_in, arr);
    }

    // Unfill north and south
    if(neighbours[NORTH] != EDGE) {
      sync_data(nx*PAD, &north_buffer_in, &north_buffer_in, SEND);

      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      retrieve_north<<<nblocks, NTHREADS>>>(
          nx, ny, north_buffer_in, arr);
    }

    if(neighbours[SOUTH] != EDGE) {
      sync_data(nx*PAD, &south_buffer_in, &south_buffer_in, SEND);

      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      retrieve_south<<<nblocks, NTHREADS>>>(
          nx, ny, south_buffer_in, arr);
    }
  }
#endif

  // Perform the boundary reflections, potentially with the data updated from neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if(neighbours[NORTH] == EDGE) {
    int nblocks = ceil(nx*PAD/(double)NTHREADS);
    north_boundary<<<nblocks, NTHREADS>>>(
        nx, ny, y_inversion_coeff, arr);
  }
  // reflect at the south
  if(neighbours[SOUTH] == EDGE) {
    int nblocks = ceil(nx*PAD/(double)NTHREADS);
    south_boundary<<<nblocks, NTHREADS>>>(
        nx, ny, y_inversion_coeff, arr);
  }
  // reflect at the east
  if(neighbours[EAST] == EDGE) {
    int nblocks = ceil(ny*PAD/(double)NTHREADS);
    east_boundary<<<nblocks, NTHREADS>>>(
        nx, ny, x_inversion_coeff, arr);
  }
  // reflect at the west
  if(neighbours[WEST] == EDGE) {
    int nblocks = ceil(ny*PAD/(double)NTHREADS);
    west_boundary<<<nblocks, NTHREADS>>>(
        nx, ny, x_inversion_coeff, arr);
  }
  STOP_PROFILING(&comms_profile, __func__);
}

