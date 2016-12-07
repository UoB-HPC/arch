#include "../comms.h"
#include "../mesh.h"
#include "halos.k"
#include "config.h"

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
  MPI_Request req[2*NNEIGHBOURS];
#endif

  int* neighbours = mesh->neighbours;

#ifdef MPI
  if(fill) {
    // fill east and west
    if(neighbours[EAST] != EDGE) {
      int nthreads_per_block = ceil(ny*PAD/(double)NBLOCKS);
      fill_east<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, east_buffer_out, arr);

      sync_data(PAD*ny, east_buffer_out, east_buffer_out, RECV);

      MPI_Isend(east_buffer_out, (ny-2*PAD)*PAD, MPI_DOUBLE, 
          neighbours[EAST], 2, MPI_COMM_WORLD, &req[nmessages++]);
      MPI_Irecv(east_buffer_in, (ny-2*PAD)*PAD, MPI_DOUBLE,
          neighbours[EAST], 3, MPI_COMM_WORLD, &req[nmessages++]);
    }

    if(neighbours[WEST] != EDGE) {
      int nthreads_per_block = ceil(ny*PAD/(double)NBLOCKS);
      fill_west<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, west_buffer_out, arr);

      sync_data(PAD*ny, west_buffer_out, west_buffer_out, RECV);

      MPI_Isend(west_buffer_out, (ny-2*PAD)*PAD, MPI_DOUBLE,
          neighbours[WEST], 3, MPI_COMM_WORLD, &req[nmessages++]);
      MPI_Irecv(west_buffer_in, (ny-2*PAD)*PAD, MPI_DOUBLE, 
          neighbours[WEST], 2, MPI_COMM_WORLD, &req[nmessages++]);
    }

    // fill north and south
    if(neighbours[NORTH] != EDGE) {
      int nthreads_per_block = ceil(nx*PAD/(double)NBLOCKS);
      fill_north<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, north_buffer_out, arr);

      sync_data(nx*PAD, north_buffer_out, north_buffer_out, RECV);

      MPI_Isend(north_buffer_out, (nx-2*PAD)*PAD, MPI_DOUBLE, 
          neighbours[NORTH], 1, MPI_COMM_WORLD, &req[nmessages++]);
      MPI_Irecv(north_buffer_in, (nx-2*PAD)*PAD, MPI_DOUBLE,
          neighbours[NORTH], 0, MPI_COMM_WORLD, &req[nmessages++]);
    }

    if(neighbours[SOUTH] != EDGE) {
      int nthreads_per_block = ceil(nx*PAD/(double)NBLOCKS);
      fill_south<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, south_buffer_out, arr);

      sync_data(nx*PAD, south_buffer_out, south_buffer_out, RECV);

      MPI_Isend(south_buffer_out, (nx-2*PAD)*PAD, MPI_DOUBLE, 
          neighbours[SOUTH], 0, MPI_COMM_WORLD, &req[nmessages++]);
      MPI_Irecv(south_buffer_in, (nx-2*PAD)*PAD, MPI_DOUBLE,
          neighbours[SOUTH], 1, MPI_COMM_WORLD, &req[nmessages++]);
    }

    MPI_Waitall(nmessages, req, MPI_STATUSES_IGNORE);

    // Unfill east and west
    if(neighbours[WEST] != EDGE) {
      sync_data(PAD*ny, west_buffer_in, west_buffer_in, SEND);

      int nthreads_per_block = ceil(ny*PAD/(double)NBLOCKS);
      retrieve_west<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, west_buffer_in, arr);
    }

    if(neighbours[EAST] != EDGE) {
      sync_data(PAD*ny, east_buffer_in, east_buffer_in, SEND);

      int nthreads_per_block = ceil(ny*PAD/(double)NBLOCKS);
      retrieve_east<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, east_buffer_in, arr);
    }

    // Unfill north and south
    if(neighbours[NORTH] != EDGE) {
      sync_data(nx*PAD, north_buffer_in, north_buffer_in, SEND);

      int nthreads_per_block = ceil(nx*PAD/(double)NBLOCKS);
      retrieve_north<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, north_buffer_in, arr);
    }

    if(neighbours[SOUTH] != EDGE) {
      sync_data(nx*PAD, south_buffer_in, south_buffer_in, SEND);

      int nthreads_per_block = ceil(nx*PAD/(double)NBLOCKS);
      retrieve_south<<<nthreads_per_block, NBLOCKS>>>(
          nx, ny, south_buffer_in, arr);
    }
  }
#endif

  // Perform the boundary reflections, potentially with the data updated from neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if(neighbours[NORTH] == EDGE) {
    int nthreads_per_block = ceil(nx*PAD/(double)NBLOCKS);
    north_boundary<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, y_inversion_coeff, arr);
  }
  // reflect at the south
  if(neighbours[SOUTH] == EDGE) {
    int nthreads_per_block = ceil(nx*PAD/(double)NBLOCKS);
    south_boundary<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, y_inversion_coeff, arr);
  }
  // reflect at the east
  if(neighbours[EAST] == EDGE) {
    int nthreads_per_block = ceil(ny*PAD/(double)NBLOCKS);
    east_boundary<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, x_inversion_coeff, arr);
  }
  // reflect at the west
  if(neighbours[WEST] == EDGE) {
    int nthreads_per_block = ceil(ny*PAD/(double)NBLOCKS);
    west_boundary<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, x_inversion_coeff, arr);
  }
  STOP_PROFILING(&comms_profile, __func__);
}

