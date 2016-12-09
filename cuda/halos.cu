#include "../comms.h"
#include "../mesh.h"
#include "halos.k"
#include "shared.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int prepare)
{
  START_PROFILING(&comms_profile);

  int* neighbours = mesh->neighbours;

#ifdef MPI
  int nmessages = 0;

  if(prepare) {
    // prepare east and west
    if(neighbours[EAST] != EDGE) {
      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      prepare_east<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->east_buffer_out, arr);

      sync_data(ny*PAD, &mesh->east_buffer_out, &mesh->h_east_buffer_out, RECV);
      non_block_send(mesh->h_east_buffer_out, ny*PAD, neighbours[EAST], 2, nmessages++);
      non_block_recv(mesh->h_east_buffer_in, ny*PAD, neighbours[EAST], 3, nmessages++);
    }

    if(neighbours[WEST] != EDGE) {
      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      prepare_west<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->west_buffer_out, arr);

      sync_data(ny*PAD, &mesh->west_buffer_out, &mesh->h_west_buffer_out, RECV);
      non_block_send(mesh->h_west_buffer_out, ny*PAD, neighbours[WEST], 3, nmessages++);
      non_block_recv(mesh->h_west_buffer_in, ny*PAD, neighbours[WEST], 2, nmessages++);
    }

    // prepare north and south
    if(neighbours[NORTH] != EDGE) {
      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      prepare_north<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->north_buffer_out, arr);

      sync_data(nx*PAD, &mesh->north_buffer_out, &mesh->h_north_buffer_out, RECV);
      non_block_send(mesh->h_north_buffer_out, nx*PAD, neighbours[NORTH], 1, nmessages++);
      non_block_recv(mesh->h_north_buffer_in, nx*PAD, neighbours[NORTH], 0, nmessages++);
    }

    if(neighbours[SOUTH] != EDGE) {
      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      prepare_south<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->south_buffer_out, arr);

      sync_data(nx*PAD, &mesh->south_buffer_out, &mesh->h_south_buffer_out, RECV);
      non_block_send(mesh->h_south_buffer_out, nx*PAD, neighbours[SOUTH], 0, nmessages++);
      non_block_recv(mesh->h_south_buffer_in, nx*PAD, neighbours[SOUTH], 1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unprepare east and west
    if(neighbours[WEST] != EDGE) {
      sync_data(ny*PAD, &mesh->h_west_buffer_in, &mesh->west_buffer_in, SEND);

      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      retrieve_west<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->west_buffer_in, arr);
    }

    if(neighbours[EAST] != EDGE) {
      sync_data(ny*PAD, &mesh->h_east_buffer_in, &mesh->east_buffer_in, SEND);

      int nblocks = ceil(ny*PAD/(double)NTHREADS);
      retrieve_east<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->east_buffer_in, arr);
    }

    // Unprepare north and south
    if(neighbours[NORTH] != EDGE) {
      sync_data(nx*PAD, &mesh->h_north_buffer_in, &mesh->north_buffer_in, SEND);

      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      retrieve_north<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->north_buffer_in, arr);
    }

    if(neighbours[SOUTH] != EDGE) {
      sync_data(nx*PAD, &mesh->h_south_buffer_in, &mesh->south_buffer_in, SEND);

      int nblocks = ceil(nx*PAD/(double)NTHREADS);
      retrieve_south<<<nblocks, NTHREADS>>>(
          nx, ny, mesh->south_buffer_in, arr);
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

  gpu_check(cudaDeviceSynchronize());
}

