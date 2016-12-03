#include <stdio.h>
#include <math.h>
#include "shared.h"
#include "comms.h"

void init_mpi(
    int argc, char** argv, int* rank, int* nranks)
{
#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, nranks);
#endif
}

// Initialise the communications, potentially invoking MPI
void initialise_comms(
    Mesh* mesh)
{
  for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
    mesh->neighbours[ii] = EDGE;
  }

#ifdef MPI
  decompose_2d_cartesian(
      mesh->rank, mesh->nranks, mesh->global_nx, mesh->global_ny, 
      mesh->neighbours, &mesh->local_nx, &mesh->local_ny, &mesh->x_off, &mesh->y_off);

  // Add on the halo padding to the local mesh
  mesh->local_nx += 2*PAD;
  mesh->local_ny += 2*PAD;
#endif 

  if(mesh->rank == MASTER)
    printf("Problem dimensions %dx%d for %d iterations.\n", 
        mesh->global_nx, mesh->global_ny, mesh->niters);
}

#ifdef MPI
static inline double mpi_all_reduce(
    double local_val, MPI_Op op)
{
  double global_val = local_val;
  START_PROFILING(&compute_profile);
  MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  STOP_PROFILING(&compute_profile, "communications");
  return global_val;
}
#endif

// Reduces the value across all ranks and returns minimum result
double reduce_all_min(double local_val)
{
  double global_val = local_val;
#ifdef MPI
  global_val = mpi_all_reduce(local_val, MPI_MIN);
#endif
  return global_val;
}

// Reduces the value across all ranks and returns the sum
double reduce_all_sum(double local_val)
{
  double global_val = local_val;
#ifdef MPI
  global_val = mpi_all_reduce(local_val, MPI_SUM);
#endif
  return global_val;
}

// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
void decompose_2d_cartesian(
    const int rank, const int nranks, const int global_nx, const int global_ny,
    int* neighbours, int* local_nx, int* local_ny, int* x_off, int* y_off) 
{
  int ranks_x = 0;
  int ranks_y = 0;
  int found_even = 0;
  float mratio = 0.0f;

  // Determine decomposition that minimises perimeter to area ratio
  for(int ff = 1; ff <= sqrt(nranks); ++ff) {
    if(nranks % ff) continue;
    // If load balance is preferred then prioritise even split over ratio
    // Test if this split evenly decomposes into the mesh
    const int even_split_ff_x = 
      (global_nx % ff == 0 && global_ny % (nranks/ff) == 0);
    const int even_split_ff_y = 
      (global_nx % (nranks/ff) == 0 && global_ny % ff == 0);
    const int new_ranks_x = even_split_ff_x ? ff : nranks/ff;
    const int new_ranks_y = even_split_ff_x ? nranks/ff : ff;
    const int is_even = even_split_ff_x || even_split_ff_y;
    found_even |= (LOAD_BALANCE && is_even);

    const float potential_ratio = 
      (2*(new_ranks_x+new_ranks_y))/(float)(new_ranks_x*new_ranks_y);

    // Update if we minimise the ratio further, only if we don't care about load
    // balancing or have found an even split
    if((found_even <= is_even) && (mratio == 0.0f || potential_ratio < mratio)) {
      mratio = potential_ratio;
      // If we didn't find even split, prefer longer mesh edge on x dimension
      ranks_x = (!found_even && new_ranks_x > new_ranks_y) ? new_ranks_y : new_ranks_x;
      ranks_y = (!found_even && new_ranks_x > new_ranks_y) ? new_ranks_x : new_ranks_y;
    }
  }

  // Calculate the offsets up until our rank, and then fetch rank dimensions
  int off = 0;
  const int x_rank = (rank%ranks_x);
  for(int xx = 0; xx <= x_rank; ++xx) {
    *x_off = off;
    const int x_floor = global_nx/ranks_x;
    const int x_pad_req = (global_nx != (off + (ranks_x-xx)*x_floor));
    *local_nx = x_pad_req ? x_floor+1 : x_floor;
    off += *local_nx;
  }
  off = 0;
  const int y_rank = (rank/ranks_x);
  for(int yy = 0; yy <= y_rank; ++yy) {
    *y_off = off;
    const int y_floor = global_ny/ranks_y;
    const int y_pad_req = (global_ny != (off + (ranks_y-yy)*y_floor));
    *local_ny = y_pad_req ? y_floor+1 : y_floor;
    off += *local_ny;
  }

  // Calculate the surrounding ranks
  neighbours[NORTH] = (y_rank < ranks_y-1) ? rank+ranks_x : EDGE;
  neighbours[EAST] = (x_rank < ranks_x-1) ? rank+1 : EDGE;
  neighbours[SOUTH] = (y_rank > 0) ? rank-ranks_x : EDGE;
  neighbours[WEST] = (x_rank > 0) ? rank-1 : EDGE;

  printf("rank %d neighbours %d %d %d %d\n",
      rank, neighbours[NORTH], neighbours[EAST], 
      neighbours[SOUTH], neighbours[WEST]);
}

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int pack)
{
  START_PROFILING(&comms_profile);

  int* neighbours = mesh->neighbours;
#ifdef MPI
  int nmessages = 0;
  MPI_Request req[NNEIGHBOURS];
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
        mesh->west_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
      }
    }

    MPI_Isend(mesh->west_buffer_out, ny*PAD, MPI_DOUBLE,
        neighbours[WEST], 3, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(mesh->west_buffer_in, ny*PAD, MPI_DOUBLE, 
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
        mesh->east_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
      }
    }

    MPI_Isend(mesh->east_buffer_out, ny*PAD, MPI_DOUBLE, 
        neighbours[EAST], 2, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(mesh->east_buffer_in, ny*PAD, MPI_DOUBLE,
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
        mesh->north_buffer_out[dd*nx+jj] = arr[(ny-2*PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(mesh->north_buffer_out, nx*PAD, MPI_DOUBLE, 
        neighbours[NORTH], 1, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(mesh->north_buffer_in, nx*PAD, MPI_DOUBLE,
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
        mesh->south_buffer_out[dd*nx+jj] = arr[(PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(mesh->south_buffer_out, nx*PAD, MPI_DOUBLE, 
        neighbours[SOUTH], 0, MPI_COMM_WORLD, &req[nmessages++]);
    MPI_Irecv(mesh->south_buffer_in, nx*PAD, MPI_DOUBLE,
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
          arr[(ny-PAD+dd)*nx+jj] = mesh->north_buffer_in[dd*nx+jj];
        }
      }
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[dd*nx + jj] = mesh->south_buffer_in[dd*nx+jj];
        }
      }
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + dd] = mesh->west_buffer_in[ii*PAD+dd];
        }
      }
    }

    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[ii*PAD+dd];
        }
      }
    }
  }
#endif

  STOP_PROFILING(&comms_profile, __func__);
}

// Finalise the communications
void finalise_comms()
{
#ifdef MPI
  MPI_Finalize();
#endif
}

