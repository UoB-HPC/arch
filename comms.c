#include <stdio.h>
#include <math.h>
#include "shared.h"
#include "comms.h"

// Initialise the communications, potentially invoking MPI
void initialise_comms(
    int argc, char** argv, Mesh* mesh)
{
  for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
    mesh->neighbours[ii] = EDGE;
  }

#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mesh->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mesh->nranks);

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
    const double local_val, MPI_Op op)
{
  double global_val = local_val;
  START_PROFILING(&compute_profile);
  MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  STOP_PROFILING(&compute_profile, "communications");
  return global_val;
}
#endif

// Reduces the value across all ranks and returns minimum result
double reduce_all_min(const double local_val)
{
  double global_val = local_val;
#ifdef MPI
  mpi_all_reduce(local_val, MPI_SUM);
#endif
  return global_val;
}

// Reduces the value across all ranks and returns the sum
double reduce_all_sum(const double local_val)
{
  double global_val = local_val;
#ifdef MPI
  mpi_all_reduce(local_val, MPI_SUM);
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
  float min_ratio = 0.0f;

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
    if((found_even <= is_even) && (min_ratio == 0.0f || potential_ratio < min_ratio)) {
      min_ratio = potential_ratio;
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
    const int x_pad_req = (global_nx < (off + (ranks_x-xx)*x_floor));
    *local_nx = x_pad_req ? x_floor+1 : x_floor;
    off += *local_nx;
  }
  off = 0;
  const int y_rank = (rank/ranks_x);
  for(int yy = 0; yy <= y_rank; ++yy) {
    *y_off = off;
    const int y_floor = global_ny/ranks_y;
    const int y_pad_req = (global_ny < (off + (ranks_y-yy)*y_floor));
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

// Finalise the communications
void finalise_comms()
{
#ifdef MPI
  MPI_Finalize();
#endif
}

