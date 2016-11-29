#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "shared.h"

struct Profile compute_profile = {0};
struct Profile comms_profile = {0};

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
double mpi_all_min(const double local_val)
{
  double global_val = local_val;
#ifdef MPI
  mpi_all_reduce(local_val, MPI_SUM);
#endif
  return global_val;
}

// Reduces the value across all ranks and returns the sum
double mpi_all_sum(const double local_val)
{
  double global_val = local_val;
#ifdef MPI
  mpi_all_reduce(local_val, MPI_SUM);
#endif
  return global_val;
}

// This is currently duplicated from the hydro package
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

// Write out data for visualisation in visit
void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time)
{
#ifdef ENABLE_VISIT_DUMPS
  char bovname[256];
  char datname[256];
  sprintf(bovname, "%s%d.bov", name, step);
  sprintf(datname, "%s%d.dat", name, step);

  FILE* bovfp = fopen(bovname, "w");

  if(!bovfp) {
    printf("Could not open file %s\n", bovname);
    exit(1);
  }

  fprintf(bovfp, "TIME: %.4f\n", time);
  fprintf(bovfp, "DATA_FILE: %s\n", datname);
  fprintf(bovfp, "DATA_SIZE: %d %d 1\n", nx, ny);
  fprintf(bovfp, "DATA_FORMAT: DOUBLE\n");
  fprintf(bovfp, "VARIABLE: density");
  fprintf(bovfp, "DATA_ENDIAN: LITTLE\n");
  fprintf(bovfp, "CENTERING: zone\n");

#ifdef MPI
  fprintf(bovfp, "BRICK_ORIGIN: %f %f 0.\n", (float)x_off, (float)y_off);
#else
  fprintf(bovfp, "BRICK_ORIGIN: 0. 0. 0.\n");
#endif

  fprintf(bovfp, "BRICK_SIZE: %d %d 1\n", nx, ny);
  fclose(bovfp);

  FILE* datfp = fopen(datname, "wb");
  if(!datfp) {
    printf("Could not open file %s\n", datname);
    exit(1);
  }

  fwrite(data, sizeof(data), nx*ny, datfp);
  fclose(datfp);
#endif
}

// TODO: Fix this method - shouldn't be necessary to bring the data back from
// all of the ranks, this is over the top
void write_all_ranks_to_visit(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const int x_off, const int y_off, const int rank, 
    const int nranks, double* local_arr, 
    const char* name, const int tt, const double elapsed_sim_time)
{
  // If MPI is enabled need to collect the data from all 
#if defined(MPI) && defined(ENABLE_VISIT_DUMPS)
  double* global_arr;
  double** remote_data;

  if(rank == MASTER) {
    global_arr = (double*)malloc(sizeof(double)*global_nx*global_ny);
    remote_data = (double**)malloc(sizeof(double*)*nranks);
    remote_data[MASTER] = local_arr;
  }

  for(int ii = 0; ii < nranks; ++ii) {
    int dims[4];
    dims[0] = local_nx;
    dims[1] = local_ny;
    dims[2] = x_off;
    dims[3] = y_off;

    if(rank == MASTER) {
      if(ii > MASTER) {
        MPI_Recv(&dims, 4, MPI_INT, ii, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        remote_data[ii] = (double*)malloc(sizeof(double)*dims[0]*dims[1]);
        MPI_Recv(
            remote_data[ii], dims[0]*dims[1], MPI_DOUBLE, ii, 1, 
            MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }

      for(int jj = PAD; jj < dims[1]-PAD; ++jj) {
        for(int kk = PAD; kk < dims[0]-PAD; ++kk) {
          global_arr[(dims[3]+(jj-PAD))*global_nx+((kk-PAD)+dims[2])] =
            remote_data[ii][jj*dims[0]+kk];
        }
      }
    }
    else if(ii == rank) {
      MPI_Send(&dims, 4, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
      MPI_Send(local_arr, dims[0]*dims[1], MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#else
  double* global_arr = local_arr;
#endif

  if(rank == MASTER) {
    write_to_visit(global_nx, global_ny, 0, 0, global_arr, name, tt, elapsed_sim_time);
  }
}

