#pragma once

#ifdef MPI
#include "mpi.h"
#endif

#define VEC_ALIGN 32

#define strmatch(a, b) (strcmp(a, b) == 0)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define PAD 2
#define LOAD_BALANCE 0
#define MASTER 0
#define NNEIGHBOURS 4

enum { EDGE = -1, NORTH, EAST, SOUTH, WEST };
enum { NO_PACK, PACK };

#ifdef MPI
// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
void decompose_2d_cartesian(
    const int rank, const int nranks, const int global_nx, const int global_ny,
    int* neighbours, int* local_nx, int* local_ny, int* x_off, int* y_off);
#endif

// Write out data for visualisation in visit
void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time);

// Collects all of the mesh data from the fleet of ranks and then writes to 
// visit
void write_all_ranks_to_visit(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const int x_off, const int y_off, const int rank, 
    const int nranks, double* local_arr, 
    const char* name, const int tt, const double elapsed_sim_time);

