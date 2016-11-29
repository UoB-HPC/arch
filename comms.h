#pragma once 

#include "mesh.h"
#include "shared.h"

#ifdef MPI
#include "mpi.h"
#endif

#define MASTER 0             // The master rank for MPI
#define NVARS_TO_COMM 4      // This is just the max of HOT and WET 

enum { NO_PACK, PACK }; // Whether a buffer should be packed and communicated
enum { EDGE = -1, NORTH, EAST, SOUTH, WEST }; // Directional enumeration

// Initialise the communications, potentially invoking MPI
void initialise_comms(
    int argc, char** argv, Mesh* mesh);

#ifdef MPI
// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
void decompose_2d_cartesian(
    const int rank, const int nranks, const int global_nx, const int global_ny,
    int* neighbours, int* local_nx, int* local_ny, int* x_off, int* y_off);
#endif

// Reduces the value across all ranks and returns the sum
double reduce_all_sum(
    const double local_val);

// Reduces the value across all ranks and returns minimum result
double reduce_all_min(
    const double local_val);

// Finalise the communications
void finalise_comms();

