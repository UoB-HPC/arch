#pragma once

#include "profiler.h"

#ifdef MPI
#include "mpi.h"
#endif

#define ENABLE_VISIT_DUMPS 1 // Enables visit dumps in the descendent applications
#define VEC_ALIGN 32         // The vector alignment to be used by memory allocators
#define PAD 2                // The depth of halo padding, currently for all applications
#define LOAD_BALANCE 0       // Whether decomposition should attempt to load balance
#define MASTER 0             // The master rank for MPI
#define NNEIGHBOURS 4        // The number of neighbours, as expected from 2d

// Helper macros
#define strmatch(a, b) (strcmp(a, b) == 0)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define absmin(a, b) ((fabs(a) < fabs(b)) ? (a) : (b))
#define minmod(a, b) (((a*b) > 0.0) ? (absmin(a, b)) : (0.0))

// Global profile hooks
struct Profile compute_profile;
struct Profile comms_profile;

enum { EDGE = -1, NORTH, EAST, SOUTH, WEST }; // Directional enumeration
enum { NO_PACK, PACK }; // Whether a buffer should be packed and communicated

// Contains all of the data regarding a particular mesh
typedef struct
{
  int local_nx;   // Number of cells in rank in x direction (inc. halo)
  int local_ny;   // Number of cells in rank in y direction (inc. halo)
  int global_nx;  // Number of cells globally in x direction (exc. halo)
  int global_ny;  // Number of cells globally in x direction (exc. halo)
  int niters;     // Number of timestep iterations
  int width;      // Width of the problem domain
  int height;     // Height of the problem domain

  // Mesh differentials
  double* edgedx;
  double* edgedy;
  double* celldx;
  double* celldy;

  // Offset in global mesh of rank owning this local mesh
  int x_off;
  int y_off;

  // Timesteps
  double dt;
  double dt_h;

  int rank;       // Rank that owns this mesh object
  int nranks;     // Number of ranks that exist
  int neighbours[NNEIGHBOURS]; // List of neighbours

  // Buffers for MPI communication
  double* north_buffer_out;
  double* east_buffer_out;
  double* south_buffer_out;
  double* west_buffer_out;
  double* north_buffer_in;
  double* east_buffer_in;
  double* south_buffer_in;
  double* west_buffer_in;

} Mesh;

#ifdef MPI
// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
void decompose_2d_cartesian(
    const int rank, const int nranks, const int global_nx, const int global_ny,
    int* neighbours, int* local_nx, int* local_ny, int* x_off, int* y_off);
#endif

// Reduces the value across all ranks and returns the sum
double mpi_all_sum(const double local_val);

// Reduces the value across all ranks and returns minimum result
double mpi_all_min(const double local_val);

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

