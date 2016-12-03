#pragma once

#define SIM_END 10.0    // The end time in seconds for the simulation
#define VEC_ALIGN 32    // The vector alignment to be used by memory allocators
#define PAD 2           // The depth of halo padding, currently for all applications
#define LOAD_BALANCE 0  // Whether decomposition should attempt to load balance
#define WIDTH 10.0      // The width of the problem domain 
#define HEIGHT 10.0     // The height of the problem domain
#define MAX_DT 0.04     // The maximum allowed timestep
#define NNEIGHBOURS 4   // The number of neighbours, as expected from 2d

#define C_T 0.5

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

// Initialises the mesh
void initialise_mesh(
    Mesh* mesh);

// Finalises the mesh
void finalise_mesh(
    Mesh* mesh);

