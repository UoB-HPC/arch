#pragma once

#define SIM_END 10.0    // The end time in seconds for the simulation
#define PAD 2           // The depth of halo padding, currently for all applications
#define LOAD_BALANCE 0  // Whether decomposition should attempt to load balance
#define WIDTH 10.0      // The width of the problem domain 
#define HEIGHT 10.0     // The height of the problem domain
#define DEPTH 10.0      // The depth of the problem domain
#define MAX_DT 0.04     // The maximum allowed timestep

#define NNEIGHBOURS 6   // This is max size required - for 3d

#define C_T 0.3

#ifdef __cplusplus
extern "C" {
#endif

// Contains all of the data regarding a particular mesh
typedef struct
{
  int local_nx;   // Number of cells in rank in x direction (inc. halo)
  int local_ny;   // Number of cells in rank in y direction (inc. halo)
  int local_nz;   // Number of cells in rank in z direction (inc. halo)
  int global_nx;  // Number of cells globally in x direction (exc. halo)
  int global_ny;  // Number of cells globally in x direction (exc. halo)
  int global_nz;  // Number of cells globally in x direction (exc. halo)
  int niters;     // Number of timestep iterations
  int width;      // Width of the problem domain
  int height;     // Height of the problem domain
  int depth;     // Depth of the problem domain

  // Mesh differentials
  double* edgedx;
  double* edgedy;
  double* edgedz;
  double* celldx;
  double* celldy;
  double* celldz;

  // Offset in global mesh of rank owning this local mesh
  int x_off;
  int y_off;
  int z_off;

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
  double* front_buffer_out;
  double* back_buffer_out;
  double* north_buffer_in;
  double* east_buffer_in;
  double* south_buffer_in;
  double* west_buffer_in;
  double* front_buffer_in;
  double* back_buffer_in;

  // Host copies of buffers for MPI communication
  // Note that these are only allocated when the model requires them, e.g. CUDA
  double* h_north_buffer_out;
  double* h_east_buffer_out;
  double* h_south_buffer_out;
  double* h_west_buffer_out;
  double* h_front_buffer_out;
  double* h_back_buffer_out;
  double* h_north_buffer_in;
  double* h_east_buffer_in;
  double* h_south_buffer_in;
  double* h_west_buffer_in;
  double* h_front_buffer_in;
  double* h_back_buffer_in;
} Mesh;

// Initialises the mesh
void initialise_mesh_2d(
    Mesh* mesh);
void mesh_data_init_2d(
    const int local_nx, const int local_ny, const int global_nx, 
    const int global_ny, double* edgedx, double* edgedy, 
    double* celldx, double* celldy);

void initialise_mesh_3d(
    Mesh* mesh);
void mesh_data_init_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    double* edgedx, double* edgedy, double* edgedz, 
    double* celldx, double* celldy, double* celldz);

// Finalises the mesh
void finalise_mesh(
    Mesh* mesh);

#ifdef __cplusplus
}
#endif
