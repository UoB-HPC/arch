#pragma once 

#include "mesh.h"
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

  // TODO: MAKE IT SO THAT STATE IS LOCAL TO THE APPLICATIONS???
  
// Contains all of the state information for the solver
typedef struct
{
  // Shared state (share data)
  double* rho;      // Density
  double* e;        // Energy

  // Paired state (share capacity)
  double* Ap;       // HOT: Coefficient matrix A, by conjugate vector p
  double* rho_old;  // WET: Density at beginning of timestep

  double* s_x;      // HOT: Coefficients in x direction
  double* Qxx;      // WET: Artificial viscous term in x direction

  double* s_y;      // HOT: Coefficients in y direction
  double* Qyy;      // WET: Artificial viscous term in y direction

  double* s_z;      // HOT: Coefficients in z direction
  double* Qzz;      // WET: Artificial viscous term in z direction

  double* r;        // HOT: The residual vector
  double* P;        // WET: The pressure

  double* x;        // HOT: The solution vector (new energy)
  double* u;        // WET: The velocity in the x direction

  double* p;        // HOT: The conjugate vector
  double* v;        // WET: The velocity in the y direction

  // Wet-specific state
  double* rho_u;    // Momentum in the x direction
  double* rho_v;    // Momentum in the y direction
  double* rho_w;    // Momentum in the z direction

  double* F_x;      // Mass flux in the x direction
  double* F_y;      // Mass flux in the y direction
  double* F_z;      // Mass flux in the z direction

  double* uF_x;     // Momentum in the x direction flux in the x direction 
  double* uF_y;     // Momentum in the x direction flux in the y direction
  double* uF_z;     // Momentum in the x direction flux in the z direction

  double* vF_x;     // Momentum in the y direction flux in the x direction
  double* vF_y;     // Momentum in the y direction flux in the y direction
  double* vF_z;     // Momentum in the y direction flux in the z direction

  double* wF_x;     // Momentum in the z direction flux in the x direction
  double* wF_y;     // Momentum in the z direction flux in the y direction
  double* wF_z;     // Momentum in the z direction flux in the z direction

  double* reduce_array;

  double complex* ce;

} State;

// Initialises the state variables for two dimensional applications
void initialise_state_2d(
    const int global_nx, const int global_ny, const int local_nx, const int local_ny, 
    const int x_off, const int y_off, State* state);
void state_data_init_2d(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off, double* rho, double* e, double* x);

// Initialiases the state variables for three dimensional applications
void initialise_state_3d(
    const int global_nx, const int global_ny, const int global_nz, 
    const int local_nx, const int local_ny, const int local_nz,
    const int x_off, const int y_off, const int z_off, State* state);
void state_data_init_3d(
    const int local_nx, const int local_ny, const int local_nz, 
    const int global_nx, const int global_ny, const int global_nz,
    const int x_off, const int y_off, const int z_off,
    double* rho, double* e, double* x);

// Deallocate all of the state memory
void finalise_state(
    State* state);

#ifdef __cplusplus
}
#endif

