#pragma once 

#include "state.h"
#include "mesh.h"

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

  double* r;        // HOT: The residual vector
  double* P;        // WET: The pressure

  double* x;        // HOT: The solution vector (new energy)
  double* u;        // WET: The velocity in the x direction

  double* p;        // HOT: The conjugate vector
  double* v;        // WET: The velocity in the y direction

  // Wet-specific state
  double* rho_u;    // Momentum in the x direction
  double* rho_v;    // Momentum in the y direction
  double* F_x;      // Mass flux in the x direction
  double* F_y;      // Mass flux in the y direction
  double* uF_x;     // Momentum in the x direction flux in the x direction 
  double* uF_y;     // Momentum in the x direction flux in the y direction
  double* vF_x;     // Momentum in the y direction flux in the x direction
  double* vF_y;     // Momentum in the y direction flux in the y direction

} State;

// Initialises the state variables
void initialise_state(
    const int global_nx, const int global_ny, const int local_nx, const int local_ny, 
    const int x_off, const int y_off, State* state);

void state_data_init(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y);

// Deallocate all of the state memory
void finalise_state(
    State* state);

