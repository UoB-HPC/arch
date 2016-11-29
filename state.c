#include <stdio.h>
#include "state.h"
#include "shared.h"

// Initialises the state variables
void initialise_state(
    const int global_nx, const int global_ny, const int local_nx, const int local_ny, 
    const int x_off, const int y_off, State* state) 
{
  // Shared state
  allocate_data(&state->rho, sizeof(double)*local_nx*local_ny);
  allocate_data(&state->e, sizeof(double)*local_nx*local_ny);

  // Currently flattening the capacity by sharing some of the state containers
  // between the different solves for different applications. This might not
  // be maintainable and/or desirable but seems like a reasonable optimisation
  // for now...
  double* temp1;
  allocate_data(&temp1, sizeof(double)*local_nx*local_ny);
  state->rho_old = temp1;
  state->Ap = temp1;

  double* temp2;
  allocate_data(&temp2, sizeof(double)*(local_nx+1)*(local_ny+1));
  state->s_x = temp2;
  state->Qxx = temp2;

  double* temp3;
  allocate_data(&temp3, sizeof(double)*(local_nx+1)*(local_ny+1));
  state->s_y = temp3;
  state->Qyy = temp3;

  double* temp4;
  allocate_data(&temp4, sizeof(double)*local_nx*local_ny);
  state->r = temp4;
  state->P = temp4;

  double* temp5;
  allocate_data(&temp5, sizeof(double)*(local_nx+1)*(local_ny+1));
  state->x = temp5;
  state->u = temp5;

  double* temp6;
  allocate_data(&temp6, sizeof(double)*(local_nx+1)*(local_ny+1));
  state->p = temp6;
  state->v = temp6;

  // Wet-specific state
  allocate_data(&state->rho_u, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->rho_v, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->F_x, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->F_y, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->uF_x, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->uF_y, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->vF_x, sizeof(double)*(local_nx+1)*(local_ny+1));
  allocate_data(&state->vF_y, sizeof(double)*(local_nx+1)*(local_ny+1));

  // Initialise all of the state to 0, but is this best for NUMA?
#pragma omp parallel for
  for(int ii = 0; ii < local_nx*local_ny; ++ii) {
    state->rho[ii] = 0.0;
    state->e[ii] = 0.0;
    temp1[ii] = 0.0;
    temp4[ii] = 0.0;
  }

#pragma omp parallel for
  for(int ii = 0; ii < (local_nx+1)*(local_ny+1); ++ii) {
    temp2[ii] = 0.0;
    temp3[ii] = 0.0;
    temp5[ii] = 0.0;
    temp6[ii] = 0.0;
    state->rho_u[ii] = 0.0;
    state->rho_v[ii] = 0.0;
    state->F_x[ii] = 0.0;
    state->F_y[ii] = 0.0;
    state->uF_x[ii] = 0.0;
    state->uF_y[ii] = 0.0;
    state->vF_x[ii] = 0.0;
    state->vF_y[ii] = 0.0;
  }

  // TODO: Improve what follows, make it a somewhat more general problem 
  // selection mechanism for some important stock problems

#if 0
  // WET STATE INITIALISATION

  // Initialise a default state for the energy and density on the mesh
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {
      state->rho[ii*local_nx+jj] = 0.125;
      state->e[ii*local_nx+jj] = 2.0;
    }
  }

  // Introduce a problem
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {
#if 0
      // CENTER SQUARE TEST
      const int dist = 100;
      if(jj+x_off-PAD >= global_nx/2-dist && 
          jj+x_off-PAD < global_nx/2+dist && 
          ii+y_off-PAD >= global_ny/2-dist && 
          ii+y_off-PAD < global_ny/2+dist) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      // OFF CENTER SQUARE TEST
      const int dist = 100;
      if(jj+x_off-PAD >= global_nx/4-dist && 
          jj+x_off-PAD < global_nx/4+dist && 
          ii+y_off-PAD >= global_ny/2-dist && 
          ii+y_off-PAD < global_ny/2+dist) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
      if(jj+x_off < (global_nx/2+2*PAD)) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#if 0
      if(ii <= local_ny/2) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(ii > local_ny/2) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(jj > local_nx/2) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
    }
  }
#endif // if 0

  // HOT STATE INITIALISATION
  // Set the initial state
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < local_nx; ++jj) {
      const int index = ii*local_nx+jj;
      state->rho[index] = 0.1;
      state->e[index] = state->rho[index] * 0.1;
    }
  }

  // Crooked pipe problem
#pragma omp parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < local_nx; ++jj) {
      const int index = ii*local_nx+jj;
      const int ioff = ii+y_off;
      const int joff = jj+x_off;

      // Box problem
      if(ioff > 7*(global_ny+2*PAD)/8 || ioff <= (global_ny+2*PAD)/8 ||
          joff > 7*(global_nx+2*PAD)/8 || joff <= (global_nx+2*PAD)/8) {
        state->rho[index] = 100.0;
        state->x[index] = state->rho[index]*0.1;
        state->e[index] = state->x[index];
      }
    }
  }
}

// Deallocate all of the state memory
void finalise_state(State* state)
{
  deallocate_data(state->rho);
  deallocate_data(state->e); 
  deallocate_data(state->rho_u);
  deallocate_data(state->rho_v);
  deallocate_data(state->F_x);
  deallocate_data(state->F_y);
  deallocate_data(state->uF_x);
  deallocate_data(state->uF_y);
  deallocate_data(state->vF_x);
  deallocate_data(state->vF_y);

  // Only free one of the paired states
  deallocate_data(state->Ap);
  deallocate_data(state->s_x);
  deallocate_data(state->s_y);
  deallocate_data(state->r);
  deallocate_data(state->x);
  deallocate_data(state->p);
}

