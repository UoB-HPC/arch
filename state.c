#include <stdio.h>
#include "state.h"
#include "shared.h"

// Initialises the state variables
void initialise_state(
    const int global_nx, const int global_ny, const int local_nx, const int local_ny, 
    const int x_off, const int y_off, State* state) 
{
  // Shared state
  allocate_data(&state->rho, local_nx*local_ny);
  allocate_data(&state->e, local_nx*local_ny);

  // Currently flattening the capacity by sharing some of the state containers
  // between the different solves for different applications. This might not
  // be maintainable and/or desirable but seems like a reasonable optimisation
  // for now...
  allocate_data(&state->rho_old, local_nx*local_ny);
  state->Ap = state->rho_old;

  allocate_data(&state->s_x, (local_nx+1)*(local_ny+1));
  state->Qxx = state->s_x;

  allocate_data(&state->s_y, (local_nx+1)*(local_ny+1));
  state->Qyy = state->s_y;

  allocate_data(&state->r, local_nx*local_ny);
  state->P = state->r;

  allocate_data(&state->x, (local_nx+1)*(local_ny+1));
  state->u = state->x;

  allocate_data(&state->p, (local_nx+1)*(local_ny+1));
  state->v = state->p;

  // Wet-specific state
  allocate_data(&state->rho_u, (local_nx+1)*(local_ny+1));
  allocate_data(&state->rho_v, (local_nx+1)*(local_ny+1));
  allocate_data(&state->F_x, (local_nx+1)*(local_ny+1));
  allocate_data(&state->F_y, (local_nx+1)*(local_ny+1));
  allocate_data(&state->uF_x, (local_nx+1)*(local_ny+1));
  allocate_data(&state->uF_y, (local_nx+1)*(local_ny+1));
  allocate_data(&state->vF_x, (local_nx+1)*(local_ny+1));
  allocate_data(&state->vF_y, (local_nx+1)*(local_ny+1));

  // Initialise all of the state to 0, but is this best for NUMA?
#pragma omp parallel for
  for(int ii = 0; ii < local_nx*local_ny; ++ii) {
    state->rho[ii] = 0.0;
    state->e[ii] = 0.0;
    state->rho_old[ii] = 0.0;
    state->P[ii] = 0.0;
  }

#pragma omp parallel for
  for(int ii = 0; ii < (local_nx+1)*(local_ny+1); ++ii) {
    state->Qxx[ii] = 0.0;
    state->Qyy[ii] = 0.0;
    state->x[ii] = 0.0;
    state->p[ii] = 0.0;
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

  // WET STATE INITIALISATION
  // Initialise a default state for the energy and density on the mesh
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {
      state->rho[ii*local_nx+jj] = 0.125;
      state->e[ii*local_nx+jj] = 2.0;
      state->x[ii*local_nx+jj] = state->rho[ii*local_nx+jj]*0.1;
    }
  }

  // Introduce a problem
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {
      // CENTER SQUARE TEST
      if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
          jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
          ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
          ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
        state->x[ii*local_nx+jj] = state->rho[ii*local_nx+jj]*0.1;
      }

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

#if 0
      if(jj+x_off < ((global_nx+2*PAD)/2)) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0

#if 0
      if(ii+y_off < (global_ny+2*PAD)/2) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0

#if 0
      if(ii+y_off > (global_ny+2*PAD)/2) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0

#if 0
      if(jj+x_off > (global_nx+2*PAD)/2) {
        state->rho[ii*local_nx+jj] = 1.0;
        state->e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
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

