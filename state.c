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
  allocate_data(&state->reduce_array, (local_nx+1)*(local_ny+1));

  state_data_init(
      local_nx, local_ny, global_nx, global_ny, x_off, y_off,
      state->rho, state->e, state->rho_old, state->P, state->Qxx, state->Qyy,
      state->x, state->p, state->rho_u, state->rho_v, state->F_x, state->F_y,
      state->uF_x, state->uF_y, state->vF_x, state->vF_y, state->reduce_array);
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

