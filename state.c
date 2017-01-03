#include <stdio.h>
#include "state.h"
#include "shared.h"

// Initialises the state variables
void initialise_state_2d(
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

  state_data_init_2d(
      local_nx, local_ny, global_nx, global_ny, x_off, y_off,
      state->rho, state->e, state->x);
}

// Initialises the state variables
void initialise_state_3d(
    const int global_nx, const int global_ny, const int global_nz, 
    const int local_nx, const int local_ny, const int local_nz,
    const int x_off, const int y_off, int z_off, State* state) 
{
  // Shared state
  allocate_data(&state->rho, local_nx*local_ny*local_nz);
  allocate_data(&state->e, local_nx*local_ny*local_nz);

  // TODO: Place this initialisation behaviour in each application to save 
  // some capacity

  // Currently flattening the capacity by sharing some of the state containers
  // between the different solves for different applications. This might not
  // be maintainable and/or desirable but seems like a reasonable optimisation
  // for now...
  allocate_data(&state->rho_old, local_nx*local_ny*local_nz);
  state->Ap = state->rho_old;

  allocate_data(&state->s_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  state->Qxx = state->s_x;

  allocate_data(&state->s_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
  state->Qyy = state->s_y;

  allocate_data(&state->s_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
  state->Qzz = state->s_z;

  allocate_data(&state->r, local_nx*local_ny*local_nz);
  state->P = state->r;

  allocate_data(&state->x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  state->u = state->x;

  allocate_data(&state->p, (local_nx+1)*(local_ny+1)*(local_nz+1));
  state->v = state->p;

  // Wet-specific state
  allocate_data(&state->rho_u, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->rho_v, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->rho_w, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->F_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->F_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->F_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->uF_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->uF_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->uF_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->vF_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->vF_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->vF_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->wF_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->wF_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->wF_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
  allocate_data(&state->reduce_array, (local_nx+1)*(local_ny+1)*(local_nz+1));

  state_data_init_3d(
      local_nx, local_ny, local_nz, global_nx, global_ny, global_nz, 
      x_off, y_off, z_off, state->rho, state->e, state->x);
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

