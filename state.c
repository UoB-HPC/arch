#include "state.h"

// Initialise the state for the problem
void initialise_state(
    State* state, Mesh* mesh)
{
  // Allocate memory for all state variables
  state->rho = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->rho_old = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->P = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->e = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->rho_u = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->rho_v = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->u = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->v = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->F_x = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->F_y = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->mF_x = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->mF_y = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->slope_x0 = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->slope_y0 = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->slope_x1 = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->slope_y1 = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->Qxx = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->Qyy = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);

  // Initialise all of the memory to default values
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_nx*mesh->local_ny; ++ii) {
    state->rho[ii] = 0.0;
    state->rho_old[ii] = 0.0;
    state->P[ii] = 0.0;
    state->e[ii] = 0.0;
    state->slope_x0[ii] = 0.0;
    state->slope_y0[ii] = 0.0;
  }

#pragma omp parallel for
  for(int ii = 0; ii < (mesh->local_nx+1)*(mesh->local_ny+1); ++ii) {
    state->F_x[ii] = 0.0;
    state->F_y[ii] = 0.0;
    state->mF_x[ii] = 0.0;
    state->mF_y[ii] = 0.0;
    state->Qxx[ii] = 0.0;
    state->Qyy[ii] = 0.0;
    state->slope_x1[ii] = 0.0;
    state->slope_y1[ii] = 0.0;
    state->rho_u[ii] = 0.0;
    state->u[ii] = 0.0;
    state->v[ii] = 0.0;
    state->rho_v[ii] = 0.0;
  }

  // Initialise the entire local mesh
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
      state->rho[ii*mesh->local_nx+jj] = 0.125;
      state->e[ii*mesh->local_nx+jj] = 2.0;
    }
  }

  printf("rank %d nx %d ny %d x_off %d y_off %d global_nx %d global_ny %d\n", 
      mesh->rank, mesh->local_nx, mesh->local_ny, mesh->x_off, mesh->y_off,
      mesh->global_nx, mesh->global_ny);

  // Introduce a problem
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
#if 0
      // CENTER SQUARE TEST
      const int dist = 100;
      if(jj+mesh->x_off-PAD >= mesh->global_nx/2-dist && 
          jj+mesh->x_off-PAD < mesh->global_nx/2+dist && 
          ii+mesh->y_off-PAD >= mesh->global_ny/2-dist && 
          ii+mesh->y_off-PAD < mesh->global_ny/2+dist) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      // OFF CENTER SQUARE TEST
      const int dist = 100;
      if(jj+mesh->x_off-PAD >= mesh->global_nx/4-dist && 
          jj+mesh->x_off-PAD < mesh->global_nx/4+dist && 
          ii+mesh->y_off-PAD >= mesh->global_ny/2-dist && 
          ii+mesh->y_off-PAD < mesh->global_ny/2+dist) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
      if(jj+mesh->x_off < (mesh->global_nx/2+2*PAD)) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#if 0
      if(ii <= mesh->local_ny/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(ii > mesh->local_ny/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(jj > mesh->local_nx/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
    }
  }
}

// Deallocate all of the state memory
void finalise_state(State* state)
{
  _mm_free(state->F_x);
  _mm_free(state->F_y);
  _mm_free(state->rho);
  _mm_free(state->rho_old);
  _mm_free(state->slope_x0);
  _mm_free(state->slope_y0);
  _mm_free(state->slope_x1);
  _mm_free(state->slope_y1);
  _mm_free(state->u);
  _mm_free(state->v);
  _mm_free(state->P);
  _mm_free(state->e);
}

