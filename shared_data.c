#include <stdio.h>
#include "shared_data.h"
#include "shared.h"

// Initialises the shared_data variables
void initialise_shared_data_2d(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const int x_off, const int y_off, const int mesh_width, 
    const int mesh_height, const char* problem_def_filename, const double* edgex, 
    const double* edgey, SharedData* shared_data) 
{
  const int ndims = 2;

  // Shared shared_data
  allocate_data(&shared_data->rho, local_nx*local_ny);
  allocate_data(&shared_data->e, local_nx*local_ny);

  // Currently flattening the capacity by sharing some of the shared_data containers
  // between the different solves for different applications. This might not
  // be maintainable and/or desirable but seems like a reasonable optimisation
  // for now...
  allocate_data(&shared_data->rho_old, local_nx*local_ny);
  shared_data->Ap = shared_data->rho_old;

  allocate_data(&shared_data->s_x, (local_nx+1)*(local_ny+1));
  shared_data->Qxx = shared_data->s_x;

  allocate_data(&shared_data->s_y, (local_nx+1)*(local_ny+1));
  shared_data->Qyy = shared_data->s_y;

  allocate_data(&shared_data->r, local_nx*local_ny);
  shared_data->P = shared_data->r;

  allocate_data(&shared_data->x, (local_nx+1)*(local_ny+1));
  shared_data->u = shared_data->x;

  allocate_data(&shared_data->p, (local_nx+1)*(local_ny+1));
  shared_data->v = shared_data->p;

  set_problem_2d(
      global_nx, global_ny, local_nx, local_ny, x_off, y_off, mesh_width, 
      mesh_height, edgex, edgey, ndims, problem_def_filename, shared_data->rho, 
      shared_data->e, shared_data->x);
}

// Initialises the shared_data variables
void initialise_shared_data_3d(
    const int global_nx, const int global_ny, const int global_nz, 
    const int local_nx, const int local_ny, const int local_nz,
    const int x_off, const int y_off, int z_off, SharedData* shared_data) 
{
  // Shared shared_data
  allocate_data(&shared_data->rho, local_nx*local_ny*local_nz);
  allocate_data(&shared_data->e, local_nx*local_ny*local_nz);

  // TODO: Place this initialisation behaviour in each application to save 
  // some capacity

  // Currently flattening the capacity by sharing some of the shared_data containers
  // between the different solves for different applications. This might not
  // be maintainable and/or desirable but seems like a reasonable optimisation
  // for now...
  allocate_data(&shared_data->rho_old, local_nx*local_ny*local_nz);
  shared_data->Ap = shared_data->rho_old;

  allocate_data(&shared_data->s_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  shared_data->Qxx = shared_data->s_x;

  allocate_data(&shared_data->s_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
  shared_data->Qyy = shared_data->s_y;

  allocate_data(&shared_data->s_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
  shared_data->Qzz = shared_data->s_z;

  allocate_data(&shared_data->r, local_nx*local_ny*local_nz);
  shared_data->P = shared_data->r;

  allocate_data(&shared_data->x, (local_nx+1)*(local_ny+1)*(local_nz+1));
  shared_data->u = shared_data->x;

  allocate_data(&shared_data->p, (local_nx+1)*(local_ny+1)*(local_nz+1));
  shared_data->v = shared_data->p;

  allocate_data(&shared_data->reduce_array, (local_nx+1)*(local_ny+1));

  set_problem_3d(
      local_nx, local_ny, local_nz, global_nx, global_ny, global_nz, 
      x_off, y_off, z_off, shared_data->rho, shared_data->e, shared_data->x);
}

// Deallocate all of the shared_data memory
void finalise_shared_data(SharedData* shared_data)
{
  deallocate_data(shared_data->rho);
  deallocate_data(shared_data->e); 

  // Only free one of the paired shared_datas
  deallocate_data(shared_data->Ap);
  deallocate_data(shared_data->s_x);
  deallocate_data(shared_data->s_y);
  deallocate_data(shared_data->r);
  deallocate_data(shared_data->x);
  deallocate_data(shared_data->p);
}

