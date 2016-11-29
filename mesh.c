#include "shared.h"
#include "mesh.h"

// Initialise the mesh describing variables
void initialise_mesh(
    Mesh* mesh)
{
  mesh->edgedx = (double*)_mm_malloc(sizeof(double)*mesh->local_nx+1, VEC_ALIGN);
  mesh->celldx = (double*)_mm_malloc(sizeof(double)*mesh->local_nx, VEC_ALIGN);
  mesh->edgedy = (double*)_mm_malloc(sizeof(double)*mesh->local_ny+1, VEC_ALIGN);
  mesh->celldy = (double*)_mm_malloc(sizeof(double)*mesh->local_ny, VEC_ALIGN);
  mesh->dt = 0.01*C_T*MAX_DT;
  mesh->dt_h = 0.01*C_T*MAX_DT;

  // Simple uniform rectilinear initialisation
  for(int ii = 0; ii < mesh->local_ny+1; ++ii) {
    mesh->edgedy[ii] = 10.0 / (mesh->global_ny);
  }
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    mesh->celldy[ii] = 10.0 / (mesh->global_ny);
  }
  for(int ii = 0; ii < mesh->local_nx+1; ++ii) {
    mesh->edgedx[ii] = 10.0 / (mesh->global_nx);
  }
  for(int ii = 0; ii < mesh->local_nx; ++ii) {
    mesh->celldx[ii] = 10.0 / (mesh->global_nx);
  }

  mesh->north_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->north_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
}

// Deallocate all of the mesh memory
void finalise_mesh(Mesh* mesh)
{
  _mm_free(mesh->edgedy);
  _mm_free(mesh->celldy);
  _mm_free(mesh->edgedx);
  _mm_free(mesh->celldx);
}


