#include "shared.h"
#include "mesh.h"
#include "comms.h"

// Initialise the mesh describing variables
void initialise_mesh(
    Mesh* mesh)
{
  allocate_data(&mesh->edgedx, (mesh->local_nx+1));
  allocate_data(&mesh->celldx, (mesh->local_nx+1));
  allocate_data(&mesh->edgedy, (mesh->local_ny+1));
  allocate_data(&mesh->celldy, (mesh->local_ny+1));

  mesh->dt = 0.01*(1.0-C_T)*(1.0-C_T)*MAX_DT;
  mesh->dt_h = 0.01*(1.0-C_T)*(1.0-C_T)*MAX_DT;

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

  allocate_data(&mesh->north_buffer_out, (mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->east_buffer_out, (mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->south_buffer_out, (mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->west_buffer_out, (mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->north_buffer_in, (mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->east_buffer_in, (mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->south_buffer_in, (mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  allocate_data(&mesh->west_buffer_in, (mesh->local_ny+1)*PAD*NVARS_TO_COMM);
}

// Deallocate all of the mesh memory
void finalise_mesh(Mesh* mesh)
{
  deallocate_data(mesh->edgedy);
  deallocate_data(mesh->celldy);
  deallocate_data(mesh->edgedx);
  deallocate_data(mesh->celldx);
  deallocate_data(mesh->north_buffer_out);
  deallocate_data(mesh->east_buffer_out);
  deallocate_data(mesh->south_buffer_out);
  deallocate_data(mesh->west_buffer_out);
  deallocate_data(mesh->north_buffer_in);
  deallocate_data(mesh->east_buffer_in);
  deallocate_data(mesh->south_buffer_in);
  deallocate_data(mesh->west_buffer_in);
}

