#include "shared.h"
#include "mesh.h"

// Initialise the mesh describing variables
void initialise_mesh(
    Mesh* mesh)
{
  allocate_data(&mesh->edgedx, (mesh->local_nx+1));
  allocate_data(&mesh->celldx, (mesh->local_nx+1));
  allocate_data(&mesh->edgedy, (mesh->local_ny+1));
  allocate_data(&mesh->celldy, (mesh->local_ny+1));

  mesh->dt = 0.1*C_T*MAX_DT;
  mesh->dt_h = 0.1*C_T*MAX_DT;

  mesh_data_init(
      mesh->local_nx, mesh->local_ny, mesh->global_nx, mesh->global_ny, 
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  allocate_data(&mesh->north_buffer_out, (mesh->local_nx+1)*PAD);
  allocate_data(&mesh->east_buffer_out, (mesh->local_ny+1)*PAD);
  allocate_data(&mesh->south_buffer_out, (mesh->local_nx+1)*PAD);
  allocate_data(&mesh->west_buffer_out, (mesh->local_ny+1)*PAD);
  allocate_data(&mesh->north_buffer_in, (mesh->local_nx+1)*PAD);
  allocate_data(&mesh->east_buffer_in, (mesh->local_ny+1)*PAD);
  allocate_data(&mesh->south_buffer_in, (mesh->local_nx+1)*PAD);
  allocate_data(&mesh->west_buffer_in, (mesh->local_ny+1)*PAD);
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

