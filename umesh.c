#include "umesh.h"
#include "params.h"
#include "shared.h"
#include <assert.h>
#include <stdlib.h>

// Converts an ordinary structured mesh into an unstructured equivalent
size_t convert_mesh_to_umesh_3d(UnstructuredMesh* umesh, Mesh* mesh) {
  const int nx = mesh->local_nx;
  const int ny = mesh->local_ny;
  const int nz = mesh->local_nz;
  umesh->nnodes_by_cell = NNODES_BY_CELL;
  umesh->nnodes_by_node = NNODES_BY_NODE;

  umesh->nboundary_nodes =
      2 * (nx + 1) * (ny + 1) + 2 * (ny-1) * (nz) + 2 * (nz-1) * (nx);
  umesh->nnodes = (nx + 1) * (ny + 1) * (nz + 1);
  umesh->ncells = (nx * ny * nz);
  umesh->nfaces = nx * ny * (nz + 1) + (nx * (ny + 1) + (nx + 1) * ny) * nz;

  // Allocate the data structures that we now know the sizes of
  size_t allocated = allocate_data(&umesh->cell_centroids_x, umesh->ncells);
  allocated += allocate_data(&umesh->cell_centroids_y, umesh->ncells);
  allocated += allocate_data(&umesh->cell_centroids_z, umesh->ncells);
  allocated +=
      allocate_int_data(&umesh->nodes_to_cells_offsets, umesh->nnodes + 1);
  allocated +=
      allocate_int_data(&umesh->cells_to_nodes_offsets, umesh->ncells + 1);
  allocated +=
      allocate_int_data(&umesh->nodes_to_nodes_offsets, umesh->nnodes + 1);
  allocated += allocate_int_data(&umesh->cells_to_nodes,
                                 umesh->ncells * NNODES_BY_CELL);
  allocated += allocate_int_data(&umesh->nodes_to_nodes,
                                 umesh->nnodes * NNODES_BY_NODE);
  allocated += allocate_int_data(&umesh->nodes_to_cells,
                                 umesh->nnodes * NCELLS_BY_NODE);
  allocated += allocate_data(&umesh->nodes_x0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_z0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_x1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_z1, umesh->nnodes);
  allocated +=
      allocate_int_data(&umesh->faces_to_nodes_offsets, umesh->nfaces + 1);
  allocated +=
      allocate_int_data(&umesh->faces_to_nodes, umesh->nfaces * NNODES_BY_FACE);
  allocated += allocate_int_data(&umesh->faces_cclockwise_cell, umesh->nfaces);
  allocated +=
      allocate_int_data(&umesh->cells_to_faces_offsets, umesh->ncells + 1);
  allocated += allocate_int_data(&umesh->cells_to_faces,
                                 umesh->ncells * NFACES_BY_CELL);
  allocated +=
      allocate_int_data(&umesh->nodes_to_faces, umesh->nnodes * NFACES_BY_NODE);
  allocated +=
      allocate_int_data(&umesh->nodes_to_faces_offsets, umesh->nnodes + 1);
  allocated += allocate_int_data(&umesh->faces_to_cells0, umesh->nfaces);
  allocated += allocate_int_data(&umesh->faces_to_cells1, umesh->nfaces);

  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nnodes);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nnodes);
  allocated += allocate_data(&umesh->boundary_normal_z, umesh->nnodes);
  allocated += allocate_int_data(&umesh->boundary_index, umesh->nnodes);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nnodes);

  // Initialises the list of nodes to cells
  init_nodes_to_cells_3d(nx, ny, nz, mesh, umesh);

  // Initialises the list of nodes to nodes
  init_nodes_to_nodes_3d(nx, ny, nz, umesh);

  // Initialises the offsets between faces and nodes
  init_faces_to_nodes_offsets_3d(umesh);

  // Initialises the connectivity between faces and nodes
  init_faces_to_nodes_3d(nx, ny, nz, umesh);

  // Initialises the offsets between cells and faces
  init_cells_to_faces_offsets_3d(umesh);

  // Initialises the connectivity between cells and faces
  init_cells_to_faces_3d(nx, ny, nz, umesh);

  // Initialises the offsets between nodes and faces
  init_nodes_to_faces_offsets_3d(umesh);

  // Initialises the connectivity between nodes and faces
  init_nodes_to_faces_3d(nx, ny, nz, umesh);

  // Initialises the connectivity between faces and cells
  init_faces_to_cells_3d(nx, ny, nz, umesh);

  // Initialises the cells to nodes connectivity
  init_cells_to_nodes_3d(nx, ny, nz, umesh);

  // Initialises the boundary normals
  init_boundary_normals_3d(nx, ny, nz, umesh);

  return allocated;
}
