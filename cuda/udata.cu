#include "../umesh.h"
#include "../shared.h"
#include "shared.h"
#include "udata.k"

// Initialises the offsets between faces and nodes
void init_faces_to_nodes_offsets_3d(UnstructuredMesh* umesh) {

  int nblocks = ceil((umesh->nfaces+1)/(double)NTHREADS);
  faces_to_nodes_offsets_3d<<<nblocks, NTHREADS>>>(
      umesh->nfaces, umesh->faces_to_nodes_offsets, umesh->faces_cclockwise_cell);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the offsets between cells and faces
void init_cells_to_faces_offsets_3d(UnstructuredMesh* umesh) {

  int nblocks = ceil((umesh->ncells+1)/(double)NTHREADS);
  cells_to_faces_offsets_3d<<<nblocks, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_faces_offsets);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the offsets between nodes and faces
void init_nodes_to_faces_offsets_3d(UnstructuredMesh* umesh) {

  int nblocks = ceil((umesh->nnodes+1)/(double)NTHREADS);
  nodes_to_faces_offsets_3d<<<nblocks, NTHREADS>>>(
      umesh->nnodes, umesh->nodes_to_faces_offsets);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the connectivity between faces and cells
void init_faces_to_cells_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  int nblocks = ceil((nz+1)/(double)NTHREADS);
  faces_to_cells_3d<<<nblocks, NTHREADS>>>(
      nx, ny, nz, umesh->faces_to_cells0, umesh->faces_to_cells1);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the connectivity between nodes and faces
void init_nodes_to_faces_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  int nblocks = ceil((nx+1)*(ny+1)*(nz+1)/(double)NTHREADS);
  nodes_to_faces_3d<<<nblocks, NTHREADS>>>(
      nx, ny, nz, umesh->nodes_to_faces_offsets, umesh->nodes_to_faces);

  gpu_check(cudaDeviceSynchronize());
}

// Initialises the cells to nodes connectivity
void init_cells_to_nodes_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  int nblocks = ceil((umesh->ncells+1)/(double)NTHREADS);
  cells_to_nodes_offsets_3d<<<nblocks, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_nodes_offsets);

  nblocks = ceil(nx*ny*nz/(double)NTHREADS);
  cells_to_nodes_3d<<<nblocks, NTHREADS>>>(nx, ny, nz, umesh->cells_to_nodes);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the connectivity between faces and nodes
void init_faces_to_nodes_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  // Connectivity of faces to nodes, the nodes are stored in a counter-clockwise
  // ordering around the face
  int nblocks = ceil((nz+1)/(double)NTHREADS);
  faces_to_nodes_3d<<<nblocks, NTHREADS>>>(
      nx, ny, nz, umesh->faces_cclockwise_cell, umesh->faces_to_nodes);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the connectivity between cells and faces
void init_cells_to_faces_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  int nblocks = ceil(nx*ny*nz/(double)NTHREADS);
  cells_to_faces_3d<<<nblocks, NTHREADS>>>(nx, ny, nz, umesh->cells_to_faces);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the list of nodes to nodes
void init_nodes_to_nodes_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  int nblocks = ceil((umesh->nnodes+1)/(double)NTHREADS);
  nodes_to_nodes_offsets_3d<<<nblocks, NTHREADS>>>(
      umesh->nnodes, umesh->nodes_to_nodes_offsets);

  nblocks = ceil((nx+1)*(ny+1)*(nz+1)/(double)NTHREADS);
  nodes_to_nodes_3d<<<nblocks, NTHREADS>>>(nx, ny, nz, umesh->nodes_to_nodes);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the list of nodes to cells
void init_nodes_to_cells_3d(const int nx, const int ny, const int nz,
    Mesh* mesh, UnstructuredMesh* umesh) {

  int nblocks = ceil((nx+1)*(ny+1)*(nz+1)/(double)NTHREADS);
  nodes_3d<<<nblocks, NTHREADS>>>(
      nx, ny, nz, mesh->edgex, mesh->edgey, mesh->edgez, 
      umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0);

  nodes_to_cells_3d<<<nblocks, NTHREADS>>>(nx, ny, nz, umesh->nodes_to_cells);
  gpu_check(cudaDeviceSynchronize());
}

// Initialises the boundary normals
void init_boundary_normals_3d(const int nx, const int ny, const int nz,
    UnstructuredMesh* umesh) {

  int* boundary_index;
  int* boundary_type;
  double* boundary_normal_x;
  double* boundary_normal_y;
  double* boundary_normal_z;
  allocate_host_int_data(&boundary_index, umesh->nnodes);
  allocate_host_int_data(&boundary_type, umesh->nboundary_nodes);
  allocate_host_data(&boundary_normal_x, umesh->nboundary_nodes);
  allocate_host_data(&boundary_normal_y, umesh->nboundary_nodes);
  allocate_host_data(&boundary_normal_z, umesh->nboundary_nodes);

  // Determine all of the boundary edges
  int nboundary_nodes = 0;
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {

        const int boundary_count = ((ii == 0) + (ii == nz) + (jj == 0) +
            (jj == ny) + (kk == 0) + (kk == nx));

        const int node_index =
          (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        // Check if we are on the edge
        if (boundary_count > 0) {
          int index = nboundary_nodes++;
          boundary_index[(node_index)] = index;

          if (boundary_count == 3) {
            boundary_type[(index)] = IS_CORNER;
          } else if (boundary_count == 2) {
            boundary_type[(index)] = IS_EDGE;
            if (kk == 0) {
              if (jj == 0) {
                boundary_normal_z[(index)] = 1.0;
              } else if (jj == ny) {
                boundary_normal_z[(index)] = 1.0;
              } else if (ii == 0) {
                boundary_normal_y[(index)] = 1.0;
              } else if (ii == nz) {
                boundary_normal_y[(index)] = 1.0;
              }
            } else if (jj == 0) {
              if (kk == nx) {
                boundary_normal_z[(index)] = 1.0;
              } else if (ii == 0) {
                boundary_normal_x[(index)] = 1.0;
              } else if (ii == nz) {
                boundary_normal_x[(index)] = 1.0;
              }
            } else if (ii == 0) {
              if (kk == nx) {
                boundary_normal_y[(index)] = 1.0;
              } else if (jj == ny) {
                boundary_normal_x[(index)] = 1.0;
              }
            } else if (kk == nx) {
              if (ii == nz) {
                boundary_normal_y[(index)] = 1.0;
              } else if (jj == ny) {
                boundary_normal_z[(index)] = 1.0;
              }
            } else if (jj == ny) {
              if (ii == nz) {
                boundary_normal_x[(index)] = 1.0;
              }
            }
          } else if (boundary_count == 1) {
            boundary_type[(index)] = IS_BOUNDARY;

            // TODO: WE DON'T NEED ANYTHING SPECIAL HERE AS WE KNOW THE
            // NORMALS
            // FROM THE CONSTRUCTION OF THE MESH, ALTHOUGH WE WILL NEED A
            // SUFFICIENT METHOD WHEN WE START USING MORE COMPLEX MESHES
            boundary_normal_x[(index)] =
              (kk == 0) ? -1.0 : ((kk == nx) ? 1.0 : 0.0);
            boundary_normal_y[(index)] =
              (jj == 0) ? -1.0 : ((jj == ny) ? 1.0 : 0.0);
            boundary_normal_z[(index)] =
              (ii == 0) ? -1.0 : ((ii == nz) ? 1.0 : 0.0);
          }
        } else {
          boundary_index[(node_index)] = IS_INTERIOR;
        }
      }
    }
  }

  copy_buffer(umesh->nboundary_nodes, &boundary_normal_x, &umesh->boundary_normal_x, 1);
  copy_buffer(umesh->nboundary_nodes, &boundary_normal_y, &umesh->boundary_normal_y, 1);
  copy_buffer(umesh->nboundary_nodes, &boundary_normal_z, &umesh->boundary_normal_z, 1);
  copy_int_buffer(umesh->nboundary_nodes, &boundary_type, &umesh->boundary_type, 1);
  copy_int_buffer(umesh->nnodes, &boundary_index, &umesh->boundary_index, 1);

  deallocate_host_int_data(boundary_type);
  deallocate_host_int_data(boundary_index);
  deallocate_host_data(boundary_normal_x);
  deallocate_host_data(boundary_normal_y);
  deallocate_host_data(boundary_normal_z);

  gpu_check(cudaDeviceSynchronize());
}
