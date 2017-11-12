#ifndef __UMESHHDR
#define __UMESHHDR

#include "mesh.h"
#include <stdlib.h>

/* Problem-Independent Constants */
#define IS_INTERIOR -1
#define IS_BOUNDARY -2
#define IS_EDGE -3
#define IS_CORNER -4

// Constants for defining cartesian mesh in fully unstructured manner
#define NNODES_BY_CELL 8
#define NNODES_BY_NODE 6
#define NNODES_BY_FACE 4
#define NFACES_BY_NODE 12
#define NFACES_BY_CELL 6

#ifdef __cplusplus
extern "C" {
#endif

/*
 * UNSTRUCTURED MESHES
 */

// Stores unstructured mesh
typedef struct {
  int ncells;
  int nnodes;
  int nnodes_by_cell;
  int nnodes_by_node;
  int nregional_variables;
  int nboundary_nodes;
  int nfaces;

  int* boundary_index;
  int* boundary_type;

  // CURRENTLY 2D
  int* nodes_to_cells;
  int* cells_to_nodes;
  int* nodes_to_cells_offsets;
  int* cells_to_nodes_offsets;

  // CURRENTLY 3D
  int* faces_to_cells0;
  int* faces_to_cells1;
  int* cells_to_faces;
  int* faces_to_nodes;
  int* nodes_to_faces;
  int* nodes_to_nodes;
  int* faces_cclockwise_cell;
  int* faces_to_nodes_offsets;
  int* cells_to_faces_offsets;
  int* nodes_to_faces_offsets;
  int* nodes_to_nodes_offsets;

  double* nodes_x0;
  double* nodes_y0;
  double* nodes_z0;
  double* nodes_x1;
  double* nodes_y1;
  double* nodes_z1;
  double* cell_centroids_x;
  double* cell_centroids_y;
  double* cell_centroids_z;
  double* boundary_normal_x;
  double* boundary_normal_y;
  double* boundary_normal_z;
  double* sub_cell_volume;

  char* node_filename;
  char* ele_filename;

} UnstructuredMesh;

// Initialises the unstructured mesh variables
size_t init_unstructured_mesh(UnstructuredMesh* umesh);

// Reads the element data from the unstructured mesh definition
size_t read_unstructured_mesh(UnstructuredMesh* umesh, double*** variables,
                              int nvars);

// Finds the normals for all boundary cells
void find_boundary_normals(UnstructuredMesh* umesh, int* boundary_face_list);
void find_boundary_normals_3d(UnstructuredMesh* umesh, int* boundary_face_list);

// Converts an ordinary structured mesh into an unstructured equivalent
size_t convert_mesh_to_umesh(UnstructuredMesh* umesh, Mesh* mesh);

// Converts an ordinary structured mesh into an unstructured equivalent
size_t convert_mesh_to_umesh_3d(UnstructuredMesh* umesh, Mesh* mesh);

// Converts the lists of cell counts to a list of offsets
void convert_cell_counts_to_offsets(UnstructuredMesh* umesh);

// Fill in the list of cells surrounding nodes
void fill_nodes_to_cells(UnstructuredMesh* umesh);

// Determine the cells that neighbour other cells
void fill_cells_to_cells(UnstructuredMesh* umesh);

// Determine the nodes that surround other nodes
void fill_nodes_to_nodes(UnstructuredMesh* umesh);

// Initialises the list of nodes to cells
void init_nodes_to_cells_3d(const int nx, const int ny, const int nz,
                            Mesh* mesh, UnstructuredMesh* umesh);

// Initialises the list of nodes to nodes
void init_nodes_to_nodes_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh);

// Initialises the connectivity between faces and nodes
void init_faces_to_nodes_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh);

// Initialises the offsets between faces and nodes
void init_faces_to_nodes_offsets_3d(UnstructuredMesh* umesh);

// Initialises the offsets between cells and faces
void init_cells_to_faces_offsets_3d(UnstructuredMesh* umesh);

// Initialises the connectivity between cells and faces
void init_cells_to_faces_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh);

// Initialises the offsets between nodes and faces
void init_nodes_to_faces_offsets_3d(UnstructuredMesh* umesh);

// Initialises the connectivity between nodes and faces
void init_nodes_to_faces_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh);

// Initialises the connectivity between faces and cells
void init_faces_to_cells_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh);

// Initialises the cells to nodes connectivity
void init_cells_to_nodes_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh);

// Initialises the boundary normals
void init_boundary_normals_3d(const int nx, const int ny, const int nz,
                              UnstructuredMesh* umesh);

#ifdef __cplusplus
}
#endif

#endif
