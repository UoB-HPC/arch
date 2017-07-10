#ifndef __UMESHHDR
#define __UMESHHDR

#include <stdlib.h>
#include "mesh.h"

/* Problem-Independent Constants */
#define IS_INTERIOR_NODE -1
#define IS_FIXED -1
#define IS_BOUNDARY -2

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
    int nregional_variables;
    int nboundary_cells;

    int* nodes_to_cells;
    int* cells_to_nodes; 
    int* nodes_to_cells_off;
    int* cells_to_nodes_off; 
    int* boundary_index;
    int* boundary_type;
    int* node_neighbours;

    double* nodes_x0; 
    double* nodes_y0; 
    double* nodes_x1; 
    double* nodes_y1;
    double* cell_centroids_x;
    double* cell_centroids_y;
    double* boundary_normal_x;
    double* boundary_normal_y;
    double* sub_cell_volume;

    char* node_filename;
    char* ele_filename;

  } UnstructuredMesh;

  // Initialises the unstructured mesh variables
  size_t initialise_unstructured_mesh(
      UnstructuredMesh* umesh);

  // Reads the nodes data from the unstructured mesh definition
  void read_nodes_data(
      UnstructuredMesh* umesh);

  // Reads the element data from the unstructured mesh definition
  size_t read_element_data(
      UnstructuredMesh* umesh, double** variables);

  // Finds the normals for all boundary cells
  void find_boundary_normals(
      UnstructuredMesh* umesh, int* boundary_edge_list);

  // Finalises the mesh
  void finalise_mesh(
      Mesh* mesh);

#ifdef __cplusplus
}
#endif

#endif
