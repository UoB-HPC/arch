#include <stdlib.h>
#include <assert.h>
#include "shared.h"
#include "umesh.h"
#include "params.h"

// Initialises the unstructured mesh variables
size_t initialise_unstructured_mesh(
    UnstructuredMesh* umesh)
{
  // Allocate the data structures that we now know the sizes of
  size_t allocated = allocate_data(&umesh->nodes_x0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_x1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y1, umesh->nnodes);
  allocated += allocate_data(&umesh->cell_centroids_x, umesh->ncells);
  allocated += allocate_data(&umesh->cell_centroids_y, umesh->ncells);
  allocated += allocate_int_data(&umesh->boundary_index, umesh->nnodes);
  allocated += allocate_int_data(&umesh->nodes_to_cells_off, umesh->nnodes+1);
  allocated += allocate_int_data(&umesh->cells_to_nodes_off, umesh->ncells+1);
  allocated += allocate_int_data(&umesh->cells_to_nodes, umesh->ncells*umesh->nnodes_by_cell);
  allocated += allocate_int_data(&umesh->nodes_to_cells, umesh->ncells*umesh->nnodes_by_cell);
  allocated += allocate_int_data(&umesh->node_neighbours, umesh->nnodes);
  return allocated;
}

// Reads the nodes data from the unstructured mesh definition
void read_nodes_data(
    UnstructuredMesh* umesh)
{
  // Open the files
  FILE* node_fp = fopen(umesh->node_filename, "r");
  FILE* ele_fp = fopen(umesh->ele_filename, "r");
  if(!node_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->node_filename);
  }
  if(!ele_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->ele_filename);
  }

  // Fetch the first line of the nodes file
  char buf[MAX_STR_LEN];
  char* line = buf;

  // Read the number of nodes, for allocation
  fgets(line, MAX_STR_LEN, node_fp);
  skip_whitespace(&line);
  sscanf(line, "%d", &umesh->nnodes);

  // Read meta data from the element file
  fgets(line, MAX_STR_LEN, ele_fp);
  skip_whitespace(&line);
  sscanf(line, "%d%d%d", &umesh->ncells, &umesh->nnodes_by_cell, &umesh->nregional_variables);

  fclose(ele_fp);
  fclose(node_fp);

  // Skip first line of both files
  fgets(line, MAX_STR_LEN, node_fp);
  fgets(line, MAX_STR_LEN, ele_fp);

  // Loop through the node file, storing all of the nodes in our data structure
  umesh->nboundary_cells = 0;
  while(fgets(line, MAX_STR_LEN, node_fp)) {
    int index;
    int is_boundary;
    int discard;
    sscanf(line, "%d", &index); 
    sscanf(line, "%d%lf%lf%d", &discard, &umesh->nodes_x0[(index)], 
        &umesh->nodes_y0[(index)], &is_boundary);

    umesh->boundary_index[(index)] = (is_boundary) 
      ? umesh->nboundary_cells++ : IS_INTERIOR_NODE;
  }

  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nboundary_cells);
}

// Reads the element data from the unstructured mesh definition
size_t read_element_data(
    UnstructuredMesh* umesh, double** variables)
{
  size_t allocated = initialise_unstructured_mesh(umesh);

  // Open the files
  FILE* ele_fp = fopen(umesh->ele_filename, "r");
  if(!ele_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->ele_filename);
  }

  char buf[MAX_STR_LEN];
  char* line = buf;

  int boundary_edge_index = 0;
  int* boundary_edge_list;
  allocated += allocate_int_data(&boundary_edge_list, umesh->nboundary_cells*2);

  // Loop through the element file and flatten into data structure
  while(fgets(line, MAX_STR_LEN, ele_fp)) {
    // Read in the index
    int index;
    char* line_temp = line;
    read_token(&line_temp, "%d", &index);

    // Read in each of the node locations
    int node[umesh->nnodes_by_cell];
    for(int ii = 0; ii < umesh->nnodes_by_cell; ++ii) {
      read_token(&line_temp, "%d", &node[ii]);
    }

    // Read in each of the regional variables
    for(int ii = 0; ii < umesh->nregional_variables; ++ii) {
      read_token(&line_temp, "%lf", &variables[ii][index]);
    }

    // Store the cell offsets in case of future mixed cell geometry
    umesh->cells_to_nodes_off[(index+1)] = 
      umesh->cells_to_nodes_off[(index)] + umesh->nnodes_by_cell;

    // Store cells to nodes and check if we are at a boundary edge cell
    int nboundary_nodes = 0;
    for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
      umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+nn] = node[(nn)];
      umesh->nodes_to_cells_off[(node[(nn)])+1]++;
      nboundary_nodes += (umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE);
    }

    // Only store edges that are on the boundary, maintaining the 
    // counter-clockwise order
    if(nboundary_nodes == 2) {
      for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
        const int next_node_index = (nn+1 == umesh->nnodes_by_cell ? 0 : nn+1);
        if(umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE && 
            umesh->boundary_index[(node[next_node_index])] != IS_INTERIOR_NODE) {
          boundary_edge_list[boundary_edge_index++] = node[nn];
          boundary_edge_list[boundary_edge_index++] = node[next_node_index];
          break;
        }
      }
    }

    // Check that we are storing the nodes in the correct order
    double A = 0.0;
    for(int ii = 0; ii < umesh->nnodes_by_cell; ++ii) {
      const int ii2 = (ii+1) % umesh->nnodes_by_cell; 
      A += (umesh->nodes_x0[node[ii]]+umesh->nodes_x0[node[ii2]])*
        (umesh->nodes_y0[node[ii2]]-umesh->nodes_y0[node[ii]]);
    }
    assert(A > 0.0 && "Nodes are not stored in counter-clockwise order.\n");
  }

  // Turning count container into an offset container
  for(int nn = 0; nn < umesh->nnodes; ++nn) {
    umesh->nodes_to_cells_off[(nn+1)] += umesh->nodes_to_cells_off[(nn)];
  }

  // Fill all nodes with undetermined values
  for(int ii = 0; ii < umesh->ncells*umesh->nnodes_by_cell; ++ii) {
    umesh->nodes_to_cells[(ii)] = -1;
  }

  // Fill in the list of cells surrounding nodes
  for(int cc = 0; cc < umesh->ncells; ++cc) {
    for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
      const int node = umesh->cells_to_nodes[(cc*umesh->nnodes_by_cell)+(nn)];
      const int off = umesh->nodes_to_cells_off[(node)];
      const int len = umesh->nodes_to_cells_off[(node+1)]-off;

      // Using a discovery loop, is there any better way?
      for(int ii = 0; ii < len; ++ii) {
        if(umesh->nodes_to_cells[(off+ii)] == -1) {
          umesh->nodes_to_cells[(off+ii)] = cc;
          break;
        }
      }
    }
  }

  find_boundary_normals(umesh, boundary_edge_list);

  return allocated;
}

// Reads an unstructured mesh from an input file
size_t convert_mesh_to_umesh(
    UnstructuredMesh* umesh, Mesh* mesh)
{
  size_t allocated = initialise_unstructured_mesh(umesh);

  // Loop through the node file, storing all of the nodes in our data structure
  umesh->nboundary_cells = 0;
  umesh->nnodes_by_cell = 4; // Initialising as rectilinear mesh
  umesh->nnodes = (mesh->local_nx+1)*(mesh->local_ny+1);

  // Store the boundary index value for all cells
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
      const int index = (ii*mesh->local_nx)+(jj);

      if(ii == 0 || jj == 0 || ii == (mesh->local_ny-1) || jj == (mesh->local_nx-1)) {
        umesh->boundary_index[(index)] = umesh->nboundary_cells++;
      }
      else {
        umesh->boundary_index[(index)] = IS_INTERIOR_NODE;
      }
    }
  }

  int* boundary_edge_list;
  int boundary_edge_index = 0;
  allocated += allocate_int_data(&boundary_edge_list, umesh->nboundary_cells*2);
  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nboundary_cells);

  // Calculate the cells to nodes offsets and values
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
      const int index = (ii*mesh->local_nx)+(jj);
      umesh->cells_to_nodes_off[(index+1)] = 
        umesh->cells_to_nodes_off[(index+1)] + umesh->nnodes_by_cell;

      // Simple closed form calculation for the nodes surrounding a cell
      umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+0] = (ii*mesh->local_nx)+(jj);
      umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+1] = (ii*mesh->local_nx)+(jj+1);
      umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+2] = ((ii+1)*mesh->local_nx)+(jj);
      umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+3] = ((ii+1)*mesh->local_nx)+(jj+1);
    }
  }

  // Initialise the offsets and list of nodes to cells, counter-clockwise order
  for(int ii = 0; ii < (mesh->local_ny+1); ++ii) {
    for(int jj = 0; jj < (mesh->local_nx+1); ++jj) {
      const int node_index = (ii*(mesh->local_nx+1))+(jj);

      umesh->nodes_y0[(node_index)] = mesh->edgey[(ii)];
      umesh->nodes_x0[(node_index)] = mesh->edgex[(jj)];

      int off = umesh->nodes_to_cells_off[(node_index)];
      if(ii == 0) {
        if(jj == 0) {
          umesh->nodes_to_cells[(off++)] = (ii*mesh->local_nx)+(jj);
        }
        else if(jj == (mesh->local_nx)) {
          umesh->nodes_to_cells[(off++)] = (ii*mesh->local_nx)+(jj-1);
        }
        else {
          // Boundary nodes have two adjoining cells
          umesh->nodes_to_cells[(off++)] = (ii*mesh->local_nx)+(jj-1);
          umesh->nodes_to_cells[(off++)] = (ii*mesh->local_nx)+(jj);
        }
      }
      else if(ii == (mesh->local_ny)) {
        // Corner nodes only have a single adjoining cell
        if(jj == 0) { 
          umesh->nodes_to_cells[(off++)] = ((mesh->local_ny-1)*mesh->local_nx)+(jj);
        }
        else if(jj == (mesh->local_nx)) {
          umesh->nodes_to_cells[(off++)] = ((mesh->local_ny-1)*mesh->local_nx)+(jj-1);
        }
        else {
          // Boundary nodes have two adjoining cells
          umesh->nodes_to_cells[(off++)] = ((mesh->local_ny-1)*mesh->local_nx)+(jj-1);
          umesh->nodes_to_cells[(off++)] = ((mesh->local_ny-1)*mesh->local_nx)+(jj);
        }
      }
      else if(jj == 0) {
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx)+(jj);
        umesh->nodes_to_cells[(off++)] = ((ii-1)*mesh->local_nx)+(jj);
      }
      else if(jj == (mesh->local_nx)) {
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx)+(jj-1);
        umesh->nodes_to_cells[(off++)] = ((ii-1)*mesh->local_nx)+(jj-1);
      }
      else {
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx)+(jj);
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx)+(jj+1);
        umesh->nodes_to_cells[(off++)] = ((ii+1)*mesh->local_nx)+(jj+1);
        umesh->nodes_to_cells[(off++)] = ((ii+1)*mesh->local_nx)+(jj);
      }

      // Store the calculated offset
      umesh->nodes_to_cells_off[(node_index)] = off;
    }
  }

  // Initialise the boundary edge list
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
      const int cell_index = (ii*mesh->local_nx)+(jj);
      const int cell_offset = umesh->cells_to_nodes_off[(cell_index)];
      const int* nodes = &umesh->cells_to_nodes[(cell_offset)];
      for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
        const int next_node_index = (nn+1 == umesh->nnodes_by_cell ? 0 : nn+1);
        if(umesh->boundary_index[(nodes[(nn)])] != IS_INTERIOR_NODE && 
            umesh->boundary_index[(nodes[(next_node_index)])] != IS_INTERIOR_NODE) {
          boundary_edge_list[(boundary_edge_index++)] = nodes[(nn)];
          boundary_edge_list[(boundary_edge_index++)] = nodes[(next_node_index)];
        }
      }
    }
  }

  find_boundary_normals(umesh, boundary_edge_list);

  return allocated;
}

