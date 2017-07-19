#include "umesh.h"
#include "params.h"
#include "shared.h"
#include <assert.h>
#include <stdlib.h>

// Converts the lists of cell counts to a list of offsets
void convert_cell_counts_to_offsets(UnstructuredMesh* umesh);

// Fill in the list of cells surrounding nodes
void fill_nodes_to_cells(UnstructuredMesh* umesh);

// Determine the cells that neighbour other cells
void fill_cells_to_cells(UnstructuredMesh* umesh);

// Determine the nodes that surround other nodes
void fill_nodes_to_nodes(UnstructuredMesh* umesh);

// Initialises the unstructured mesh variables
size_t initialise_unstructured_mesh(UnstructuredMesh* umesh) {
  // Allocate the data structures that we now know the sizes of
  size_t allocated = allocate_data(&umesh->cell_centroids_x, umesh->ncells);
  allocated += allocate_data(&umesh->cell_centroids_y, umesh->ncells);
  allocated += allocate_data(&umesh->cell_centroids_z, umesh->ncells);
  allocated += allocate_int_data(&umesh->nodes_offsets, umesh->nnodes + 1);
  allocated += allocate_int_data(&umesh->cells_offsets, umesh->ncells + 1);
  allocated += allocate_int_data(&umesh->cells_to_nodes,
                                 umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_int_data(&umesh->nodes_to_cells,
                                 umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_int_data(&umesh->nodes_to_nodes,
                                 umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_int_data(&umesh->cells_to_cells,
                                 umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&umesh->sub_cell_volume,
                             umesh->ncells * umesh->nnodes_by_cell);
  return allocated;
}

// Reads the nodes data from the unstructured mesh definition
size_t read_unstructured_mesh(UnstructuredMesh* umesh, double*** cell_variables,
                              int nvars) {
  // Open the files
  FILE* node_fp = fopen(umesh->node_filename, "r");
  if (!node_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->node_filename);
  }

  // Fetch the first line of the nodes file
  char buf[MAX_STR_LEN];
  char* line = buf;

  // Read the number of nodes, for allocation
  fgets(line, MAX_STR_LEN, node_fp);
  skip_whitespace(&line);
  sscanf(line, "%d", &umesh->nnodes);

  size_t allocated = allocate_data(&umesh->nodes_x0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_x1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y1, umesh->nnodes);
  allocated += allocate_int_data(&umesh->boundary_index, umesh->nnodes);

  // Loop through the node file, storing all of the nodes in our data structure
  umesh->nboundary_cells = 0;
  while (fgets(line, MAX_STR_LEN, node_fp)) {
    int index;
    int is_boundary;
    int discard;
    sscanf(line, "%d", &index);
    sscanf(line, "%d%lf%lf%d", &discard, &umesh->nodes_x0[(index)],
           &umesh->nodes_y0[(index)], &is_boundary);

    umesh->boundary_index[(index)] =
        (is_boundary) ? umesh->nboundary_cells++ : IS_INTERIOR_NODE;
  }

  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nboundary_cells);

  fclose(node_fp);

  // Open the files
  FILE* ele_fp = fopen(umesh->ele_filename, "r");
  if (!ele_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->ele_filename);
  }

  // Read meta data from the element file
  fgets(line, MAX_STR_LEN, ele_fp);
  skip_whitespace(&line);
  sscanf(line, "%d%d%d", &umesh->ncells, &umesh->nnodes_by_cell,
         &umesh->nregional_variables);

  // Constant nedges by construction
  umesh->nsub_cell_edges = 4;

  int boundary_edge_index = 0;
  int* boundary_edge_list;

  assert(nvars == umesh->nregional_variables &&
         "The number of variables passed to read_element_data \
      doesn't match the input file.");

  allocated += initialise_unstructured_mesh(umesh);
  for (int ii = 0; ii < umesh->nregional_variables; ++ii) {
    allocated += allocate_data(cell_variables[ii], umesh->ncells);
  }

  allocate_int_data(&boundary_edge_list, umesh->nboundary_cells * 2);

  // Loop through the element file and flatten into data structure
  while (fgets(line, MAX_STR_LEN, ele_fp)) {
    // Read in the index
    int index;
    char* line_temp = line;
    read_token(&line_temp, "%d", &index);

    // Read in each of the node locations
    int node[umesh->nnodes_by_cell];
    for (int ii = 0; ii < umesh->nnodes_by_cell; ++ii) {
      read_token(&line_temp, "%d", &node[ii]);
    }

    // Check that we are storing the nodes in the correct order
    double A = 0.0;
    for (int ii = 0; ii < umesh->nnodes_by_cell; ++ii) {
      const int ii2 = (ii + 1) % umesh->nnodes_by_cell;
      A += (umesh->nodes_x0[node[ii]] + umesh->nodes_x0[node[ii2]]) *
           (umesh->nodes_y0[node[ii2]] - umesh->nodes_y0[node[ii]]);
    }
    assert(A > 0.0 && "Nodes are not stored in counter-clockwise order.\n");

    // Read in each of the regional variables
    for (int ii = 0; ii < umesh->nregional_variables; ++ii) {
      read_token(&line_temp, "%lf", &(*(cell_variables[ii]))[index]);
    }

    // Store the cell offsets in case of future mixed cell geometry
    umesh->cells_offsets[(index + 1)] =
        umesh->cells_offsets[(index)] + umesh->nnodes_by_cell;

    // Store cells to nodes and check if we are at a boundary edge cell
    int nboundary_nodes = 0;
    for (int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
      umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + nn] = node[(nn)];
      umesh->nodes_offsets[(node[(nn)]) + 1]++;
      nboundary_nodes +=
          (umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE);
    }

    // Only store edges that are on the boundary, maintaining the
    // counter-clockwise order
    if (nboundary_nodes == 2) {
      for (int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
        const int next_node_index =
            (nn + 1 == umesh->nnodes_by_cell ? 0 : nn + 1);
        if (umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE &&
            umesh->boundary_index[(node[next_node_index])] !=
                IS_INTERIOR_NODE) {
          boundary_edge_list[boundary_edge_index++] = node[nn];
          boundary_edge_list[boundary_edge_index++] = node[next_node_index];
          break;
        }
      }
    }
  }

  convert_cell_counts_to_offsets(umesh);
  fill_nodes_to_cells(umesh);
  fill_cells_to_cells(umesh);
  find_boundary_normals(umesh, boundary_edge_list);
  deallocate_int_data(boundary_edge_list);

  fclose(ele_fp);
  return allocated;
}

// Converts the lists of cell counts to a list of offsets
void convert_cell_counts_to_offsets(UnstructuredMesh* umesh) {
  for (int nn = 0; nn < umesh->nnodes; ++nn) {
    umesh->nodes_offsets[(nn + 1)] += umesh->nodes_offsets[(nn)];
  }
}

// Fill in the list of cells surrounding nodes
void fill_nodes_to_cells(UnstructuredMesh* umesh) {
  // Fill all nodes with undetermined values
  for (int nn = 0; nn < umesh->ncells * umesh->nnodes_by_cell; ++nn) {
    umesh->nodes_to_cells[(nn)] = -1;
  }

  for (int cc = 0; cc < umesh->ncells; ++cc) {
    // Fill the next slot
    for (int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
      const int node =
          umesh->cells_to_nodes[(cc * umesh->nnodes_by_cell) + (nn)];
      const int off = umesh->nodes_offsets[(node)];
      const int len = umesh->nodes_offsets[(node + 1)] - off;

      // Using a discovery loop, is there any better way?
      for (int ii = 0; ii < len; ++ii) {
        if (umesh->nodes_to_cells[(off + ii)] == -1) {
          umesh->nodes_to_cells[(off + ii)] = cc;
          break;
        }
      }
    }
  }
}

// Determine the cells that neighbour other cells
void fill_cells_to_cells(UnstructuredMesh* umesh) {
  // Initialise all of the neighbour entries as boundary entries
  for (int cc = 0; cc < umesh->ncells; ++cc) {
    const int cells_off = umesh->cells_offsets[(cc)];
    const int nnodes_by_cell = umesh->cells_offsets[(cc + 1)] - cells_off;

    // Look at each of the nodes
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      umesh->cells_to_cells[(cells_off) + (nn)] = IS_BOUNDARY;
    }
  }

  // Loop over all of the cells
  for (int cc = 0; cc < umesh->ncells; ++cc) {
    const int cells_off = umesh->cells_offsets[(cc)];
    const int nnodes_by_cell = umesh->cells_offsets[(cc + 1)] - cells_off;

    // Look at each of the nodes
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_c_index = umesh->cells_to_nodes[(cells_off) + (nn)];

      // We're going to consider the nodes next to our current node
      const int node_l_index =
          (nn - 1 >= 0)
              ? umesh->cells_to_nodes[(cells_off + nn - 1)]
              : umesh->cells_to_nodes[(cells_off + nnodes_by_cell - 1)];
      const int nodes_off = umesh->nodes_offsets[(node_c_index)];
      const int ncells_by_node =
          umesh->nodes_offsets[(node_c_index + 1)] - nodes_off;

      // Look at all of the cells that connect to this particular node
      int nneighbours_found = 0;
      for (int cc2 = 0; cc2 < ncells_by_node; ++cc2) {
        const int cell_index2 = umesh->nodes_to_cells[(nodes_off + cc2)];
        if (cc == cell_index2) {
          continue;
        }

        const int cells_off2 = umesh->cells_offsets[(cell_index2)];
        const int nnodes_by_cell2 =
            umesh->cells_offsets[(cell_index2 + 1)] - cells_off2;

        // Check whether this cell contains the edge node
        for (int nn2 = 0; nn2 < nnodes_by_cell2; ++nn2) {
          if (node_l_index == umesh->cells_to_nodes[(cells_off2 + nn2)]) {
            umesh->cells_to_cells[(cells_off) + (nn)] = cell_index2;
            nneighbours_found++;
            break;
          }
        }
      }
    }
  }
}

// Determine the nodes that surround other nodes
void fill_nodes_to_nodes(UnstructuredMesh* umesh) {
  for (int nn = 0; nn < umesh->nnodes; ++nn) {
    const int nodes_off = umesh->nodes_offsets[(nn)];
    for (int cc = 0; cc < umesh->ncells; ++cc) {
      const int cell_index = umesh->nodes_to_cells[(nodes_off + cc)];
      const int cells_off = umesh->cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          umesh->cells_offsets[(cell_index + 1)] - cells_off;

      // Look at the nodes surrounding the cell and pick the one that is
      // clockwise from the current node
      for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        const int node_index = umesh->cells_to_nodes[(cells_off + nn2)];
        if (node_index == nn) {
          umesh->nodes_to_nodes[(nodes_off + cc)] =
              (nn2 > 0)
                  ? umesh->cells_to_nodes[(cells_off + (nn2 - 1))]
                  : umesh->cells_to_nodes[(cells_off + (nnodes_by_cell - 1))];
          break;
        }
      }
    }
  }
}

// Converts an ordinary structured mesh into an unstructured equivalent
size_t convert_mesh_to_umesh(UnstructuredMesh* umesh, Mesh* mesh) {
  size_t allocated = initialise_unstructured_mesh(umesh);
  allocated += allocate_data(&umesh->nodes_x0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_x1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y1, umesh->nnodes);
  allocated += allocate_int_data(&umesh->boundary_index, umesh->nnodes);

  // Loop through the node file, storing all of the nodes in our data structure
  umesh->nboundary_cells = 0;
  umesh->nnodes_by_cell = 4; // Initialising as rectilinear mesh
  umesh->nnodes = (mesh->local_nx + 1) * (mesh->local_ny + 1);

  // Store the boundary index value for all cells
  for (int ii = 0; ii < mesh->local_ny; ++ii) {
    for (int jj = 0; jj < mesh->local_nx; ++jj) {
      const int index = (ii * mesh->local_nx) + (jj);

      if (ii == 0 || jj == 0 || ii == (mesh->local_ny - 1) ||
          jj == (mesh->local_nx - 1)) {
        umesh->boundary_index[(index)] = umesh->nboundary_cells++;
      } else {
        umesh->boundary_index[(index)] = IS_INTERIOR_NODE;
      }
    }
  }

  int* boundary_edge_list;
  int boundary_edge_index = 0;
  allocated +=
      allocate_int_data(&boundary_edge_list, umesh->nboundary_cells * 2);
  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nboundary_cells);

  // Calculate the cells to nodes offsets and values
  for (int ii = 0; ii < mesh->local_ny; ++ii) {
    for (int jj = 0; jj < mesh->local_nx; ++jj) {
      const int index = (ii * mesh->local_nx) + (jj);
      umesh->cells_offsets[(index + 1)] =
          umesh->cells_offsets[(index)] + umesh->nnodes_by_cell;

      // Simple closed form calculation for the nodes surrounding a cell
      umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 0] =
          (ii * mesh->local_nx) + (jj);
      umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 1] =
          (ii * mesh->local_nx) + (jj + 1);
      umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 2] =
          ((ii + 1) * mesh->local_nx) + (jj);
      umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 3] =
          ((ii + 1) * mesh->local_nx) + (jj + 1);
    }
  }

  // Initialise the offsets and list of nodes to cells, counter-clockwise order
  for (int ii = 0; ii < (mesh->local_ny + 1); ++ii) {
    for (int jj = 0; jj < (mesh->local_nx + 1); ++jj) {
      const int node_index = (ii * (mesh->local_nx + 1)) + (jj);

      umesh->nodes_y0[(node_index)] = mesh->edgey[(ii)];
      umesh->nodes_x0[(node_index)] = mesh->edgex[(jj)];

      int off = umesh->nodes_offsets[(node_index)];
      if (ii == 0) {
        if (jj == 0) {
          umesh->nodes_to_cells[(off++)] = (ii * mesh->local_nx) + (jj);
        } else if (jj == (mesh->local_nx)) {
          umesh->nodes_to_cells[(off++)] = (ii * mesh->local_nx) + (jj - 1);
        } else {
          // Boundary nodes have two adjoining cells
          umesh->nodes_to_cells[(off++)] = (ii * mesh->local_nx) + (jj - 1);
          umesh->nodes_to_cells[(off++)] = (ii * mesh->local_nx) + (jj);
        }
      } else if (ii == (mesh->local_ny)) {
        // Corner nodes only have a single adjoining cell
        if (jj == 0) {
          umesh->nodes_to_cells[(off++)] =
              ((mesh->local_ny - 1) * mesh->local_nx) + (jj);
        } else if (jj == (mesh->local_nx)) {
          umesh->nodes_to_cells[(off++)] =
              ((mesh->local_ny - 1) * mesh->local_nx) + (jj - 1);
        } else {
          // Boundary nodes have two adjoining cells
          umesh->nodes_to_cells[(off++)] =
              ((mesh->local_ny - 1) * mesh->local_nx) + (jj - 1);
          umesh->nodes_to_cells[(off++)] =
              ((mesh->local_ny - 1) * mesh->local_nx) + (jj);
        }
      } else if (jj == 0) {
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx) + (jj);
        umesh->nodes_to_cells[(off++)] = ((ii - 1) * mesh->local_nx) + (jj);
      } else if (jj == (mesh->local_nx)) {
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx) + (jj - 1);
        umesh->nodes_to_cells[(off++)] = ((ii - 1) * mesh->local_nx) + (jj - 1);
      } else {
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx) + (jj);
        umesh->nodes_to_cells[(off++)] = ((ii)*mesh->local_nx) + (jj + 1);
        umesh->nodes_to_cells[(off++)] = ((ii + 1) * mesh->local_nx) + (jj + 1);
        umesh->nodes_to_cells[(off++)] = ((ii + 1) * mesh->local_nx) + (jj);
      }

      // Store the calculated offset
      umesh->nodes_offsets[(node_index)] = off;
    }
  }

  // Initialise the boundary edge list
  for (int ii = 0; ii < mesh->local_ny; ++ii) {
    for (int jj = 0; jj < mesh->local_nx; ++jj) {
      const int cell_index = (ii * mesh->local_nx) + (jj);
      const int cells_off = umesh->cells_offsets[(cell_index)];
      const int* nodes = &umesh->cells_to_nodes[(cells_off)];
      for (int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
        const int next_node_index =
            (nn + 1 == umesh->nnodes_by_cell ? 0 : nn + 1);
        if (umesh->boundary_index[(nodes[(nn)])] != IS_INTERIOR_NODE &&
            umesh->boundary_index[(nodes[(next_node_index)])] !=
                IS_INTERIOR_NODE) {
          boundary_edge_list[(boundary_edge_index++)] = nodes[(nn)];
          boundary_edge_list[(boundary_edge_index++)] =
              nodes[(next_node_index)];
        }
      }
    }
  }

  find_boundary_normals(umesh, boundary_edge_list);

  return allocated;
}

// Converts an ordinary structured mesh into an unstructured equivalent
size_t convert_mesh_to_umesh_3d(UnstructuredMesh* umesh, Mesh* mesh) {
  umesh->nboundary_cells = 0;
  umesh->nnodes_by_cell = 8; // Initialising as rectilinear mesh
  umesh->nnodes_by_cell = 8; // Initialising as rectilinear mesh
  umesh->nnodes =
      (mesh->local_nx + 1) * (mesh->local_ny + 1) * (mesh->local_nz + 1);
  umesh->ncells = (mesh->local_nx * mesh->local_ny * mesh->local_nz);

  size_t allocated = initialise_unstructured_mesh(umesh);
  allocated += allocate_data(&umesh->nodes_x0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_z0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_x1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_z1, umesh->nnodes);
  allocated += allocate_int_data(&umesh->boundary_index, umesh->nnodes);

  // Store the boundary index value for all cells
  for (int ii = 0; ii < mesh->local_nz; ++ii) {
    for (int jj = 0; jj < mesh->local_ny; ++jj) {
      for (int kk = 0; kk < mesh->local_nx; ++kk) {
        const int index = (ii * mesh->local_nx * mesh->local_ny) +
                          (jj * mesh->local_nx) + (kk);

        if (ii == 0 || jj == 0 || kk == 0 || ii == (mesh->local_nz - 1) ||
            jj == (mesh->local_ny - 1) || kk == (mesh->local_nx - 1)) {
          umesh->boundary_index[(index)] = umesh->nboundary_cells++;
        } else {
          umesh->boundary_index[(index)] = IS_INTERIOR_NODE;
        }
      }
    }
  }

  int* boundary_edge_list;
  int boundary_edge_index = 0;
  allocated +=
      allocate_int_data(&boundary_edge_list, umesh->nboundary_cells * 2);
  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_z, umesh->nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nboundary_cells);

  // Calculate the cells to nodes offsets and values
  for (int ii = 0; ii < mesh->local_nz; ++ii) {
    for (int jj = 0; jj < mesh->local_ny; ++jj) {
      for (int kk = 0; kk < mesh->local_nx; ++kk) {
        const int index = (ii * mesh->local_nx * mesh->local_ny) +
                          (jj * mesh->local_nx) + (kk);
        umesh->cells_offsets[(index + 1)] =
            umesh->cells_offsets[(index)] + umesh->nnodes_by_cell;

        // Simple closed form calculation for the nodes surrounding a cell
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 0] =
            (ii * mesh->local_nx * mesh->local_ny) + (jj * mesh->local_nx) +
            (kk);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 1] =
            (ii * mesh->local_nx * mesh->local_ny) + (jj * mesh->local_nx) +
            (kk + 1);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 2] =
            (ii * mesh->local_nx * mesh->local_ny) +
            ((jj + 1) * mesh->local_nx) + (kk);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 3] =
            (ii * mesh->local_nx * mesh->local_ny) +
            ((jj + 1) * mesh->local_nx) + (kk + 1);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 4] =
            ((ii + 1) * mesh->local_nx * mesh->local_ny) +
            (jj * mesh->local_nx) + (kk);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 5] =
            ((ii + 1) * mesh->local_nx * mesh->local_ny) +
            (jj * mesh->local_nx) + (kk + 1);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 6] =
            ((ii + 1) * mesh->local_nx * mesh->local_ny) +
            ((jj + 1) * mesh->local_nx) + (kk);
        umesh->cells_to_nodes[(index * umesh->nnodes_by_cell) + 7] =
            ((ii + 1) * mesh->local_nx * mesh->local_ny) +
            ((jj + 1) * mesh->local_nx) + (kk + 1);
      }
    }
  }

  // Initialise the offsets and list of nodes to cells, counter-clockwise order
  for (int ii = 0; ii < (mesh->local_nz + 1); ++ii) {
    for (int jj = 0; jj < (mesh->local_ny + 1); ++jj) {
      for (int kk = 0; kk < (mesh->local_nx + 1); ++kk) {
        const int node_index =
            (ii * (mesh->local_nx + 1) * (mesh->local_ny + 1)) +
            (jj * (mesh->local_nx + 1)) + (kk);

        umesh->nodes_z0[(node_index)] = mesh->edgez[(ii)];
        umesh->nodes_y0[(node_index)] = mesh->edgey[(jj)];
        umesh->nodes_x0[(node_index)] = mesh->edgex[(kk)];

        int off = umesh->nodes_offsets[(node_index)];

        // Fill in all of the cells that surround a node
        // NOTE: The order of the statements is important for data layout
        if (ii > 0 && jj > 0 && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * mesh->local_nx * mesh->local_ny) +
              ((jj - 1) * mesh->local_nx) + (kk - 1);
        }
        if (ii > 0 && jj > 0 && kk < mesh->local_nx) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * mesh->local_nx * mesh->local_ny) +
              ((jj - 1) * mesh->local_nx) + (kk);
        }
        if (ii > 0 && jj < mesh->local_ny && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * mesh->local_nx * mesh->local_ny) +
              (jj * mesh->local_nx) + (kk - 1);
        }
        if (ii > 0 && jj < mesh->local_ny && kk < mesh->local_nx) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * mesh->local_nx * mesh->local_ny) +
              (jj * mesh->local_nx) + (kk);
        }
        if (ii < mesh->local_nz && jj > 0 && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              (ii * mesh->local_nx * mesh->local_ny) +
              ((jj - 1) * mesh->local_nx) + (kk - 1);
        }
        if (ii < mesh->local_nz && jj > 0 && kk < mesh->local_nx) {
          umesh->nodes_to_cells[(off++)] =
              (ii * mesh->local_nx * mesh->local_ny) +
              ((jj - 1) * mesh->local_nx) + (kk);
        }
        if (ii < mesh->local_nz && jj < mesh->local_ny && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              (ii * mesh->local_nx * mesh->local_ny) + (jj * mesh->local_nx) +
              (kk - 1);
        }
        if (ii < mesh->local_nz && jj < mesh->local_ny && kk < mesh->local_nx) {
          umesh->nodes_to_cells[(off++)] =
              (ii * mesh->local_nx * mesh->local_ny) + (jj * mesh->local_nx) +
              (kk);
        }
        printf("off: %d\n", off);

        // Store the calculated offset
        umesh->nodes_offsets[(node_index + 1)] = off;
      }
    }
  }

  /// TODO: Should this become the boundary face list?

  // Initialise the boundary edge list
  for (int ii = 0; ii < mesh->local_nz; ++ii) {
    for (int jj = 0; jj < mesh->local_ny; ++jj) {
      for (int kk = 0; kk < mesh->local_nx; ++kk) {
        const int cell_index = (ii * mesh->local_nx * mesh->local_ny) +
                               (jj * mesh->local_nx) + (kk);
        const int cells_off = umesh->cells_offsets[(cell_index)];
        const int* nodes = &umesh->cells_to_nodes[(cells_off)];
        for (int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
          const int next_node_index =
              (nn + 1 == umesh->nnodes_by_cell ? 0 : nn + 1);
          if (umesh->boundary_index[(nodes[(nn)])] != IS_INTERIOR_NODE &&
              umesh->boundary_index[(nodes[(next_node_index)])] !=
                  IS_INTERIOR_NODE) {
            boundary_edge_list[(boundary_edge_index++)] = nodes[(nn)];
            boundary_edge_list[(boundary_edge_index++)] =
                nodes[(next_node_index)];
          }
        }
      }
    }
  }

  find_boundary_normals(umesh, boundary_edge_list);

  return allocated;
}
