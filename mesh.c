#include <stdlib.h>
#include <assert.h>
#include "shared.h"
#include "mesh.h"
#include "params.h"

// Initialise the mesh describing variables
void initialise_mesh_2d(
    Mesh* mesh)
{
  allocate_data(&mesh->edgex, (mesh->local_nx+1));
  allocate_data(&mesh->edgey, (mesh->local_ny+1));
  allocate_data(&mesh->edgedx, (mesh->local_nx+1));
  allocate_data(&mesh->edgedy, (mesh->local_ny+1));
  allocate_data(&mesh->celldx, (mesh->local_nx+1));
  allocate_data(&mesh->celldy, (mesh->local_ny+1));

  mesh_data_init_2d(
      mesh->local_nx, mesh->local_ny, mesh->global_nx, mesh->global_ny,
      mesh->pad, mesh->x_off, mesh->y_off, mesh->width, mesh->height, mesh->edgex, 
      mesh->edgey, mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  allocate_data(&mesh->north_buffer_out, (mesh->local_nx+1)*mesh->pad);
  allocate_data(&mesh->east_buffer_out, (mesh->local_ny+1)*mesh->pad);
  allocate_data(&mesh->south_buffer_out, (mesh->local_nx+1)*mesh->pad);
  allocate_data(&mesh->west_buffer_out, (mesh->local_ny+1)*mesh->pad);
  allocate_data(&mesh->north_buffer_in, (mesh->local_nx+1)*mesh->pad);
  allocate_data(&mesh->east_buffer_in, (mesh->local_ny+1)*mesh->pad);
  allocate_data(&mesh->south_buffer_in, (mesh->local_nx+1)*mesh->pad);
  allocate_data(&mesh->west_buffer_in, (mesh->local_ny+1)*mesh->pad);

  allocate_host_data(&mesh->h_north_buffer_out, (mesh->local_nx+1)*mesh->pad);
  allocate_host_data(&mesh->h_east_buffer_out, (mesh->local_ny+1)*mesh->pad);
  allocate_host_data(&mesh->h_south_buffer_out, (mesh->local_nx+1)*mesh->pad);
  allocate_host_data(&mesh->h_west_buffer_out, (mesh->local_ny+1)*mesh->pad);
  allocate_host_data(&mesh->h_north_buffer_in, (mesh->local_nx+1)*mesh->pad);
  allocate_host_data(&mesh->h_east_buffer_in, (mesh->local_ny+1)*mesh->pad);
  allocate_host_data(&mesh->h_south_buffer_in, (mesh->local_nx+1)*mesh->pad);
  allocate_host_data(&mesh->h_west_buffer_in, (mesh->local_ny+1)*mesh->pad);
}

// Initialise the mesh describing variables
void initialise_mesh_3d(
    Mesh* mesh)
{
  allocate_data(&mesh->edgex, (mesh->local_nx+1));
  allocate_data(&mesh->edgey, (mesh->local_ny+1));
  allocate_data(&mesh->edgez, (mesh->local_nz+1));
  allocate_data(&mesh->edgedx, (mesh->local_nx+1));
  allocate_data(&mesh->edgedy, (mesh->local_ny+1));
  allocate_data(&mesh->edgedz, (mesh->local_nz+1));
  allocate_data(&mesh->celldx, (mesh->local_nx+1));
  allocate_data(&mesh->celldy, (mesh->local_ny+1));
  allocate_data(&mesh->celldz, (mesh->local_nz+1));

  mesh_data_init_3d(
      mesh->local_nx, mesh->local_ny, mesh->local_nz, 
      mesh->global_nx, mesh->global_ny, mesh->local_nz,
      mesh->pad, mesh->x_off, mesh->y_off, mesh->z_off,
      mesh->width, mesh->height, mesh->depth,
      mesh->edgex, mesh->edgey, mesh->edgez, 
      mesh->edgedx, mesh->edgedy, mesh->edgedz, 
      mesh->celldx, mesh->celldy, mesh->celldz);

  allocate_data(&mesh->north_buffer_out, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->east_buffer_out, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->south_buffer_out, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->west_buffer_out, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->front_buffer_out, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
  allocate_data(&mesh->back_buffer_out, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
  allocate_data(&mesh->north_buffer_in, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->east_buffer_in, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->south_buffer_in, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->west_buffer_in, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_data(&mesh->front_buffer_in, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
  allocate_data(&mesh->back_buffer_in, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);

  allocate_host_data(&mesh->h_north_buffer_out, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_east_buffer_out, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_south_buffer_out, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_west_buffer_out, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_front_buffer_out, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
  allocate_host_data(&mesh->h_back_buffer_out, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
  allocate_host_data(&mesh->h_north_buffer_in, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_east_buffer_in, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_south_buffer_in, (mesh->local_nx+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_west_buffer_in, (mesh->local_ny+1)*(mesh->local_nz+1)*mesh->pad);
  allocate_host_data(&mesh->h_front_buffer_in, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
  allocate_host_data(&mesh->h_back_buffer_in, (mesh->local_nx+1)*(mesh->local_ny+1)*mesh->pad);
}

// We need this data to be able to initialise any data arrays etc
void read_unstructured_mesh_sizes(
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
}

// Reads an unstructured mesh from an input file
size_t read_unstructured_mesh(
    UnstructuredMesh* umesh, double** variables)
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

  int* boundary_edge_list;
  int boundary_edge_index = 0;
  allocated += allocate_int_data(&boundary_edge_list, umesh->nboundary_cells*2);
  allocated += allocate_data(&umesh->boundary_normal_x, umesh->nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, umesh->nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, umesh->nboundary_cells);

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

