#include "../umesh.h"
#include "../shared.h"

// The indirections are messy but quite compact.
#define XZPLANE_FACE_INDEX(ii, jj, kk)                                         \
  (((ii) * (3 * nx * ny + nx + ny)) + (nx * ny) + ((jj) * (2 * nx + 1)) +      \
   (((jj) < ny) ? (2 * (kk) + 1) : (kk)))
#define XYPLANE_FACE_INDEX(ii, jj, kk)                                         \
  (((ii) * (3 * nx * ny + nx + ny)) + ((jj)*nx) + (kk))
#define YZPLANE_FACE_INDEX(ii, jj, kk)                                         \
  (((ii) * (3 * nx * ny + nx + ny)) + (nx * ny) + ((jj) * (2 * nx + 1)) +      \
   (2 * (kk)))

// Initialises the offsets between faces and nodes
void init_faces_to_nodes_offsets_3d(UnstructuredMesh* umesh) {

  const int nfaces = umesh->nfaces;
  int* faces_to_nodes_offsets = umesh->faces_to_nodes_offsets;
  int* faces_cclockwise_cell = umesh->faces_cclockwise_cell;

#pragma acc parallel \
  present(faces_cclockwise_cell[:umesh->nfaces])
#pragma acc loop independent
  for (int ff = 0; ff < nfaces + 1; ++ff) {
    faces_to_nodes_offsets[(ff)] = ff * NNODES_BY_FACE;
    if (ff < nfaces) {
      faces_cclockwise_cell[(ff)] = -1;
    }
  }
}

// Initialises the offsets between cells and faces
void init_cells_to_faces_offsets_3d(UnstructuredMesh* umesh) {

  const int ncells = umesh->ncells;
  int* cells_to_faces_offsets = umesh->cells_to_faces_offsets;

#pragma acc parallel
#pragma acc loop independent
  for (int cc = 0; cc < ncells + 1; ++cc) {
    cells_to_faces_offsets[(cc)] = cc * NFACES_BY_CELL;
  }
}

// Initialises the offsets between nodes and faces
void init_nodes_to_faces_offsets_3d(UnstructuredMesh* umesh) {

  const int nnodes = umesh->nnodes;
  int* nodes_to_faces_offsets = umesh->nodes_to_faces_offsets;
#pragma acc parallel
#pragma acc loop independent
  for (int nn = 0; nn < nnodes + 1; ++nn) {
    nodes_to_faces_offsets[(nn)] = nn * NFACES_BY_NODE;
  }
}

// Initialises the connectivity between faces and cells
void init_faces_to_cells_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh) {

  int* faces_to_cells0 = umesh->faces_to_cells0;
  int* faces_to_cells1 = umesh->faces_to_cells1;

#pragma acc parallel \
  present(faces_to_cells0[:umesh->nfaces])\
  present(faces_to_cells1[:umesh->nfaces])
#pragma acc loop independent
  for (int ii = 0; ii < nz + 1; ++ii) {
    // All front oriented faces
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int face_index = XYPLANE_FACE_INDEX(ii, jj, kk);
        faces_to_cells0[(face_index)] =
            (ii < nz) ? (ii * nx * ny) + (jj * nx) + (kk) : -1;
        faces_to_cells1[(face_index)] =
            (ii > 0) ? ((ii - 1) * nx * ny) + (jj * nx) + (kk) : -1;
      }
    }
    if (ii < nz) {
      // Side oriented faces
      for (int jj = 0; jj < ny; ++jj) {
        for (int kk = 0; kk < nx + 1; ++kk) {
          const int face_index = YZPLANE_FACE_INDEX(ii, jj, kk);
          faces_to_cells0[(face_index)] =
              (kk < nx) ? (ii * nx * ny) + (jj * nx) + (kk) : -1;
          faces_to_cells1[(face_index)] =
              (kk > 0) ? (ii * nx * ny) + (jj * nx) + (kk - 1) : -1;
        }
      }
    }
    if (ii < nz) {
      // Bottom oriented faces
      for (int jj = 0; jj < ny + 1; ++jj) {
        for (int kk = 0; kk < nx; ++kk) {
          const int face_index = XZPLANE_FACE_INDEX(ii, jj, kk);
          faces_to_cells0[(face_index)] =
              (jj < ny) ? (ii * nx * ny) + (jj * nx) + (kk) : -1;
          faces_to_cells1[(face_index)] =
              (jj > 0) ? (ii * nx * ny) + ((jj - 1) * nx) + (kk) : -1;
        }
      }
    }
  }
}

// Initialises the connectivity between nodes and faces
void init_nodes_to_faces_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh) {

  int* nodes_to_faces_offsets = umesh->nodes_to_faces_offsets;
  int* nodes_to_faces = umesh->nodes_to_faces;

// Determine the connectivity of nodes to faces
#pragma acc parallel \
  present(nodes_to_faces[:umesh->nnodes*NFACES_BY_NODE])\
  present(nodes_to_faces_offsets[:umesh->nnodes])
#pragma acc loop independent
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {
        const int node_index =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);
        const int node_to_faces_off =
            nodes_to_faces_offsets[(node_index)];

        nodes_to_faces[(node_to_faces_off + 0)] =
            (ii < nz && kk < nx) ? XZPLANE_FACE_INDEX(ii, jj, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 1)] =
            (jj < ny && kk < nx) ? XYPLANE_FACE_INDEX(ii, jj, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 2)] =
            (ii < nz && jj < ny) ? YZPLANE_FACE_INDEX(ii, jj, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 3)] =
            (ii < nz && kk > 0) ? XZPLANE_FACE_INDEX(ii, jj, kk - 1) : -1;

        nodes_to_faces[(node_to_faces_off + 4)] =
            (jj < ny && kk > 0) ? XYPLANE_FACE_INDEX(ii, jj, kk - 1) : -1;

        nodes_to_faces[(node_to_faces_off + 5)] =
            (ii > 0 && kk < nx) ? XZPLANE_FACE_INDEX(ii - 1, jj, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 6)] =
            (ii > 0 && jj < ny) ? YZPLANE_FACE_INDEX(ii - 1, jj, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 7)] =
            (ii > 0 && kk > 0) ? XZPLANE_FACE_INDEX(ii - 1, jj, kk - 1) : -1;

        nodes_to_faces[(node_to_faces_off + 8)] =
            (jj > 0 && kk < nx) ? XYPLANE_FACE_INDEX(ii, jj - 1, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 9)] =
            (jj > 0 && kk > 0) ? XYPLANE_FACE_INDEX(ii, jj - 1, kk - 1) : -1;

        nodes_to_faces[(node_to_faces_off + 10)] =
            (ii > 0 && jj > 0) ? YZPLANE_FACE_INDEX(ii - 1, jj - 1, kk) : -1;

        nodes_to_faces[(node_to_faces_off + 11)] =
            (ii < nz && jj > 0) ? YZPLANE_FACE_INDEX(ii, jj - 1, kk) : -1;
      }
    }
  }
}

// Initialises the cells to nodes connectivity
void init_cells_to_nodes_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh) {

  const int ncells = umesh->ncells;
  int* cells_to_nodes_offsets = umesh->cells_to_nodes_offsets;
  int* cells_to_nodes = umesh->cells_to_nodes;

#pragma acc parallel
#pragma acc loop independent
  for (int cc = 0; cc < ncells + 1; ++cc) {
    cells_to_nodes_offsets[(cc)] = cc * NNODES_BY_CELL;
  }

#pragma acc parallel \
  present(cells_to_nodes[:ncells*NNODES_BY_CELL])
#pragma acc loop independent
  for (int ii = 0; ii < nz; ++ii) {
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int index = (ii * nx * ny) + (jj * nx) + (kk);

        // Simple closed form calculation for the nodes surrounding a cell
        cells_to_nodes[(index * NNODES_BY_CELL) + 0] =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        cells_to_nodes[(index * NNODES_BY_CELL) + 1] =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

        cells_to_nodes[(index * NNODES_BY_CELL) + 2] =
            (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk + 1);

        cells_to_nodes[(index * NNODES_BY_CELL) + 3] =
            (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

        cells_to_nodes[(index * NNODES_BY_CELL) + 4] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        cells_to_nodes[(index * NNODES_BY_CELL) + 5] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

        cells_to_nodes[(index * NNODES_BY_CELL) + 6] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk + 1);

        cells_to_nodes[(index * NNODES_BY_CELL) + 7] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);
      }
    }
  }
}

// Initialises the connectivity between faces and nodes
void init_faces_to_nodes_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh) {

  int* faces_to_nodes_offsets = umesh->faces_to_nodes_offsets;
  int* faces_to_nodes = umesh->faces_to_nodes;
  int* faces_cclockwise_cell = umesh->faces_cclockwise_cell;

// Connectivity of faces to nodes, the nodes are stored in a counter-clockwise
// ordering around the face
#pragma acc parallel \
  present(faces_cclockwise_cell[:umesh->nfaces])\
  present(faces_to_nodes[:umesh->nfaces*NNODES_BY_FACE])\
  present(faces_to_nodes_offsets[:umesh->nfaces])
#pragma acc loop independent
  for (int ii = 0; ii < nz + 1; ++ii) {
    // Add the front faces
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int face_index = XYPLANE_FACE_INDEX(ii, jj, kk);
        const int face_to_node_off =
            faces_to_nodes_offsets[(face_index)];

        // On the front face
        faces_to_nodes[(face_to_node_off + 0)] =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        faces_to_nodes[(face_to_node_off + 1)] =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

        faces_to_nodes[(face_to_node_off + 2)] =
            (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk + 1);

        faces_to_nodes[(face_to_node_off + 3)] =
            (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

        if (ii < nz + 1) {
          faces_cclockwise_cell[(face_index)] =
              (ii * nx * ny) + (jj * nx) + (kk);
        }
      }
    }

    if (ii < nz) {
      for (int jj = 0; jj < ny + 1; ++jj) {
        for (int kk = 0; kk < nx + 1; ++kk) {
          if (jj < ny) {
            // On the left face
            const int face_index = YZPLANE_FACE_INDEX(ii, jj, kk);
            const int face_to_node_off =
                faces_to_nodes_offsets[(face_index)];

            faces_to_nodes[(face_to_node_off + 0)] =
                (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

            faces_to_nodes[(face_to_node_off + 1)] =
                (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

            faces_to_nodes[(face_to_node_off + 2)] =
                ((ii + 1) * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

            faces_to_nodes[(face_to_node_off + 3)] =
                ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

            if (kk < nx + 1) {
              faces_cclockwise_cell[(face_index)] =
                  (ii * nx * ny) + (jj * nx) + (kk);
            }
          }

          if (kk < nx) {
            // On the bottom face
            const int face_index = XZPLANE_FACE_INDEX(ii, jj, kk);
            const int face_to_node_off =
                faces_to_nodes_offsets[(face_index)];

            faces_to_nodes[(face_to_node_off + 0)] =
                (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

            faces_to_nodes[(face_to_node_off + 1)] =
                ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

            faces_to_nodes[(face_to_node_off + 2)] =
                ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

            faces_to_nodes[(face_to_node_off + 3)] =
                (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

            if (jj < ny + 1) {
              faces_cclockwise_cell[(face_index)] =
                  (ii * nx * ny) + (jj * nx) + (kk);
            }
          }
        }
      }
    }
  }
}

// Initialises the connectivity between cells and faces
void init_cells_to_faces_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh) {

  int* cells_to_faces_offsets = umesh->cells_to_faces_offsets;
  int* cells_to_faces = umesh->cells_to_faces;

#pragma acc parallel \
  present(cells_to_faces[:umesh->ncells*NFACES_BY_CELL])\
  present(cells_to_faces_offsets[:umesh->ncells])
#pragma acc loop independent
  for (int ii = 0; ii < nz; ++ii) {
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int cell_index = (ii * nx * ny) + (jj * nx) + (kk);
        const int cell_to_faces_off =
            cells_to_faces_offsets[(cell_index)];

        cells_to_faces[(cell_to_faces_off + 0)] =
            XYPLANE_FACE_INDEX(ii, jj, kk);

        cells_to_faces[(cell_to_faces_off + 1)] =
            YZPLANE_FACE_INDEX(ii, jj, kk);

        cells_to_faces[(cell_to_faces_off + 2)] =
            XZPLANE_FACE_INDEX(ii, jj, kk);

        cells_to_faces[(cell_to_faces_off + 3)] =
            YZPLANE_FACE_INDEX(ii, jj, kk + 1);

        cells_to_faces[(cell_to_faces_off + 4)] =
            XZPLANE_FACE_INDEX(ii, jj + 1, kk);

        cells_to_faces[(cell_to_faces_off + 5)] =
            XYPLANE_FACE_INDEX(ii + 1, jj, kk);
      }
    }
  }
}

// Initialises the list of nodes to nodes
void init_nodes_to_nodes_3d(const int nx, const int ny, const int nz,
                            UnstructuredMesh* umesh) {

  const int nnodes = umesh->nnodes;
  int* nodes_to_nodes_offsets = umesh->nodes_to_nodes_offsets;
  int* nodes_to_nodes = umesh->nodes_to_nodes;

// Prefix sum to convert the counts to offsets
#pragma acc parallel
#pragma acc loop independent
  for (int nn = 0; nn < nnodes + 1; ++nn) {
    nodes_to_nodes_offsets[(nn)] = nn * NNODES_BY_NODE;
  }

// Initialise the offsets and list of nodes to cells, counter-clockwise order
#pragma acc parallel \
  present(nodes_to_nodes[:umesh->nnodes*NNODES_BY_NODE])\
  present(nodes_to_nodes_offsets[:umesh->nnodes])
#pragma acc loop independent
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {
        const int node_index =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        int off = nodes_to_nodes_offsets[(node_index)];

        // Initialise all to boundary
        for (int nn = 0; nn < NNODES_BY_NODE; ++nn) {
          nodes_to_nodes[(off + nn)] = -1;
        }

        if (kk < nx) {
          nodes_to_nodes[(off++)] =
              (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);
        }
        if (kk > 0) {
          nodes_to_nodes[(off++)] =
              (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk - 1);
        }
        if (jj < ny) {
          nodes_to_nodes[(off++)] =
              (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);
        }
        if (jj > 0) {
          nodes_to_nodes[(off++)] =
              (ii * (nx + 1) * (ny + 1)) + ((jj - 1) * (nx + 1)) + (kk);
        }
        if (ii < nz) {
          nodes_to_nodes[(off++)] =
              ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);
        }
        if (ii > 0) {
          nodes_to_nodes[(off++)] =
              ((ii - 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);
        }
      }
    }
  }
}

// Initialises the list of nodes to cells
void init_nodes_to_cells_3d(const int nx, const int ny, const int nz,
                            Mesh* mesh, UnstructuredMesh* umesh) {

  int* nodes_to_cells = umesh->nodes_to_cells;
  double* nodes_x0 = umesh->nodes_x0;
  double* nodes_y0 = umesh->nodes_y0;
  double* nodes_z0 = umesh->nodes_z0;
  double* edgex = mesh->edgex;
  double* edgey = mesh->edgey;
  double* edgez = mesh->edgez;

// Initialise the offsets and list of nodes to cells, counter-clockwise order
#pragma acc parallel \
  present(nodes_x0[:umesh->nnodes])\
  present(nodes_y0[:umesh->nnodes])\
  present(nodes_z0[:umesh->nnodes])\
  present(edgex[:nx+1])\
  present(edgey[:ny+1])\
  present(edgez[:nz+1])
#pragma acc loop independent
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {
        const int node_index =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        nodes_z0[(node_index)] = edgez[(ii)];
        nodes_y0[(node_index)] = edgey[(jj)];
        nodes_x0[(node_index)] = edgex[(kk)];
      }
    }
  }

// Override the original initialisation of the nodes_to_cells layout
#pragma acc parallel \
  present(nodes_to_cells[:umesh->nnodes*NCELLS_BY_NODE])
#pragma acc loop independent
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {
        const int node_index =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        // Fill in all of the cells that surround a node
        nodes_to_cells[node_index * NCELLS_BY_NODE + 0] =
            (ii > 0 && jj > 0 && kk > 0)
                ? ((ii - 1) * nx * ny) + ((jj - 1) * nx) + (kk - 1)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 1] =
            (ii > 0 && jj > 0 && kk < nx)
                ? ((ii - 1) * nx * ny) + ((jj - 1) * nx) + (kk)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 2] =
            (ii > 0 && jj < ny && kk > 0)
                ? ((ii - 1) * nx * ny) + (jj * nx) + (kk - 1)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 3] =
            (ii > 0 && jj < ny && kk < nx)
                ? ((ii - 1) * nx * ny) + (jj * nx) + (kk)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 4] =
            (ii < nz && jj > 0 && kk > 0)
                ? (ii * nx * ny) + ((jj - 1) * nx) + (kk - 1)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 5] =
            (ii < nz && jj > 0 && kk < nx)
                ? (ii * nx * ny) + ((jj - 1) * nx) + (kk)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 6] =
            (ii < nz && jj < ny && kk > 0)
                ? (ii * nx * ny) + (jj * nx) + (kk - 1)
                : -1;
        nodes_to_cells[node_index * NCELLS_BY_NODE + 7] =
            (ii < nz && jj < ny && kk < nx) ? (ii * nx * ny) + (jj * nx) + (kk)
                                            : -1;
      }
    }
  }

#if 0
// Determine the count of cells by nodes
#pragma acc parallel
#pragma acc loop independent
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {
        int off = 0;
        off += (ii > 0 && jj > 0 && kk > 0);
        off += (ii > 0 && jj > 0 && kk < nx);
        off += (ii > 0 && jj < ny && kk > 0);
        off += (ii > 0 && jj < ny && kk < nx);
        off += (ii < nz && jj > 0 && kk > 0);
        off += (ii < nz && jj > 0 && kk < nx);
        off += (ii < nz && jj < ny && kk > 0);
        off += (ii < nz && jj < ny && kk < nx);

        const int node_index =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);
        umesh->nodes_to_cells_offsets[(node_index + 1)] = off;
      }
    }
  }

  // Perform prefix sum of the counts to get offsets (in serial for now)
  for (int nn = 0; nn < umesh->nnodes; ++nn) {
    umesh->nodes_to_cells_offsets[(nn + 1)] +=
        umesh->nodes_to_cells_offsets[(nn)];
  }

// Initialise the offsets and list of nodes to cells, counter-clockwise order
#pragma acc parallel
#pragma acc loop independent
  for (int ii = 0; ii < (nz + 1); ++ii) {
    for (int jj = 0; jj < (ny + 1); ++jj) {
      for (int kk = 0; kk < (nx + 1); ++kk) {
        const int node_index =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

        umesh->nodes_z0[(node_index)] = mesh->edgez[(ii)];
        umesh->nodes_y0[(node_index)] = mesh->edgey[(jj)];
        umesh->nodes_x0[(node_index)] = mesh->edgex[(kk)];

        int off = umesh->nodes_to_cells_offsets[(node_index)];

        // Fill in all of the cells that surround a node
        // NOTE: The order of the statements is important for data layout
        if (ii > 0 && jj > 0 && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * nx * ny) + ((jj - 1) * nx) + (kk - 1);
        }
        if (ii > 0 && jj > 0 && kk < nx) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * nx * ny) + ((jj - 1) * nx) + (kk);
        }
        if (ii > 0 && jj < ny && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * nx * ny) + (jj * nx) + (kk - 1);
        }
        if (ii > 0 && jj < ny && kk < nx) {
          umesh->nodes_to_cells[(off++)] =
              ((ii - 1) * nx * ny) + (jj * nx) + (kk);
        }
        if (ii < nz && jj > 0 && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              (ii * nx * ny) + ((jj - 1) * nx) + (kk - 1);
        }
        if (ii < nz && jj > 0 && kk < nx) {
          umesh->nodes_to_cells[(off++)] =
              (ii * nx * ny) + ((jj - 1) * nx) + (kk);
        }
        if (ii < nz && jj < ny && kk > 0) {
          umesh->nodes_to_cells[(off++)] =
              (ii * nx * ny) + (jj * nx) + (kk - 1);
        }
        if (ii < nz && jj < ny && kk < nx) {
          umesh->nodes_to_cells[(off++)] = (ii * nx * ny) + (jj * nx) + (kk);
        }
      }
    }
  }
#endif // if 0
}

// Initialises the boundary normals
void init_boundary_normals_3d(const int nx, const int ny, const int nz,
                              UnstructuredMesh* umesh) {

  int* boundary_type = umesh->boundary_type;
  double* boundary_normal_x = umesh->boundary_normal_x;
  double* boundary_normal_y = umesh->boundary_normal_y;
  double* boundary_normal_z = umesh->boundary_normal_z;
  int* boundary_index = umesh->boundary_index;

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
            // NORMALS FROM THE CONSTRUCTION OF THE MESH, ALTHOUGH WE WILL NEED A
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

#pragma acc update device(boundary_index[:umesh->nnodes])
#pragma acc update device(boundary_normal_x[:umesh->nboundary_nodes])
#pragma acc update device(boundary_normal_y[:umesh->nboundary_nodes])
#pragma acc update device(boundary_normal_z[:umesh->nboundary_nodes])
#pragma acc update device(boundary_type[:umesh->nboundary_nodes])
}
