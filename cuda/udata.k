// The indirections are messy but quite compact.
#define XZPLANE_FACE_INDEX(ii, jj, kk)                                         \
  (((ii) * (3 * nx * ny + nx + ny)) + (nx * ny) + ((jj) * (2 * nx + 1)) +      \
   (((jj) < ny) ? (2 * (kk) + 1) : (kk)))
#define XYPLANE_FACE_INDEX(ii, jj, kk)                                         \
  (((ii) * (3 * nx * ny + nx + ny)) + ((jj)*nx) + (kk))
#define YZPLANE_FACE_INDEX(ii, jj, kk)                                         \
  (((ii) * (3 * nx * ny + nx + ny)) + (nx * ny) + ((jj) * (2 * nx + 1)) +      \
   (2 * (kk)))

__global__ void faces_to_nodes_offsets_3d(
    const int nfaces, int* faces_to_nodes_offsets, int* faces_cclockwise_cell)
{
  const int ff = blockIdx.x*blockDim.x + threadIdx.x;
  if(ff >= nfaces+1) {
    return;
  }

  faces_to_nodes_offsets[(ff)] = ff * NNODES_BY_FACE;
  if (ff < nfaces) {
    faces_cclockwise_cell[(ff)] = -1;
  }
}

__global__ void cells_to_faces_offsets_3d(
    const int ncells, int* cells_to_faces_offsets)
{
  const int cc = blockIdx.x*blockDim.x + threadIdx.x;
  if(cc >= ncells+1) {
    return;
  }

  cells_to_faces_offsets[cc] = cc * NFACES_BY_CELL;
}

__global__ void nodes_to_faces_offsets_3d(
    const int nnodes, int* nodes_to_faces_offsets)
{
  const int nn = blockIdx.x*blockDim.x + threadIdx.x;
  if(nn >= nnodes+1) {
    return;
  }

  nodes_to_faces_offsets[(nn)] = nn * NFACES_BY_NODE;
}

__global__ void cells_to_nodes_offsets_3d(
    const int ncells, int* cells_to_nodes_offsets)
{
  const int cc = blockIdx.x*blockDim.x + threadIdx.x;
  if(cc >= ncells+1) {
    return;
  }

  cells_to_nodes_offsets[cc] = cc * NNODES_BY_CELL;
}

__global__ void nodes_to_nodes_offsets_3d(
    const int nnodes, int* nodes_to_nodes_offsets)
{
  const int nn = blockIdx.x*blockDim.x + threadIdx.x;
  if(nn >= nnodes+1) {
    return;
  }

  nodes_to_nodes_offsets[nn] = nn * NNODES_BY_NODE;
}

__global__ void faces_to_cells_3d(
    const int nx, const int ny, const int nz, int* faces_to_cells0, int* faces_to_cells1)
{
  const int ii = blockIdx.x*blockDim.x + threadIdx.x;
  if(ii >= nz + 1) {
    return;
  }

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

__global__ void nodes_to_faces_3d(
    const int nx, const int ny, const int nz, 
    const int* nodes_to_faces_offsets, int* nodes_to_faces)
{
  // Determine the connectivity of nodes to faces
  const int node_index = blockIdx.x*blockDim.x + threadIdx.x;
  if(node_index >= (nx+1)*(ny+1)*(nz+1)) {
    return;
  }

  const int ii = node_index / ((nx+1)*(ny+1));
  const int jj = (node_index / (nx+1)) % (ny+1);
  const int kk = node_index % (nx+1);

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

__global__ void cells_to_nodes_3d(
    const int nx, const int ny, const int nz, int* cells_to_nodes)
{
  // Determine the connectivity of nodes to faces
  const int cell_index = blockIdx.x*blockDim.x + threadIdx.x;
  if(cell_index >= nx*ny*nz) {
    return;
  }

  const int ii = cell_index / (nx*ny);
  const int jj = (cell_index / nx) % ny;
  const int kk = cell_index % nx;

  // Simple closed form calculation for the nodes surrounding a cell
  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 0] =
    (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 1] =
    (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 2] =
    (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk + 1);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 3] =
    (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 4] =
    ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 5] =
    ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 6] =
    ((ii + 1) * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk + 1);

  cells_to_nodes[(cell_index * NNODES_BY_CELL) + 7] =
    ((ii + 1) * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);
}

__global__ void faces_to_nodes_3d(
    const int nx, const int ny, const int nz, 
    int* faces_cclockwise_cell, int* faces_to_nodes)
{
  // Determine the connectivity of nodes to faces
  const int ii = blockIdx.x*blockDim.x + threadIdx.x;
  if(ii >= nz+1) {
    return;
  }
  // Add the front faces
  for (int jj = 0; jj < ny; ++jj) {
    for (int kk = 0; kk < nx; ++kk) {
      const int face_index = XYPLANE_FACE_INDEX(ii, jj, kk);

      // On the front face
      faces_to_nodes[(face_index * NNODES_BY_FACE + 0)] =
        (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

      faces_to_nodes[(face_index * NNODES_BY_FACE + 1)] =
        (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

      faces_to_nodes[(face_index * NNODES_BY_FACE + 2)] =
        (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk + 1);

      faces_to_nodes[(face_index * NNODES_BY_FACE + 3)] =
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

          faces_to_nodes[(face_index * NNODES_BY_FACE + 0)] =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 1)] =
            (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 2)] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 3)] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

          if (kk < nx + 1) {
            faces_cclockwise_cell[(face_index)] =
              (ii * nx * ny) + (jj * nx) + (kk);
          }
        }

        if (kk < nx) {
          // On the bottom face
          const int face_index = XZPLANE_FACE_INDEX(ii, jj, kk);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 0)] =
            (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 1)] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 2)] =
            ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);

          faces_to_nodes[(face_index * NNODES_BY_FACE + 3)] =
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

__global__ void cells_to_faces_3d(
    const int nx, const int ny, const int nz, int* cells_to_faces)
{
  // Determine the connectivity of nodes to faces
  const int cell_index = blockIdx.x*blockDim.x + threadIdx.x;
  if(cell_index >= nx*ny*nz) {
    return;
  }

  const int ii = cell_index / (nx*ny);
  const int jj = (cell_index / nx) % ny;
  const int kk = cell_index % nx;

  cells_to_faces[(cell_index*NFACES_BY_CELL + 0)] =
    XYPLANE_FACE_INDEX(ii, jj, kk);

  cells_to_faces[(cell_index*NFACES_BY_CELL + 1)] =
    YZPLANE_FACE_INDEX(ii, jj, kk);

  cells_to_faces[(cell_index*NFACES_BY_CELL + 2)] =
    XZPLANE_FACE_INDEX(ii, jj, kk);

  cells_to_faces[(cell_index*NFACES_BY_CELL + 3)] =
    YZPLANE_FACE_INDEX(ii, jj, kk + 1);

  cells_to_faces[(cell_index*NFACES_BY_CELL + 4)] =
    XZPLANE_FACE_INDEX(ii, jj + 1, kk);

  cells_to_faces[(cell_index*NFACES_BY_CELL + 5)] =
    XYPLANE_FACE_INDEX(ii + 1, jj, kk);
}

__global__ void nodes_to_nodes_3d(
    const int nx, const int ny, const int nz, int* nodes_to_nodes)
{
  // Determine the connectivity of nodes to nodes
  const int node_index = blockIdx.x*blockDim.x + threadIdx.x;
  if(node_index >= (nx+1)*(ny+1)*(nz+1)) {
    return;
  }

  const int ii = node_index / ((nx+1)*(ny+1));
  const int jj = (node_index / (nx+1)) % (ny+1);
  const int kk = node_index % (nx+1);

  // Initialise all to boundary
  for (int nn = 0; nn < NNODES_BY_NODE; ++nn) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + nn] = -1;
  }

  if (kk < nx) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + 0] =
      (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk + 1);
  }
  if (kk > 0) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + 1] =
      (ii * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk - 1);
  }
  if (jj < ny) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + 2] =
      (ii * (nx + 1) * (ny + 1)) + ((jj + 1) * (nx + 1)) + (kk);
  }
  if (jj > 0) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + 3] =
      (ii * (nx + 1) * (ny + 1)) + ((jj - 1) * (nx + 1)) + (kk);
  }
  if (ii < nz) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + 4] =
      ((ii + 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);
  }
  if (ii > 0) {
    nodes_to_nodes[node_index*NNODES_BY_NODE + 5] =
      ((ii - 1) * (nx + 1) * (ny + 1)) + (jj * (nx + 1)) + (kk);
  }
}

__global__ void nodes_3d(
    const int nx, const int ny, const int nz, 
    const double* edgex, const double* edgey, const double* edgez,
    double* nodes_x0, double* nodes_y0, double* nodes_z0)
{
  // Determine the connectivity of nodes to nodes
  const int node_index = blockIdx.x*blockDim.x + threadIdx.x;
  if(node_index >= (nx+1)*(ny+1)*(nz+1)) {
    return;
  }

  const int ii = node_index / ((nx+1)*(ny+1));
  const int jj = (node_index / (nx+1)) % (ny+1);
  const int kk = node_index % (nx+1);

  nodes_z0[node_index] = edgez[ii];
  nodes_y0[node_index] = edgey[jj];
  nodes_x0[node_index] = edgex[kk];
}

__global__ void nodes_to_cells_3d(
    const int nx, const int ny, const int nz, int* nodes_to_cells)
{
  // Determine the connectivity of nodes to cells
  const int node_index = blockIdx.x*blockDim.x + threadIdx.x;
  if(node_index >= (nx+1)*(ny+1)*(nz+1)) {
    return;
  }

  const int ii = node_index / ((nx+1)*(ny+1));
  const int jj = (node_index / (nx+1)) % (ny+1);
  const int kk = node_index % (nx+1);

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
