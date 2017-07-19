#include "../comms.h"
#include "../mesh.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(const int nx, const int ny, Mesh* mesh, double* arr,
                        const int invert, const int pack) {
  START_PROFILING(&comms_profile);

  const int pad = mesh->pad;
  int* neighbours = mesh->neighbours;

#ifdef MPI
  double* east_buffer_out = mesh->east_buffer_out;
  double* east_buffer_in = mesh->east_buffer_in;
  double* west_buffer_out = mesh->west_buffer_out;
  double* west_buffer_in = mesh->west_buffer_in;
  double* north_buffer_out = mesh->north_buffer_out;
  double* north_buffer_in = mesh->north_buffer_in;
  double* south_buffer_out = mesh->south_buffer_out;
  double* south_buffer_in = mesh->south_buffer_in;

  int nmessages = 0;
  if (pack) {
    // Pack east and west
    if (neighbours[EAST] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          east_buffer_out[(ii - pad) * pad + dd] =
              arr[(ii * nx) + (nx - 2 * pad + dd)];
        }
      }

      copy_buffer(pad * ny, &east_buffer_out, &east_buffer_out, RECV);
      non_block_send(east_buffer_out, (ny - 2 * pad) * pad, neighbours[EAST], 2,
                     nmessages++);
      non_block_recv(east_buffer_in, (ny - 2 * pad) * pad, neighbours[EAST], 3,
                     nmessages++);
    }

    if (neighbours[WEST] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          west_buffer_out[(ii - pad) * pad + dd] = arr[(ii * nx) + (pad + dd)];
        }
      }

      copy_buffer(pad * ny, &west_buffer_out, &west_buffer_out, RECV);
      non_block_send(west_buffer_out, (ny - 2 * pad) * pad, neighbours[WEST], 3,
                     nmessages++);
      non_block_recv(west_buffer_in, (ny - 2 * pad) * pad, neighbours[WEST], 2,
                     nmessages++);
    }

    // Pack north and south
    if (neighbours[NORTH] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          north_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
              arr[(ny - 2 * pad + dd) * nx + jj];
        }
      }

      copy_buffer(nx * pad, &north_buffer_out, &north_buffer_out, RECV);
      non_block_send(north_buffer_out, (nx - 2 * pad) * pad, neighbours[NORTH],
                     1, nmessages++);
      non_block_recv(north_buffer_in, (nx - 2 * pad) * pad, neighbours[NORTH],
                     0, nmessages++);
    }

    if (neighbours[SOUTH] != EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          south_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
              arr[(pad + dd) * nx + jj];
        }
      }

      copy_buffer(nx * pad, &south_buffer_out, &south_buffer_out, RECV);
      non_block_send(south_buffer_out, (nx - 2 * pad) * pad, neighbours[SOUTH],
                     0, nmessages++);
      non_block_recv(south_buffer_in, (nx - 2 * pad) * pad, neighbours[SOUTH],
                     1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if (neighbours[WEST] != EDGE) {
      copy_buffer(pad * ny, &west_buffer_in, &west_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          arr[ii * nx + dd] = west_buffer_in[(ii - pad) * pad + dd];
        }
      }
    }

    if (neighbours[EAST] != EDGE) {
      copy_buffer(pad * ny, &east_buffer_in, &east_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          arr[ii * nx + (nx - pad + dd)] =
              east_buffer_in[(ii - pad) * pad + dd];
        }
      }
    }

    // Unpack north and south
    if (neighbours[NORTH] != EDGE) {
      copy_buffer(nx * pad, &north_buffer_in, &north_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          arr[(ny - pad + dd) * nx + jj] =
              north_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
        }
      }
    }

    if (neighbours[SOUTH] != EDGE) {
      copy_buffer(nx * pad, &south_buffer_in, &south_buffer_in, SEND);

#pragma omp target teams distribute parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          arr[dd * nx + jj] = south_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
        }
      }
    }
  }
#endif

  // Perform the boundary reflections, potentially with the data updated from
  // neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if (neighbours[NORTH] == EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int dd = 0; dd < pad; ++dd) {
      for (int jj = pad; jj < nx - pad; ++jj) {
        arr[(ny - pad + dd) * nx + jj] =
            y_inversion_coeff * arr[(ny - 1 - pad - dd) * nx + jj];
      }
    }
  }
  // reflect at the south
  if (neighbours[SOUTH] == EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int dd = 0; dd < pad; ++dd) {
      for (int jj = pad; jj < nx - pad; ++jj) {
        arr[(pad - 1 - dd) * nx + jj] =
            y_inversion_coeff * arr[(pad + dd) * nx + jj];
      }
    }
  }
  // reflect at the east
  if (neighbours[EAST] == EDGE) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int ii = pad; ii < ny - pad; ++ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (nx - pad + dd)] =
            x_inversion_coeff * arr[ii * nx + (nx - 1 - pad - dd)];
      }
    }
  }
  if (neighbours[WEST] == EDGE) {
// reflect at the west
#pragma omp target teams distribute parallel for collapse(2)
    for (int ii = pad; ii < ny - pad; ++ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (pad - 1 - dd)] =
            x_inversion_coeff * arr[ii * nx + (pad + dd)];
      }
    }
  }
  STOP_PROFILING(&comms_profile, __func__);
}

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect(const int nnodes, const int* boundary_index,
                                 const int* boundary_type,
                                 const double* boundary_normal_x,
                                 const double* boundary_normal_y,
                                 double* velocity_x, double* velocity_y) {
#pragma omp target teams distribute parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int index = boundary_index[(nn)];
    if (index == IS_INTERIOR_NODE) {
      continue;
    }

    if (boundary_type[(index)] == IS_BOUNDARY) {
      // Project the velocity onto the face direction
      const double boundary_parallel_x = boundary_normal_y[(index)];
      const double boundary_parallel_y = -boundary_normal_x[(index)];
      const double vel_dot_parallel = (velocity_x[(nn)] * boundary_parallel_x +
                                       velocity_y[(nn)] * boundary_parallel_y);
      velocity_x[(nn)] = boundary_parallel_x * vel_dot_parallel;
      velocity_y[(nn)] = boundary_parallel_y * vel_dot_parallel;
    } else if (boundary_type[(index)] == IS_FIXED) {
      velocity_x[(nn)] = 0.0;
      velocity_y[(nn)] = 0.0;
    }
  }
}
