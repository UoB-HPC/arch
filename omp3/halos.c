#include "../comms.h"
#include "../mesh.h"
#include "../umesh.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(const int nx, const int ny, Mesh* mesh, double* arr,
                        const int invert, const int pack) {
  START_PROFILING(&comms_profile);

  const int pad = mesh->pad;
  int* neighbours = mesh->neighbours;

#ifdef MPI
  int nmessages = 0;

  if (pack) {
    // Pack east and west
    if (neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          mesh->east_buffer_out[(ii - pad) * pad + dd] =
              arr[(ii * nx) + (nx - 2 * pad + dd)];
        }
      }

      non_block_send(mesh->east_buffer_out, (ny - 2 * pad) * pad,
                     neighbours[EAST], 2, nmessages++);
      non_block_recv(mesh->east_buffer_in, (ny - 2 * pad) * pad,
                     neighbours[EAST], 3, nmessages++);
    }

    if (neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          mesh->west_buffer_out[(ii - pad) * pad + dd] =
              arr[(ii * nx) + (pad + dd)];
        }
      }

      non_block_send(mesh->west_buffer_out, (ny - 2 * pad) * pad,
                     neighbours[WEST], 3, nmessages++);
      non_block_recv(mesh->west_buffer_in, (ny - 2 * pad) * pad,
                     neighbours[WEST], 2, nmessages++);
    }

    // Pack north and south
    if (neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          mesh->north_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
              arr[(ny - 2 * pad + dd) * nx + jj];
        }
      }

      non_block_send(mesh->north_buffer_out, (nx - 2 * pad) * pad,
                     neighbours[NORTH], 1, nmessages++);
      non_block_recv(mesh->north_buffer_in, (nx - 2 * pad) * pad,
                     neighbours[NORTH], 0, nmessages++);
    }

    if (neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          mesh->south_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
              arr[(pad + dd) * nx + jj];
        }
      }

      non_block_send(mesh->south_buffer_out, (nx - 2 * pad) * pad,
                     neighbours[SOUTH], 0, nmessages++);
      non_block_recv(mesh->south_buffer_in, (nx - 2 * pad) * pad,
                     neighbours[SOUTH], 1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if (neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          arr[ii * nx + dd] = mesh->west_buffer_in[(ii - pad) * pad + dd];
        }
      }
    }

    if (neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int ii = pad; ii < ny - pad; ++ii) {
        for (int dd = 0; dd < pad; ++dd) {
          arr[ii * nx + (nx - pad + dd)] =
              mesh->east_buffer_in[(ii - pad) * pad + dd];
        }
      }
    }

    // Unpack north and south
    if (neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          arr[(ny - pad + dd) * nx + jj] =
              mesh->north_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
        }
      }
    }

    if (neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for (int dd = 0; dd < pad; ++dd) {
        for (int jj = pad; jj < nx - pad; ++jj) {
          arr[dd * nx + jj] =
              mesh->south_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
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
#pragma omp parallel for collapse(2)
    for (int dd = 0; dd < pad; ++dd) {
      for (int jj = pad; jj < nx - pad; ++jj) {
        arr[(ny - pad + dd) * nx + jj] =
            y_inversion_coeff * arr[(ny - 1 - pad - dd) * nx + jj];
      }
    }
  }
  // reflect at the south
  if (neighbours[SOUTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for (int dd = 0; dd < pad; ++dd) {
      for (int jj = pad; jj < nx - pad; ++jj) {
        arr[(pad - 1 - dd) * nx + jj] =
            y_inversion_coeff * arr[(pad + dd) * nx + jj];
      }
    }
  }
  // reflect at the east
  if (neighbours[EAST] == EDGE) {
#pragma omp parallel for collapse(2)
    for (int ii = pad; ii < ny - pad; ++ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (nx - pad + dd)] =
            x_inversion_coeff * arr[ii * nx + (nx - 1 - pad - dd)];
      }
    }
  }
  if (neighbours[WEST] == EDGE) {
// reflect at the west
#pragma omp parallel for collapse(2)
    for (int ii = pad; ii < ny - pad; ++ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (pad - 1 - dd)] =
            x_inversion_coeff * arr[ii * nx + (pad + dd)];
      }
    }
  }
  STOP_PROFILING(&comms_profile, __func__);
}

// Enforce reflective boundary conditions on the problem state
void handle_boundary_3d(const int nx, const int ny, const int nz, Mesh* mesh,
                        double* arr, const int invert, const int pack) {
#if 0
  START_PROFILING(&comms_profile);

  int* neighbours = mesh->neighbours;

#ifdef MPI
  int nmessages = 0;

  if(pack) {
    // Pack east and west
    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < pad; ++dd) {
            mesh->east_buffer_out[(ii*ny*pad)+(jj*pad)+(dd)] =
              arr[(ii*nx*ny)+(jj*nx)+(nx-2*pad+dd)];
          }
        }
      }

      non_block_send(
          mesh->east_buffer_out, nz*ny*pad, neighbours[EAST], 2, nmessages++);
      non_block_recv(
          mesh->east_buffer_in, nz*ny*pad, neighbours[EAST], 3, nmessages++);
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < pad; ++dd) {
            mesh->west_buffer_out[(ii*ny*pad)+(jj*pad)+(dd)] =
              arr[(ii*nx*ny)+(jj*nx)+(pad+dd)];
          }
        }
      }

      non_block_send(
          mesh->west_buffer_out, nz*ny*pad, neighbours[WEST], 3, nmessages++);
      non_block_recv(
          mesh->west_buffer_in, nz*ny*pad, neighbours[WEST], 2, nmessages++);
    }

    // Pack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < pad; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->north_buffer_out[(ii*pad*nx)+(dd*nx)+(kk)] =
              arr[(ii*nx*ny)+((ny-2*pad+dd)*nx)+(kk)];
          }
        }
      }

      non_block_send(
          mesh->north_buffer_out, nz*nx*pad, neighbours[NORTH], 1, nmessages++);
      non_block_recv(
          mesh->north_buffer_in, nz*nx*pad, neighbours[NORTH], 0, nmessages++);
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < pad; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->south_buffer_out[(ii*pad*nx)+(dd*nx)+(kk)] =
              arr[(ii*nx*ny)+((pad+dd)*nx)+(kk)];
          }
        }
      }

      non_block_send(
          mesh->south_buffer_out, nz*nx*pad, neighbours[SOUTH], 0, nmessages++);
      non_block_recv(
          mesh->south_buffer_in, nz*nx*pad, neighbours[SOUTH], 1, nmessages++);
    }

    // Pack front and back
    if(neighbours[FRONT] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < pad; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->front_buffer_out[(dd*nx*ny)+(jj*nx)+(kk)] =
              arr[((pad+dd)*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }

      non_block_send(
          mesh->front_buffer_out, nx*ny*pad, neighbours[FRONT], 4, nmessages++);
      non_block_recv(
          mesh->front_buffer_in, nx*ny*pad, neighbours[FRONT], 5, nmessages++);
    }

    if(neighbours[BACK] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < pad; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->back_buffer_out[(dd*nx*ny)+(jj*nx)+(kk)] =
              arr[((nz-2*pad+dd)*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }

      non_block_send(
          mesh->back_buffer_out, nx*ny*pad, neighbours[BACK], 5, nmessages++);
      non_block_recv(
          mesh->back_buffer_in, nx*ny*pad, neighbours[BACK], 4, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < pad; ++dd) {
            arr[(ii*nx*ny)+(jj*nx)+(nx-pad+dd)] =
              mesh->east_buffer_in[(ii*ny*pad)+(jj*pad)+(dd)];
          }
        }
      }
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < pad; ++dd) {
            arr[(ii*nx*ny)+(jj*nx)+dd] =
              mesh->west_buffer_in[(ii*ny*pad)+(jj*pad)+(dd)];
          }
        }
      }
    }

    // Unpack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < pad; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[(ii*nx*ny)+((ny-pad+dd)*nx)+(kk)] =
              mesh->north_buffer_in[(ii*pad*nx)+(dd*nx)+(kk)];
          }
        }
      }
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < pad; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[(ii*nx*ny)+(dd*nx)+(kk)] =
              mesh->south_buffer_in[(ii*pad*nx)+(dd*nx)+(kk)];
          }
        }
      }
    }

    // Unpack front and back
    if(neighbours[FRONT] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < pad; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[(dd*nx*ny)+(jj*nx)+(kk)] = 
              mesh->front_buffer_in[(dd*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }
    }

    if(neighbours[BACK] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < pad; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[((nz-pad+dd)*nx*ny)+(jj*nx)+(kk)] =
              mesh->back_buffer_in[(dd*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }
    }
  }
#endif

  // Perform the boundary reflections, potentially with the data updated from neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;
  double z_inversion_coeff = (invert == INVERT_Z) ? -1.0 : 1.0;

  // Reflect at the east
  if(neighbours[EAST] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int dd = 0; dd < pad; ++dd) {
          arr[(ii*nx*ny)+(jj*nx)+(nx-pad+dd)] = 
            x_inversion_coeff*arr[(ii*nx*ny)+(jj*nx)+(nx-1-pad-dd)];
        }
      }
    }
  }

  // Reflect at the west
  if(neighbours[WEST] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int dd = 0; dd < pad; ++dd) {
          arr[(ii*nx*ny)+(jj*nx)+(pad-1-dd)] =
            x_inversion_coeff*arr[(ii*nx*ny)+(jj*nx)+(pad+dd)];
        }
      }
    }
  }

  // Reflect at north
  if(neighbours[NORTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int dd = 0; dd < pad; ++dd) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[(ii*nx*ny)+((ny-pad+dd)*nx)+(kk)] =
            y_inversion_coeff*arr[(ii*nx*ny)+((ny-1-pad-dd)*nx)+(kk)];
        }
      }
    }
  }

  // Reflect at the south
  if(neighbours[SOUTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int dd = 0; dd < pad; ++dd) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[(ii*nx*ny)+((pad-1-dd)*nx)+(kk)] =
            y_inversion_coeff*arr[(ii*nx*ny)+((pad+dd)*nx)+(kk)];
        }
      }
    }
  }

  // Reflect at the front
  if(neighbours[FRONT] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < pad; ++dd) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[((pad-1-dd)*nx*ny)+(jj*nx)+(kk)] =
            z_inversion_coeff*arr[((pad+dd)*nx*ny)+(jj*nx)+(kk)];
        }
      }
    }
  }

  // Reflect at the back
  if(neighbours[BACK] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < pad; ++dd) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[((nz-pad+dd)*nx*ny)+(jj*nx)+(kk)] =
            z_inversion_coeff*arr[((nz-1-pad-dd)*nx*ny)+(jj*nx)+(kk)];
        }
      }
    }
  }

  STOP_PROFILING(&comms_profile, __func__);
#endif // if 0
}

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect(const int nnodes, const int* boundary_index,
                                 const int* boundary_type,
                                 const double* boundary_normal_x,
                                 const double* boundary_normal_y,
                                 double* velocity_x, double* velocity_y) {
#pragma omp parallel for
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

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect_3d(const int nnodes, const int* boundary_index,
                                    const int* boundary_type,
                                    const double* boundary_normal_x,
                                    const double* boundary_normal_y,
                                    const double* boundary_normal_z,
                                    double* velocity_x, double* velocity_y,
                                    double* velocity_z) {
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int index = boundary_index[(nn)];
    if (index == IS_INTERIOR_NODE) {
      continue;
    }

    if (boundary_type[(index)] == IS_BOUNDARY) {

      // TODO: WE NEED TO CREATE A BASIS FOR THE PLANE USING THE NORMAL VECTOR
      // HERE AND THEN PROJECT THE VECTOR ONTO THAT PLANE.... USE ORTHOGONAL
      // PROJECT???

      // Project the velocity onto the face direction
      const double boundary_parallel_x = boundary_normal_y[(index)];
      const double boundary_parallel_y = -boundary_normal_x[(index)];
      const double boundary_parallel_z = -boundary_normal_z[(index)];

      const double vel_dot_parallel = (velocity_x[(nn)] * boundary_parallel_x +
                                       velocity_y[(nn)] * boundary_parallel_y);
      velocity_x[(nn)] = boundary_parallel_x * vel_dot_parallel;
      velocity_y[(nn)] = boundary_parallel_y * vel_dot_parallel;
      velocity_z[(nn)] = boundary_parallel_z * vel_dot_parallel;
    } else if (boundary_type[(index)] == IS_FIXED) {
      velocity_x[(nn)] = 0.0;
      velocity_y[(nn)] = 0.0;
      velocity_z[(nn)] = 0.0;
    }
  }
}
