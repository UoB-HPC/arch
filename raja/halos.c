#include "../comms.h"
#include "../mesh.h"
#include "../umesh.h"
#include "shared.h"

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
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
        for (int dd = 0; dd < pad; ++dd) {
          mesh->east_buffer_out[(ii - pad) * pad + dd] =
              arr[(ii * nx) + (nx - 2 * pad + dd)];
        }
      });

      non_block_send(mesh->east_buffer_out, (ny - 2 * pad) * pad,
                     neighbours[EAST], 2, nmessages++);
      non_block_recv(mesh->east_buffer_in, (ny - 2 * pad) * pad,
                     neighbours[EAST], 3, nmessages++);
    }

    if (neighbours[WEST] != EDGE) {
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
        for (int dd = 0; dd < pad; ++dd) {
          mesh->west_buffer_out[(ii - pad) * pad + dd] =
              arr[(ii * nx) + (pad + dd)];
        }
      });

      non_block_send(mesh->west_buffer_out, (ny - 2 * pad) * pad,
                     neighbours[WEST], 3, nmessages++);
      non_block_recv(mesh->west_buffer_in, (ny - 2 * pad) * pad,
                     neighbours[WEST], 2, nmessages++);
    }

    // Pack north and south
    if (neighbours[NORTH] != EDGE) {
      for (int dd = 0; dd < pad; ++dd) {
        RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx-pad), [=] RAJA_DEVICE (int jj) {
          mesh->north_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
              arr[(ny - 2 * pad + dd) * nx + jj];
        });
      }

      non_block_send(mesh->north_buffer_out, (nx - 2 * pad) * pad,
                     neighbours[NORTH], 1, nmessages++);
      non_block_recv(mesh->north_buffer_in, (nx - 2 * pad) * pad,
                     neighbours[NORTH], 0, nmessages++);
    }

    if (neighbours[SOUTH] != EDGE) {
      for (int dd = 0; dd < pad; ++dd) {
        RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx-pad), [=] RAJA_DEVICE (int jj) {
          mesh->south_buffer_out[dd * (nx - 2 * pad) + (jj - pad)] =
              arr[(pad + dd) * nx + jj];
        });
      }

      non_block_send(mesh->south_buffer_out, (nx - 2 * pad) * pad,
                     neighbours[SOUTH], 0, nmessages++);
      non_block_recv(mesh->south_buffer_in, (nx - 2 * pad) * pad,
                     neighbours[SOUTH], 1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if (neighbours[WEST] != EDGE) {
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
        for (int dd = 0; dd < pad; ++dd) {
          arr[ii * nx + dd] = mesh->west_buffer_in[(ii - pad) * pad + dd];
        }
      });
    }

    if (neighbours[EAST] != EDGE) {
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
        for (int dd = 0; dd < pad; ++dd) {
          arr[ii * nx + (nx - pad + dd)] = mesh->east_buffer_in[(ii - pad) * pad + dd];
        }
      });
    }

    // Unpack north and south
    if (neighbours[NORTH] != EDGE) {
      for (int dd = 0; dd < pad; ++dd) {
        RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx-pad), [=] RAJA_DEVICE (int jj) {
          arr[(ny - pad + dd) * nx + jj] =
              mesh->north_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
        });
      }
    }

    if (neighbours[SOUTH] != EDGE) {
      for (int dd = 0; dd < pad; ++dd) {
        RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx-pad), [=] RAJA_DEVICE (int jj) {
          arr[dd * nx + jj] =
              mesh->south_buffer_in[dd * (nx - 2 * pad) + (jj - pad)];
        });
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
    for (int dd = 0; dd < pad; ++dd) {
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx-pad), [=] RAJA_DEVICE (int jj) {
        arr[(ny - pad + dd) * nx + jj] =
            y_inversion_coeff * arr[(ny - 1 - pad - dd) * nx + jj];
      });
    }
  }
  // reflect at the south
  if (neighbours[SOUTH] == EDGE) {
    for (int dd = 0; dd < pad; ++dd) {
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx-pad), [=] RAJA_DEVICE (int jj) {
        arr[(pad - 1 - dd) * nx + jj] =
            y_inversion_coeff * arr[(pad + dd) * nx + jj];
      });
    }
  }
  // reflect at the east
  if (neighbours[EAST] == EDGE) {
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (nx - pad + dd)] =
            x_inversion_coeff * arr[ii * nx + (nx - 1 - pad - dd)];
      }
    });
  }
  if (neighbours[WEST] == EDGE) {
// reflect at the west
      RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (pad - 1 - dd)] =
            x_inversion_coeff * arr[ii * nx + (pad + dd)];
      }
    });
  }
  STOP_PROFILING(&comms_profile, __func__);
}

// Enforce reflective boundary conditions on the problem state
void handle_boundary_3d(const int nx, const int ny, const int nz, Mesh* mesh,
                        double* arr, const int invert, const int pack) {
  TERMINATE("Not implemented\n");
}

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect(const int nnodes, const int* boundary_index,
                                 const int* boundary_type,
                                 const double* boundary_normal_x,
                                 const double* boundary_normal_y,
                                 double* velocity_x, double* velocity_y) {

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nnodes), [=] RAJA_DEVICE (int nn) {

    const int index = boundary_index[(nn)];
    if (index != IS_INTERIOR) {
      if (boundary_type[(index)] == IS_BOUNDARY) {
        // Project the velocity onto the face direction
        const double boundary_parallel_x = boundary_normal_y[(index)];
        const double boundary_parallel_y = -boundary_normal_x[(index)];
        const double vel_dot_parallel = (velocity_x[(nn)] * boundary_parallel_x +
                                         velocity_y[(nn)] * boundary_parallel_y);
        velocity_x[(nn)] = boundary_parallel_x * vel_dot_parallel;
        velocity_y[(nn)] = boundary_parallel_y * vel_dot_parallel;
      } else if (boundary_type[(index)] == IS_CORNER) {
        velocity_x[(nn)] = 0.0;
        velocity_y[(nn)] = 0.0;
      }
    }

  });
}

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect_3d(const int nnodes, const int* boundary_index,
                                    const int* boundary_type,
                                    const double* boundary_normal_x,
                                    const double* boundary_normal_y,
                                    const double* boundary_normal_z,
                                    double* velocity_x, double* velocity_y,
                                    double* velocity_z) {

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nnodes), [=] RAJA_DEVICE (int nn) {
    const int index = boundary_index[(nn)];
    if (index != IS_INTERIOR) {

    if (boundary_type[(index)] == IS_EDGE) {
      // The normal here isn't actually a normal but a projection vector
      const double ab = (velocity_x[(nn)] * boundary_normal_x[(index)] +
                         velocity_y[(nn)] * boundary_normal_y[(index)] +
                         velocity_z[(nn)] * boundary_normal_z[(index)]);

      // Project the vector onto the edge line
      velocity_x[(nn)] = ab * boundary_normal_x[(index)];
      velocity_y[(nn)] = ab * boundary_normal_y[(index)];
      velocity_z[(nn)] = ab * boundary_normal_z[(index)];
    } else if (boundary_type[(index)] == IS_BOUNDARY) {
      // Perform an orthogonal projection, assuming normal vector is normalised
      const double un = (velocity_x[(nn)] * boundary_normal_x[(index)] +
                         velocity_y[(nn)] * boundary_normal_y[(index)] +
                         velocity_z[(nn)] * boundary_normal_z[(index)]);
      velocity_x[(nn)] -= un * boundary_normal_x[(index)];
      velocity_y[(nn)] -= un * boundary_normal_y[(index)];
      velocity_z[(nn)] -= un * boundary_normal_z[(index)];
    } else if (boundary_type[(index)] == IS_CORNER) {
      velocity_x[(nn)] = 0.0;
      velocity_y[(nn)] = 0.0;
      velocity_z[(nn)] = 0.0;
    }
    }
  });
}
