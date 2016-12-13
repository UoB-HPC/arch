#include "../comms.h"
#include "../mesh.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int pack)
{
  START_PROFILING(&comms_profile);

  int* neighbours = mesh->neighbours;

#ifdef MPI
  int nmessages = 0;

  if(pack) {
    // Pack east and west
    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          mesh->east_buffer_out[(ii-PAD)*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
        }
      }

      non_block_send(mesh->east_buffer_out, (ny-2*PAD)*PAD, neighbours[EAST], 2, nmessages++);
      non_block_recv(mesh->east_buffer_in, (ny-2*PAD)*PAD, neighbours[EAST], 3, nmessages++);
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          mesh->west_buffer_out[(ii-PAD)*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
        }
      }

      non_block_send(mesh->west_buffer_out, (ny-2*PAD)*PAD, neighbours[WEST], 3, nmessages++);
      non_block_recv(mesh->west_buffer_in, (ny-2*PAD)*PAD, neighbours[WEST], 2, nmessages++);
    }

    // Pack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          mesh->north_buffer_out[dd*(nx-2*PAD)+(jj-PAD)] = arr[(ny-2*PAD+dd)*nx+jj];
        }
      }

      non_block_send(mesh->north_buffer_out, (nx-2*PAD)*PAD, neighbours[NORTH], 1, nmessages++);
      non_block_recv(mesh->north_buffer_in, (nx-2*PAD)*PAD, neighbours[NORTH], 0, nmessages++);
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          mesh->south_buffer_out[dd*(nx-2*PAD)+(jj-PAD)] = arr[(PAD+dd)*nx+jj];
        }
      }

      non_block_send(mesh->south_buffer_out, (nx-2*PAD)*PAD, neighbours[SOUTH], 0, nmessages++);
      non_block_recv(mesh->south_buffer_in, (nx-2*PAD)*PAD, neighbours[SOUTH], 1, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + dd] = mesh->west_buffer_in[(ii-PAD)*PAD+dd];
        }
      }
    }

    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = PAD; ii < ny-PAD; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[(ii-PAD)*PAD+dd];
        }
      }
    }

    // Unpack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          arr[(ny-PAD+dd)*nx+jj] = mesh->north_buffer_in[dd*(nx-2*PAD)+(jj-PAD)];
        }
      }
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = PAD; jj < nx-PAD; ++jj) {
          arr[dd*nx + jj] = mesh->south_buffer_in[dd*(nx-2*PAD)+(jj-PAD)];
        }
      }
    }
  }
#endif

  // Perform the boundary reflections, potentially with the data updated from neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if(neighbours[NORTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = PAD; jj < nx-PAD; ++jj) {
        arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
      }
    }
  }
  // reflect at the south
  if(neighbours[SOUTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = PAD; jj < nx-PAD; ++jj) {
        arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
      }
    }
  }
  // reflect at the east
  if(neighbours[EAST] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = PAD; ii < ny-PAD; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
      }
    }
  }
  if(neighbours[WEST] == EDGE) {
    // reflect at the west
#pragma omp parallel for collapse(2)
    for(int ii = PAD; ii < ny-PAD; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
      }
    }
  }
  STOP_PROFILING(&comms_profile, __func__);
}

// Enforce reflective boundary conditions on the problem state
void handle_boundary_3d(
    const int nx, const int ny, const int nz, Mesh* mesh, double* arr, 
    const int invert, const int pack)
{
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
          for(int dd = 0; dd < PAD; ++dd) {
            mesh->east_buffer_out[ii*ny*PAD+jj*PAD+dd] = arr[(ii*nx*ny)+(jj*ny)+(nx-2*PAD+dd)];
          }
        }
      }

      non_block_send(mesh->east_buffer_out, nz*ny*PAD, neighbours[EAST], 2, nmessages++);
      non_block_recv(mesh->east_buffer_in, nz*ny*PAD, neighbours[EAST], 3, nmessages++);
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < PAD; ++dd) {
            mesh->west_buffer_out[ii*ny*PAD+jj*PAD+dd] = arr[(ii*nx*ny)+(jj*ny)+(PAD+dd)];
          }
        }
      }

      non_block_send(mesh->west_buffer_out, nz*ny*PAD, neighbours[WEST], 3, nmessages++);
      non_block_recv(mesh->west_buffer_in, nz*ny*PAD, neighbours[WEST], 2, nmessages++);
    }

    // Pack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->north_buffer_out[(ii*PAD*nx)+(dd*nx)+(kk)] = arr[(ii*nx*ny)+(ny-2*PAD+dd)*nx+(kk)];
          }
        }
      }

      non_block_send(mesh->north_buffer_out, nz*nx*PAD, neighbours[NORTH], 1, nmessages++);
      non_block_recv(mesh->north_buffer_in, nz*nx*PAD, neighbours[NORTH], 0, nmessages++);
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->south_buffer_out[(kk*PAD*nx)+(dd*nx)+(kk)] = arr[(ii*nx*ny)+(PAD+dd)*nx+(kk)];
          }
        }
      }

      non_block_send(mesh->south_buffer_out, nz*nx*PAD, neighbours[SOUTH], 0, nmessages++);
      non_block_recv(mesh->south_buffer_in, nz*nx*PAD, neighbours[SOUTH], 1, nmessages++);
    }

    // Pack front and back
    if(neighbours[FRONT] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->front_buffer_out[(dd*nx*ny)+(jj*nx)+(kk)] = arr[((PAD+dd)*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }

      non_block_send(mesh->front_buffer_out, nx*ny*PAD, neighbours[FRONT], 1, nmessages++);
      non_block_recv(mesh->front_buffer_in, nx*ny*PAD, neighbours[FRONT], 0, nmessages++);
    }

    if(neighbours[BACK] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            mesh->back_buffer_out[(dd*nx*ny)+(jj*nx)+(kk)] = arr[((nz-2*PAD+dd)*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }

      non_block_send(mesh->back_buffer_out, nx*ny*PAD, neighbours[BACK], 1, nmessages++);
      non_block_recv(mesh->back_buffer_in, nx*ny*PAD, neighbours[BACK], 0, nmessages++);
    }

    wait_on_messages(nmessages);

    // Unpack east and west
    if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < PAD; ++dd) {
            arr[(ii*nx*ny)+(jj*ny)+(nx-PAD+dd)] = mesh->east_buffer_out[(ii*ny*PAD)+(jj*PAD)+(dd)];
          }
        }
      }
    }

    if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int dd = 0; dd < PAD; ++dd) {
            arr[(ii*nx*ny)+(jj*nx)+dd] = mesh->west_buffer_out[ii*ny*PAD+jj*PAD+dd];
          }
        }
      }
    }

    // Unpack north and south
    if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[(ii*nx*ny)+((ny-PAD+dd)*nx)+(kk)] = mesh->north_buffer_out[(ii*PAD*nx)+(dd*nx)+(kk)];
          }
        }
      }
    }

    if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < nz; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[(ii*nx*ny)+(dd*nx)+(kk)] = mesh->south_buffer_out[(kk*PAD*nx)+(dd*nx)+(kk)];
          }
        }
      }
    }

    // Unpack front and back
    if(neighbours[FRONT] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[(dd*nx*ny)+(jj*nx)+(kk)] = mesh->front_buffer_out[(dd*nx*ny)+(jj*nx)+(kk)];
          }
        }
      }
    }

    if(neighbours[BACK] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < ny; ++jj) {
          for(int kk = 0; kk < nx; ++kk) {
            arr[((nz-PAD+dd)*nx*ny)+(jj*nx)+(kk)] = mesh->back_buffer_out[(dd*nx*ny)+(jj*nx)+(kk)];
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
  if(neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[(ii*nx*ny)+(jj*ny)+(nx-PAD+dd)] = x_inversion_coeff*arr[(ii*ny*PAD)+(jj*PAD)+(nx-1-PAD-dd)];
        }
      }
    }
  }

  // Reflect at the west
  if(neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[(ii*nx*ny)+(jj*nx)+(PAD-1-dd)] = mesh->west_buffer_out[(ii*ny*PAD)+(jj*PAD)+(PAD+dd)];
        }
      }
    }
  }

  // Reflect at north
  if(neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[(ii*nx*ny)+((ny-PAD+dd)*nx)+(kk)] = y_inversion_coeff*arr[(ii*PAD*nx)+((ny-1-PAD-dd)*nx)+(kk)];
        }
      }
    }
  }

  // Reflect at the south
  if(neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < nz; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[(ii*nx*ny)+((PAD-1-dd)*nx)+(kk)] = y_inversion_coeff*arr[(kk*PAD*nx)+((PAD+dd)*nx)+(kk)];
        }
      }
    }
  }

  // Reflect at the front
  if(neighbours[FRONT] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[((PAD-1-dd)*nx*ny)+(jj*nx)+(kk)] = z_inversion_coeff*arr[((PAD+dd)*nx*ny)+(jj*nx)+(kk)];
        }
      }
    }
  }

  // Reflect at the back
  if(neighbours[BACK] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < ny; ++jj) {
        for(int kk = 0; kk < nx; ++kk) {
          arr[((nz-PAD+dd)*nx*ny)+(jj*nx)+(kk)] = z_inversion_coeff*arr[((nz-1-PAD-dd)*nx*ny)+(jj*nx)+(kk)];
        }
      }
    }
  }
  STOP_PROFILING(&comms_profile, __func__);
}

