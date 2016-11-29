#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "shared.h"

struct Profile compute_profile = {0};
struct Profile comms_profile = {0};

// Write out data for visualisation in visit
void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time)
{
#ifdef ENABLE_VISIT_DUMPS
  char bovname[256];
  char datname[256];
  sprintf(bovname, "%s%d.bov", name, step);
  sprintf(datname, "%s%d.dat", name, step);

  FILE* bovfp = fopen(bovname, "w");

  if(!bovfp) {
    printf("Could not open file %s\n", bovname);
    exit(1);
  }

  fprintf(bovfp, "TIME: %.4f\n", time);
  fprintf(bovfp, "DATA_FILE: %s\n", datname);
  fprintf(bovfp, "DATA_SIZE: %d %d 1\n", nx, ny);
  fprintf(bovfp, "DATA_FORMAT: DOUBLE\n");
  fprintf(bovfp, "VARIABLE: density");
  fprintf(bovfp, "DATA_ENDIAN: LITTLE\n");
  fprintf(bovfp, "CENTERING: zone\n");

#ifdef MPI
  fprintf(bovfp, "BRICK_ORIGIN: %f %f 0.\n", (float)x_off, (float)y_off);
#else
  fprintf(bovfp, "BRICK_ORIGIN: 0. 0. 0.\n");
#endif

  fprintf(bovfp, "BRICK_SIZE: %d %d 1\n", nx, ny);
  fclose(bovfp);

  FILE* datfp = fopen(datname, "wb");
  if(!datfp) {
    printf("Could not open file %s\n", datname);
    exit(1);
  }

  fwrite(data, sizeof(data), nx*ny, datfp);
  fclose(datfp);
#endif
}

// TODO: Fix this method - shouldn't be necessary to bring the data back from
// all of the ranks, this is over the top
void write_all_ranks_to_visit(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const int x_off, const int y_off, const int rank, 
    const int nranks, double* local_arr, 
    const char* name, const int tt, const double elapsed_sim_time)
{
  // If MPI is enabled need to collect the data from all 
#if defined(MPI) && defined(ENABLE_VISIT_DUMPS)
  double* global_arr;
  double** remote_data;

  if(rank == MASTER) {
    global_arr = (double*)malloc(sizeof(double)*global_nx*global_ny);
    remote_data = (double**)malloc(sizeof(double*)*nranks);
    remote_data[MASTER] = local_arr;
  }

  for(int ii = 0; ii < nranks; ++ii) {
    int dims[4];
    dims[0] = local_nx;
    dims[1] = local_ny;
    dims[2] = x_off;
    dims[3] = y_off;

    if(rank == MASTER) {
      if(ii > MASTER) {
        MPI_Recv(&dims, 4, MPI_INT, ii, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        remote_data[ii] = (double*)malloc(sizeof(double)*dims[0]*dims[1]);
        MPI_Recv(
            remote_data[ii], dims[0]*dims[1], MPI_DOUBLE, ii, 1, 
            MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      }

      for(int jj = PAD; jj < dims[1]-PAD; ++jj) {
        for(int kk = PAD; kk < dims[0]-PAD; ++kk) {
          global_arr[(dims[3]+(jj-PAD))*global_nx+((kk-PAD)+dims[2])] =
            remote_data[ii][jj*dims[0]+kk];
        }
      }
    }
    else if(ii == rank) {
      MPI_Send(&dims, 4, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
      MPI_Send(local_arr, dims[0]*dims[1], MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#else
  double* global_arr = local_arr;
#endif

  if(rank == MASTER) {
    write_to_visit(global_nx, global_ny, 0, 0, global_arr, name, tt, elapsed_sim_time);
  }
}

