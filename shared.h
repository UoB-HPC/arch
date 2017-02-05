#pragma once

#include <stdio.h>
#include "profiler.h"

#define ENABLE_VISIT_DUMPS 1 // Enables visit dumps in the descendent applications
#define VEC_ALIGN 32    // The vector alignment to be used by memory allocators
#define TAG_VISIT0 1000
#define TAG_VISIT1 1001

// Helper macros
#define strmatch(a, b) (strcmp(a, b) == 0)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define samesign(a, b) ((a*b) > 0.0)
#define absmin(a, b) ((fabs(a) < fabs(b)) ? (a) : (b))
#define minmod(a, b) (samesign(a, b) ? (absmin(a, b)) : (0.0))

#define TERMINATE(...) \
  fprintf(stderr, __VA_ARGS__);\
  exit(EXIT_FAILURE);

enum { RECV=0, SEND=1 }; // Whether data is sent to/received from device

// Global profile hooks
extern struct Profile compute_profile;
extern struct Profile comms_profile;

#ifdef __cplusplus
extern "C" {
#endif

  // Initialises devices in implementation-specific manner
  void initialise_devices(int rank);

  // Wrappers for data (de)allocation
  void allocate_data(double** buf, size_t len);
  void allocate_host_data(double** buf, size_t len);
  void deallocate_data(double* buf);
  void deallocate_host_data(double* buf);
  void sync_data(const size_t len, double** src, double** dst, int send);

  // Write out data for visualisation in visit
  void write_to_visit(
      const int nx, const int ny, const int x_off, const int y_off, 
      const double* data, const char* name, const int step, const double time);
  void write_to_visit_3d(
      const int nx, const int ny, const int nz, const int x_off, const int y_off, 
      const int z_off, const double* data, const char* name, const int step, 
      const double time);

  // Collects all of the mesh data from the fleet of ranks and then writes to 
  // visit
  void write_all_ranks_to_visit(
      const int global_nx, const int global_ny, const int local_nx, 
      const int local_ny, const int x_off, const int y_off, const int rank, 
      const int nranks, int* neighbours, double* local_arr, 
      const char* name, const int tt, const double elapsed_sim_time);

#ifdef __cplusplus
}
#endif

