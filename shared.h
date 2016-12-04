#pragma once

#include "profiler.h"

#define ENABLE_VISIT_DUMPS 1 // Enables visit dumps in the descendent applications

// Helper macros
#define strmatch(a, b) (strcmp(a, b) == 0)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define samesign(a, b) ((a*b) > 0.0)
#define absmin(a, b) ((fabs(a) < fabs(b)) ? (a) : (b))
#define minmod(a, b) (samesign(a, b) ? (absmin(a, b)) : (0.0))

// Global profile hooks
struct Profile compute_profile;
struct Profile comms_profile;

// Wrappers for data (de)allocation
void allocate_data(double** buf, size_t len);
void deallocate_data(double* buf);

// Write out data for visualisation in visit
void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time);

// Collects all of the mesh data from the fleet of ranks and then writes to 
// visit
void write_all_ranks_to_visit(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const int x_off, const int y_off, const int rank, 
    const int nranks, int* neighbours, double* local_arr, 
    const char* name, const int tt, const double elapsed_sim_time);

