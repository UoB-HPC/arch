#ifndef __SHAREDDATAHDR
#define __SHAREDDATAHDR

#include "mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: MAKE IT SO THAT shared_data IS LOCAL TO THE APPLICATIONS???

// Contains all of the shared_data information for the solver
typedef struct {
  // Shared shared_data (share data)
  double* rho; // Density
  double* e;   // Energy

  // Paired shared_data (share capacity)
  // TODO: IT WOULD BE BETTER TO AUTOMATE THIS FROM A DYNAMIC DATA STRUCTURE
  double* Ap;      // HOT: Coefficient matrix A, by conjugate vector p
  double* rho_old; // WET: Density at beginning of timestep

  double* s_x; // HOT: Coefficients in x direction
  double* Qxx; // WET: Artificial viscous term in x direction

  double* s_y; // HOT: Coefficients in y direction
  double* Qyy; // WET: Artificial viscous term in y direction

  double* s_z; // HOT: Coefficients in z direction
  double* Qzz; // WET: Artificial viscous term in z direction

  double* r; // HOT: The residual vector
  double* P; // WET: The pressure

  double* x; // HOT: The solution vector (new energy)
  double* u; // WET: The velocity in the x direction

  double* p; // HOT: The conjugate vector
  double* v; // WET: The velocity in the y direction

  double* reduce_array0;
  double* reduce_array1;

} SharedData;

// Initialises the shared_data variables
void initialise_shared_data_2d(const int global_nx, const int global_ny,
                               const int local_nx, const int local_ny,
                               const int pad, const int x_off, const int y_off,
                               const double mesh_width,
                               const double mesh_height,
                               const char* problem_def_filename,
                               const double* edgex, const double* edgey,
                               SharedData* shared_data);

// Initialise state data in device specific manner
void set_problem_2d(const int global_nx, const int global_ny,
                    const int local_nx, const int local_ny, const int pad,
                    const int x_off, const int y_off, const double mesh_width,
                    const double mesh_height, const double* edgex,
                    const double* edgey, const int ndims,
                    const char* problem_def_filename, double* rho, double* e,
                    double* x);

// Initialises the shared_data variables
void initialise_shared_data_3d(
    const int global_nx, const int global_ny, const int global_nz,
    const int local_nx, const int local_ny, const int local_nz, const int pad,
    const int x_off, const int y_off, const int z_off, const double mesh_width,
    const double mesh_height, const double mesh_depth,
    const char* problem_def_filename, const double* edgex, const double* edgey,
    const double* edgez, SharedData* shared_data);

void set_problem_3d(const int global_nx, const int global_ny,
                    const int global_nz, const int local_nx, const int local_ny,
                    const int local_nz, const int pad, const int x_off,
                    const int y_off, const int z_off, const double mesh_width,
                    const double mesh_height, const double mesh_depth,
                    const double* edgex, const double* edgey,
                    const double* edgez, const int ndims,
                    const char* problem_def_filename, double* rho, double* e,
                    double* x);

// Deallocate all of the shared_data memory
void finalise_shared_data(SharedData* shared_data);

#ifdef __cplusplus
}
#endif

#endif
