#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fast.h"
#include "../fast_interface.h"
#include "../../profiler.h"
#include "../../comms.h"
#include "mkl_dfti.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

// Dummy problem solves the Poisson equation using an FFT
// Taken from: 
void solve_fft(
    const int nx, const int ny, Mesh* mesh, const double dt, double* e)
{
  int nl = 500;
  double* energy = (double*)malloc(sizeof(double)*nl);
  for(int ii = 0; ii < nl; ++ii) {
    energy[ii] = sin((double)ii);
  }
  fft_1d(nl, energy);

  for(int ii = 0; ii < nl; ++ii) {
    printf("%e\n", energy[ii]);
  }
}

// Performs a one-dimensional FFT using Cooley-Tukey algorithm
void fft_1d(
    const int nl, const double* energy0, double* energy1)
{
  
}

// Performs a one-dimensional FFT using Intel MKL
void mkl_fft_1d(
    const int nl, double* energy)
{
  DFTI_DESCRIPTOR_HANDLE desc_handle;
  MKL_LONG status;
  status = DftiCreateDescriptor(&desc_handle, DFTI_DOUBLE, DFTI_REAL, 1, nl);
  status = DftiCommitDescriptor(desc_handle);

  if(status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
    printf ("Error: %s\n", DftiErrorMessage(status));
    exit(1);
  }

  status = DftiComputeForward(desc_handle, energy);
  status = DftiFreeDescriptor(&desc_handle);
}

