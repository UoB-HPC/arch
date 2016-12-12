#pragma once

#include "../../shared.h"
#include "../../mesh.h"

// Calculates a 1d FFT using MKL
void mkl_fft_1d(
    const int nl, double* energy);

void fft_1d(
    const int nl, const double* energy0, double* energy1);

