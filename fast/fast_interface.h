#pragma once

#include "../mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

// Performs the CG solve
void solve_fft(
    const int nx, const int ny, double* e);

#ifdef __cplusplus
}
#endif
