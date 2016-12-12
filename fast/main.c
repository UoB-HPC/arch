#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fast_interface.h"
#include "../profiler.h"
#include "../comms.h"
#include "../shared.h"
#include "../state.h"
#include "../mesh.h"

int main(int argc, char** argv) 
{
  if(argc < 4)
  {
    printf("Usage: ./fast.exe <nx> <ny> <niters>\n");
    exit(1);
  }

  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.width = WIDTH;
  mesh.height = HEIGHT;
  mesh.dt = MAX_HOT_DT;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh(&mesh);

  State state = {0};
  initialise_state(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &state);

  struct Profile wallclock = {0};

  START_PROFILING(&wallclock);

  int tt = 0;
  double elapsed_sim_time = 0.0;
  for(tt = 0; tt < mesh.niters; ++tt) {
    if(mesh.rank == MASTER)
      printf("step %d\n", tt+1);

    solve_fft(
        mesh.local_nx, mesh.local_ny, state.e);

    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= SIM_END) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }
  }

  STOP_PROFILING(&wallclock, "wallclock");

  if(mesh.rank == MASTER) {
    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    PRINT_PROFILING_RESULTS(&compute_profile);
    printf("wallclock %.2fs, elapsed simulation time %.4fs\n", pe.time, elapsed_sim_time);
  }

  finalise_state(&state);
  finalise_mesh(&mesh);
  finalise_comms();
}

