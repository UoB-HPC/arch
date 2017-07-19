#include "../shared.h"
#include <omp.h>

void initialise_devices(int rank) {
  int ndevices = omp_get_num_devices();

  // Assume uniform distribution of devices on nodes
  omp_set_default_device(rank % ndevices);
}
