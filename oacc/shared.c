#include "../shared.h"
#include "../comms.h"
#include <omp.h>
#include <openacc.h>

void initialise_devices(int rank) {
  const acc_device_t device_type = acc_get_device_type();
  const int ndevices = acc_get_num_devices(device_type);

  if(rank == MASTER) {
    printf("Found %d NVIDIA devices\n", ndevices);
  }

  if(ndevices == 0) {
    TERMINATE("Could not find any NVIDIA devices.\n");
  }

  // Assume uniform distribution of devices on nodes
  acc_set_device_num(rank % ndevices, device_type);
}
