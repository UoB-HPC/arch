#include "../shared.h"
#include "shared.h"

// Selects a GPU
void initialise_devices(int rank) {
  int count = 0;
  gpu_check(cudaGetDeviceCount(&count));

  int device_num = rank % count;

#ifdef DEBUG
  printf("rank %d selects device %d\n", rank, device_num);
#endif

  // Assume uniform distribution of devices on nodes
  gpu_check(cudaSetDevice(device_num));
}
