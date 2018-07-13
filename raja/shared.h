#ifndef RAJA_API_HDR
#define RAJA_API_HDR

#include "RAJA/RAJA.hpp"

#if defined(RAJA_USE_CUDA)

#include "../cuda/shared.h"

#define CUDA_EXEC_BLOCK_SZ 128
#define CUDA_REDUCE_BLOCK_SZ 128

#define exec_policy RAJA::cuda_exec<CUDA_EXEC_BLOCK_SZ>
#define reduce_policy RAJA::cuda_reduce<CUDA_REDUCE_BLOCK_SZ>

#else

#define exec_policy RAJA::omp_parallel_for_exec
#define reduce_policy RAJA::omp_reduce

#endif

#endif // RAJA_API_HDR
