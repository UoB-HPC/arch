
#define NBLOCKS 128

#define set_cuda_indices(padx) \
  const int gid = threadIdx.x+blockIdx.x*blockDim.x; \
const int jj = (gid % (nx+padx));\
const int ii = (gid / (nx+padx));

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)


