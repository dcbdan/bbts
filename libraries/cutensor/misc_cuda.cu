#include "misc_cuda.h"

bool misc_cuda_cuda_malloc(void** ptr, size_t size) {
  return cudaSuccess == cudaMalloc(ptr, size);
}


