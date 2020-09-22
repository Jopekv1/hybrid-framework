#include "load_balancer.h"

#include <cuda_runtime.h>


void LoadBalancer::getDeviceCount() {
	cudaGetDeviceCount(&this->gpuCount);
}

void LoadBalancer::synchronize() {
	cudaDeviceSynchronize();
}
