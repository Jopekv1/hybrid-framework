#include "Engine.h"
#include <cuda_runtime.h>
#include <iostream>

Engine::Engine()
{
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		throw std::exception();
	}
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		memorySizes.push_back(static_cast<float>(deviceProp.totalGlobalMem / gigaByte));
	}
}