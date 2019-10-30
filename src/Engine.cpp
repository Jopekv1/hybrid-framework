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
		processors.push_back(Processor{ dev, ProcType::gpu });
	}
}

template <typename Type>
void Engine::allocateMemoryInCuda(Type* pointer, std::uint32_t dataNumber, int deviceNumber)
{
	std::int32_t ptr = -1;
	for (std::uint32_t i = 0; i < allocations.size(); i++)
	{
		if (allocations[i].first == deviceNumber && allocations[i].second == pointer)
		{
			ptr = i;
			break;
		}
	}
	if (ptr == -1)
	{
		cudaSetDevice(deviceNumber);
		cudaMallocManaged(&pointer, dataNumber * sizeof(Type));
		allocations.push_back(std::make_pair(deviceNumber, pointer));
	}
	else
	{
		printf("This pointer was already used on this device");
	}
}

void Engine::freeMemoryInCuda(void* pointer, int deviceNumber)
{
	std::int32_t ptr = -1, pos = 0;
	for (std::uint32_t i = 0; i < allocations.size(); i++)
	{
		if (allocations[i].first == deviceNumber && allocations[i].second == pointer)
		{
			ptr = i;
			break;
		}
		pos++;
	}
	if (ptr == -1)
	{
		printf("There is no such pointer on this device");
	}
	else
	{
		cudaSetDevice(deviceNumber);
		cudaFree(pointer);
		allocations.erase(allocations.begin() + pos);
	}
}


template<typename Type>
Algorithm<Type>::DataBlock Engine::runOnGPU(Algorithm<Type> algorithm, Algorithm<Type>::DataBlock data, std::uint32_t procNumer)
{
	cudaSetDevice(procNumer);
	return algorithm.runGPU(data);
}

Engine::~Engine()
{
	for (std::uint32_t i = 0; i < allocations.size(); i++)
	{
		cudaSetDevice(allocations[i].first);
		cudaFree(allocations[i].second);
	}
}