#pragma once
constexpr auto gigaByte = 1073741824.0f;
#include <vector>
#include "Processor.h"
#include "Algorithm.h"
#include <cuda_runtime.h>
#include <iostream>

class Engine
{
public:

    Engine()
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

    int getDeviceCount() { return deviceCount; }
    std::vector <float> getMemories() { return memorySizes; }
    std::vector <Processor> getProcessors() { return processors; }

    template<typename Type>
    void allocateMemoryInCuda(Type* pointer, std::uint32_t dataNumber)
    {
        std::int32_t ptr = -1;
        for (std::uint32_t i = 0; i < allocations.size(); i++)
        {
            if (allocations[i] == pointer)
            {
                ptr = i;
                break;
            }
        }
        if (ptr == -1)
        {
            cudaMallocManaged(&pointer, dataNumber * sizeof(Type));
            allocations.push_back();
        }
        else
        {
            printf("This pointer was already used on this device");
        }
    }

    void freeMemoryInCuda(void* pointer)
    {
        std::int32_t ptr = -1;
        for (std::uint32_t i = 0; i < allocations.size(); i++)
        {
            if (allocations[i] == pointer)
            {
                ptr = i;
                break;
            }
        }
        if (ptr == -1)
        {
            printf("There is no such pointer");
        }
        else
        {
            cudaFree(pointer);
            allocations.erase(allocations.begin() + ptr);
        }
    }

    template<typename Type>
    Algorithm<Type>::DataBlock runOnGPU(Algorithm<Type> algorithm, Algorithm<Type>::DataBlock data, std::uint32_t procNumer)
    {
        cudaSetDevice(procNumer);
        return algorithm.runGPU(data);
    }

    ~Engine()
    {
        for (std::uint32_t i = 0; i < allocations.size(); i++)
        {
            cudaFree(allocations[i]);
        }
    }

private:
    int deviceCount;
    std::vector <float> memorySizes;
    std::vector <Processor> processors;
    std::vector <void*> allocations;
};