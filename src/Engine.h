#pragma once
constexpr auto gigaByte = 1073741824.0f;
#include <vector>
#include "Processor.h"
#include "Algorithm.h"


class Engine
{
public:

	Engine();
	int getDeviceCount() { return deviceCount; }
	std::vector <float> getMemories() { return memorySizes; }
	std::vector <Processor> getProcessors() { return processors; }
	template<typename Type>
	void allocateMemoryInCuda(Type* pointer, std::uint32_t dataNumber, int deviceNumber);
	void freeMemoryInCuda(void* pointer, int deviceNumber);
	template<typename Type>
	Algorithm<Type>::DataBlock runOnGPU(Algorithm<Type> algorithm, Algorithm<Type>::DataBlock data, std::uint32_t procNumer);
	
	~Engine();

private:
	int deviceCount;
	std::vector <float> memorySizes;
	std::vector <Processor> processors;
	std::vector <std::pair<int,void*>> allocations;
};

