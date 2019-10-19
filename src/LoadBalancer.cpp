#include "LoadBalancer.h"

#include <cuda_runtime.h>


template<typename Type>
std::vector < std::pair<DataBlock, Processor> > LoadBalancer<Type>::calculate(DataBlock data) 
{
	std::int64_t deviceCount;
	auto retVal = cudaGetDeviceCount(&deviceCount);

	if (retVal != cudaSuccess) {
		printf("Wooden PC detected, no nVidia GPU");
		throw std::exception();
	}

	auto dividedData = this->algorithm.divide(data);
	std::vector< std::pair<DataBlock, Processor> > retVector(dividedData.size());

	for (int i = 0; i < dividedData.size(); i++) {
		auto type = i < deviceCount ? ProcType::gpu : ProcType::cpu;
		retVector.push_back(std::make_pair(dividedData[i], Processor{ i, type }));
	}

	return retVector;
}