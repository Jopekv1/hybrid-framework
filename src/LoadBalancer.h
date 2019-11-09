#pragma once
#include <utility>
#include "Processor.h"
#include "Algorithm.h"

template<typename Type>
class LoadBalancer
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;
	
	LoadBalancer(const Algorithm<Type>* algorithm) : algorithm(algorithm) {}

	std::vector < std::pair<DataBlock, Processor> > calculate(DataBlock data)
	{
		return
		{
			std::pair<DataBlock, Processor>(DataBlock(data.first, data.second/8), {0, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 1, data.second / 8), {1, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 2, data.second / 8), {2, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 3, data.second / 8), {3, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 4, data.second / 8), {4, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 5, data.second / 8), {5, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 6, data.second / 8), {6, cpu}),
			std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / 8 * 7, data.second / 8), {7, cpu})
		};
	}

private:
	const Algorithm<Type>* algorithm;
};

//template<typename Type>
//std::vector < std::pair<DataBlock, Processor> > LoadBalancer<Type>::calculate(DataBlock data)
//{
//	std::int64_t deviceCount;
//	auto retVal = cudaGetDeviceCount(&deviceCount);
//
//	if (retVal != cudaSuccess) {
//		printf("Wooden PC detected, no nVidia GPU");
//		throw std::exception();
//	}
//
//	auto dividedData = this->algorithm.divide(data);
//	std::vector< std::pair<DataBlock, Processor> > retVector(dividedData.size());
//
//	for (int i = 0; i < dividedData.size(); i++) {
//		auto type = i < deviceCount ? ProcType::gpu : ProcType::cpu;
//		retVector.push_back(std::make_pair(dividedData[i], Processor{ i, type }));
//	}
//
//	return retVector;
//}