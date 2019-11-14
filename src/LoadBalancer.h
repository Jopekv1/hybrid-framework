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

	std::vector < std::pair<DataBlock, Processor> > calculate(DataBlock data, int type)
	{
		int div = 8;
		switch (type)
		{
		case 0:
			div = 1;
			return
			{
				std::pair<DataBlock, Processor>(DataBlock(data.first, data.second / div), {0, gpu})
			};
		case 1:
			div = 1;
			return
			{
				std::pair<DataBlock, Processor>(DataBlock(data.first, data.second / div), {0, cpu})
			};
		case 2:
			div = 2;
			return
			{
				std::pair<DataBlock, Processor>(DataBlock(data.first, data.second / div), {0, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 1, data.second / div), {1, cpu})
			};
		case 3:
			div = 4;
			return
			{
				std::pair<DataBlock, Processor>(DataBlock(data.first, data.second / div), {0, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 1, data.second / div), {1, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 2, data.second / div), {2, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 3, data.second / div), {3, cpu})
			};
		case 4:
			div = 8;
			return
			{
				std::pair<DataBlock, Processor>(DataBlock(data.first, data.second / div), {0, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 1, data.second / div), {1, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 2, data.second / div), {2, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 3, data.second / div), {3, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 4, data.second / div), {4, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 5, data.second / div), {5, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 6, data.second / div), {6, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 7, data.second / div), {7, cpu})
			};
		default:
			return
			{
				std::pair<DataBlock, Processor>(DataBlock(data.first, data.second / div), {0, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 1, data.second / div), {1, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 2, data.second / div), {2, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 3, data.second / div), {3, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 4, data.second / div), {4, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 5, data.second / div), {5, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 6, data.second / div), {6, cpu}),
				std::pair<DataBlock, Processor>(DataBlock(data.first + data.second / div * 7, data.second / div), {7, cpu})
			};
		}
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