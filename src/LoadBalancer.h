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
			std::pair<DataBlock, Processor>(DataBlock(Data.first, Data.second/2), {0, gpu}),
			std::pair<DataBlock, Processor>(DataBlock(Data.first+Data.second/2, Data.second/2), {1, cpu})
		};
	}

private:
	const Algorithm<Type>* algorithm;
};
