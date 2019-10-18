#pragma once
#include <utility>
#include "Processor.h"
#include "Algorithm.h"

template<typename Type>
class LoadBalancer
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;
	
	LoadBalancer(Algorithm<Type> algorithm) : algorithm(algorithm) {}

	std::vector < std::pair<DataBlock, Processor> > calculate(DataBlock data);

private:
	const Algorithm<Type> algorithm;
};
