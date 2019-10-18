#pragma once
#include "Algorithm.h"

template<typename Type>
class PassManager
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;
	
	PassManager(Algorithm<Type> algorithm/*, engine*/) : algorithm(algorithm) {}
	~PassManager() = default;

	DataBlock run(DataBlock data);

private:
	const Algorithm<Type> algorithm;
};