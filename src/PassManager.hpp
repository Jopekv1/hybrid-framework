#pragma once
#include "Algorithm.h"
#include "LoadBalancer.h"

template<typename Type>
class PassManager
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;
	
	PassManager(Algorithm<Type>* algorithm/*, engine*/) : algorithm(algorithm) {}
	~PassManager() = default;

	DataBlock run(DataBlock data);

private:
	Algorithm<Type>* algorithm;

	DataBlock runCPU(DataBlock data);
};

template <typename Type>
auto PassManager<Type>::run(DataBlock data) -> DataBlock
{
	LoadBalancer<Type> lb(algorithm);
	auto dividedData = lb.calculate(data);

	std::vector<DataBlock> outs;
#pragma omp parallel for private(algorithm)
	for (int i = 0; i < dividedData.size(); i++)
	{
		DataBlock out;
		if (dividedData[i].second.type == cpu)
			out = runCPU(dividedData[i].first);
		else
			out = algorithm->runGPU(dividedData[i].first);//engine.runAlgorithm(dividedData[i].first);
#pragma omp critical
		outs.push_back(out);
	}
	//engine.free(data);
	DataBlock ret = algorithm->mergeBlocks(outs);
	return ret;
}

template <typename Type>
auto PassManager<Type>::runCPU(DataBlock data) -> DataBlock
{
	algorithm->preDivision(data);
	if (algorithm->isBase(data))
		return algorithm->runBaseCPU(data);
	const int n = algorithm->getChildrenNum(data);
	std::vector<DataBlock> outs(n);
	for (int i = 0; i < n; i++)
	{
		outs[i] = runCPU(algorithm->getChild(data, i));
	}
	return algorithm->merge(outs);
}
