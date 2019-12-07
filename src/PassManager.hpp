#pragma once
#include "Algorithm.h"
#include "LoadBalancer.h"

template<typename DType, typename AlgType>
class PassManager
{
public:
    using DataBlock = typename Algorithm<DType>::DataBlock;
    
    PassManager(Engine& engine, LoadBalancer<DType>& loadBalancer) : engine(engine), lb(loadBalancer) {}
    ~PassManager() = default;

    DataBlock run(DataBlock data, DataBlock params);

private:
    DataBlock runCPU(DataBlock data, AlgType* algorithm);
	LoadBalancer<DType>& lb;
	Engine& engine;
};

template<typename DType, typename AlgType>
auto PassManager<DType, AlgType>::run(DataBlock data, DataBlock params) -> DataBlock
{
    AlgType* alg = new AlgType(params);
    auto dividedData = lb.calculate(data, engine);

    std::vector<DataBlock> outs;
    std::vector<AlgType*> algs(dividedData.size());
    for (int i = 0; i < dividedData.size(); ++i)
    {
		algs[i] = new AlgType(params);
    }
#pragma omp parallel for
    for (int i = 0; i < dividedData.size(); ++i)
    {
        DataBlock out;
        if (dividedData[i].second.type == cpu)
            out = runCPU(dividedData[i].first, algs[i]);
        else
            out = engine.runOnGPU(algs[i], dividedData[i].first, dividedData[i].second.id);
#pragma omp critical
        outs.push_back(out);
    }
    DataBlock ret = alg->mergeBlocks(outs);
	for (int i = 0; i < dividedData.size(); i++)
		delete algs[i];
    return ret;
}

template<typename DType, typename AlgType>
auto PassManager<DType, AlgType>::runCPU(DataBlock data, AlgType* algorithm) -> DataBlock
{
    algorithm->preDivision(data);
    if (algorithm->isBase(data))
        return algorithm->runBaseCPU(data);
    const int n = algorithm->getChildrenNum(data);
    std::vector<DataBlock> outs(n);
    for (int i = 0; i < n; i++)
    {
        outs[i] = runCPU(algorithm->getChild(data, i), algorithm);
    }
    return algorithm->merge(outs);
}
