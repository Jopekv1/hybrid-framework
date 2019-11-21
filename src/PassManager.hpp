#pragma once
#include "Algorithm.h"
#include "LoadBalancer.h"

template<typename DType, typename AlgType>
class PassManager
{
public:
    using DataBlock = typename Algorithm<DType>::DataBlock;
    
    PassManager(/* engine*/int t, int s) { type = t; share = s; }
    ~PassManager() = default;

    DataBlock run(DataBlock data);

private:
    DataBlock runCPU(DataBlock data, AlgType* algorithm);
    int type;
    int share;
};

template<typename DType, typename AlgType>
auto PassManager<DType, AlgType>::run(DataBlock data) -> DataBlock
{
    AlgType* alg = new AlgType;
    LoadBalancer<DType> lb(alg);
    auto dividedData = lb.calculate(data, type, share);

    std::vector<DataBlock> outs;
    std::vector<AlgType*> algs(dividedData.size());
    for (int i = 0; i < dividedData.size(); ++i)
    {
        algs[i] = new AlgType;
    }
#pragma omp parallel for
    for (int i = 0; i < dividedData.size(); ++i)
    {
        DataBlock out;
        if (dividedData[i].second.type == cpu)
            out = runCPU(dividedData[i].first, algs[i]);
        else
        {
            cudaSetDevice(dividedData[i].second.id);
            out = algs[i]->runGPU(dividedData[i].first);
        }
        //engine.runAlgorithm(dividedData[i].first);
        //std::sort(dividedData[i].first.first, dividedData[i].first.first + dividedData[i].first.second);
        //auto out = dividedData[i].first;
#pragma omp critical
        outs.push_back(out);
    }
    //engine.free(data);
    DataBlock ret = alg->mergeBlocks(outs);
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
