#pragma once
#include <utility>
#include "Processor.h"
#include "Algorithm.h"
#include "Engine.h"
#include <omp.h>

template<typename Type>
class LoadBalancer
{
public:
    using DataBlock = typename Algorithm<Type>::DataBlock;

    std::vector < std::pair<DataBlock, Processor> > calculate(DataBlock data, Engine& engine)
    {
	    const auto gpuNum = engine.getDeviceCount();
		auto gpuSizes = engine.getMemories();
		auto processors = engine.getProcessors();
		for (auto& size : gpuSizes)
			size /= sizeof(Type);
		std::vector < std::pair<DataBlock, Processor> > out;
		auto restSize = data.second;
		auto* dataPtr = data.first;
    	for (int i = 0; i < gpuNum; i++)
    	{
			if (restSize <= gpuSizes[i])
			{
				out.push_back(std::pair<DataBlock, Processor>(DataBlock(dataPtr, restSize), processors[i]));
				return out;
			}
			out.push_back(std::pair<DataBlock, Processor>(DataBlock(dataPtr, gpuSizes[i]), processors[i]));
			restSize -= gpuSizes[i];
			dataPtr += gpuSizes[i];
    	}
		auto threadNum = omp_get_max_threads() - gpuNum;
		if (threadNum == 0)
			return out;
		auto threadSize = restSize / threadNum;
		auto lastThreadSize = restSize - threadSize * threadNum;
    	for (int i = 0; i < threadNum; i++)
    	{
			out.push_back(std::pair<DataBlock, Processor>(DataBlock(dataPtr, threadSize), Processor{ i, cpu }));
			dataPtr += threadSize;
    	}
		out.back().first.second += lastThreadSize;
		return out;
    }
};