#pragma once

#include "Algorithm.h"
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

template<typename Type>
class Sort : public Algorithm<Type>
{
public:
    using DataBlock = typename Algorithm<Type>::DataBlock;

	Sort(DataBlock params): Algorithm<Type>()
    {
    }

    DataBlock runBaseCPU(DataBlock data) override;
    DataBlock runBaseGPU(DataBlock data);
    DataBlock runGPU(DataBlock data) override;
    void preDivision(DataBlock data) override;
    bool isBase(DataBlock data) override;
    std::uint32_t getChildrenNum(DataBlock data) override;
    std::pair<Type*, std::uint64_t> getChild(DataBlock data, std::uint32_t idx) override;
    DataBlock merge(std::vector<DataBlock> data) override;
    DataBlock mergeBlocks(std::vector<DataBlock> data) override;

    virtual ~Sort() {};

private:
    std::vector<std::uint64_t> partitionIdx{};
};

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::runBaseCPU(DataBlock data)
{
    partitionIdx.pop_back();
    return data;
}

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::runBaseGPU(DataBlock data)
{
    thrust::device_vector<Type> dv(data.first, data.first + data.second);
    thrust::stable_sort(thrust::device, dv.begin(), dv.end());
    cudaDeviceSynchronize();
    thrust::copy(dv.begin(), dv.end(), data.first);
    cudaDeviceSynchronize();
    return data;
}

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::runGPU(DataBlock data)
{
    DataBlock out;
    auto two_and_half = std::uint64_t(512) * 1024 * 1024 * 5 / sizeof(Type);
    auto size = data.second;
    if (size > two_and_half)
    {
        std::vector<DataBlock> blocks;
        std::vector<DataBlock> outs;
        int counter = 0;
        do
        {
            blocks.push_back(DataBlock(data.first + two_and_half * counter, two_and_half));
            counter++;
            size -= two_and_half;
        } while (size > two_and_half);
        if (size > 0)
        {
            blocks.push_back(DataBlock(data.first + two_and_half * counter, size));
        }
        for (int i = 0; i < blocks.size(); i++)
        {
            outs.push_back(runBaseGPU(blocks[i]));
        }
        out = mergeBlocks(outs);
    }
    else
    {
        out = runBaseGPU(data);
    }

    return out;
}

template <typename Type>
void Sort<Type>::preDivision(DataBlock data)
{
    std::uint64_t i = 0, j = data.second - 1;
    if ((i == 0 && j == 0) || data.second == 0)
    {
        partitionIdx.push_back(0);
        return;
    }
    Type* dataPtr = data.first;
    Type pivot = dataPtr[data.second / 2];
    
    while (i <= j)
    {
        while (dataPtr[i] < pivot)
            ++i;
        while (dataPtr[j] > pivot)
            --j;
        if (i <= j)
        {
            std::swap(dataPtr[i++], dataPtr[j--]);
        }
    }
    partitionIdx.push_back(i);
}

template <typename Type>
bool Sort<Type>::isBase(DataBlock data)
{
    auto idx = partitionIdx.back();
    if (data.second == 1 || data.second == 2)
        return true;
    if (0 < idx - 1 || idx < data.second - 1)
        return false;
    return true;
}

template <typename Type>
std::uint32_t Sort<Type>::getChildrenNum(DataBlock data)
{
    return 2;
}

template <typename Type>
std::pair<Type*, std::uint64_t> Sort<Type>::getChild(DataBlock data, std::uint32_t idx)
{
    auto pivot = partitionIdx.back();
    if (idx == 0)
        return DataBlock(data.first, pivot);
    return DataBlock(data.first + pivot, data.second - pivot);
}

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::merge(std::vector<DataBlock> data)
{
    partitionIdx.pop_back();
    if (data.size() == std::uint64_t(2))
        return DataBlock(data[0].first, data[0].second + data[1].second);
    return data[0];
}

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::mergeBlocks(std::vector<DataBlock> data)
{
    std::sort(data.begin(), data.end(), [](DataBlock a, DataBlock b) { return a.first < b.first; });
    auto firstDataPtr = data.front().first;
    std::uint64_t size = data.front().second;
    for (int i = 1; i < data.size(); i++)
    {
        std::inplace_merge(firstDataPtr, data[i].first, data[i].first + data[i].second - 1);
        size += data[i].second;
    }
    return DataBlock(data.front().first, size);
}
