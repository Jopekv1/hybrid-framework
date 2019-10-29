#pragma once

#include "Algorithm.h"
#include <algorithm>
#include <cuda_runtime.h>

template<typename Type>
class Sort : public Algorithm<Type>
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;

	DataBlock runBaseCPU(DataBlock data) override;
	DataBlock runGPU(DataBlock data) const override { return data; }
	void preDivision(DataBlock data) override;
	bool isBase(DataBlock data) const override;
	std::uint32_t getChildrenNum(DataBlock data) const override;
	std::pair<Type*, std::uint64_t> getChild(DataBlock data, std::uint32_t idx) const override;
	DataBlock merge(std::vector<DataBlock> data) const override;
	DataBlock mergeBlocks(std::vector<DataBlock> data) const override;

	virtual ~Sort() {};

private:
	void quickSort(Type* data, std::uint64_t left, std::uint64_t right);
	std::vector<std::uint64_t> partitionIdx{};
};

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::runBaseCPU(DataBlock data)
{
	partitionIdx.pop_back();
	return data;
}

template <typename Type>
void Sort<Type>::preDivision(DataBlock data)
{
	std::uint64_t i = 0, j = data.second - 1;
	Type tmp;
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
bool Sort<Type>::isBase(DataBlock data) const
{
	auto idx = partitionIdx.back();
	if (0 < idx - 1 || idx < data.second - 1)
		return false;
	return true;
}

template <typename Type>
std::uint32_t Sort<Type>::getChildrenNum(DataBlock data) const
{
	auto idx = partitionIdx.back();
	std::uint32_t ret = 0;
	if (0 < idx - 1)
		++ret;
	if (idx < data.second - 1)
		++ret;
	return ret;
}

template <typename Type>
std::pair<Type*, std::uint64_t> Sort<Type>::getChild(DataBlock data, std::uint32_t idx) const
{
	auto pivot = partitionIdx.back();
	if (idx == 0 && 0 < pivot - 1)
		return DataBlock(data, pivot - 1);
	return DataBlock(data + pivot, data.second - pivot);
}

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::merge(std::vector<DataBlock> data) const
{
	if (data.size == 2)
		return DataBlock(data[0].first, data[0].second + data[1].second);
	return data[0];
}

template <typename Type>
typename Sort<Type>::DataBlock Sort<Type>::mergeBlocks(std::vector<DataBlock> data) const
{
	auto firstDataPtr = data.front().first;
	for (int i = 1; i < data.size(); i++)
		std::inplace_merge(firstDataPtr, data[i].first, data[i].first + data[i].second - 1);
}
