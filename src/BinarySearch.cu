#pragma once

#include "Algorithm.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

template<typename Type>
class BinarySearch : public Algorithm<Type>
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;

	DataBlock runBaseCPU(DataBlock data) override;
	DataBlock runGPU(DataBlock data) override;
	void preDivision(DataBlock) override {};
	bool isBase(DataBlock data) override;
	std::uint32_t getChildrenNum(DataBlock) override { return 1; }
	std::pair<Type*, std::uint64_t> getChild(DataBlock data, std::uint32_t) override;
	DataBlock merge(std::vector<DataBlock> data) override;
	DataBlock mergeBlocks(std::vector<DataBlock> data) override { return merge(data); }

	virtual ~BinarySearch() {};
private:
	const Type searchedValue = static_cast<Type> (rand());
};

template<typename Type>
auto BinarySearch<Type>::runBaseCPU(DataBlock data) -> DataBlock
{
	auto* ret = new Type;
	if (data.second == 1)
	{
		*ret = data.first[0] == searchedValue ? searchedValue : 0.0;
	}
	else if (data.second == 2)
	{
		if (data.first[0] == searchedValue || data.first[1] == searchedValue)
		{
			*ret = searchedValue;
		}
		else 
		{
			*ret = 0;
		}
	}
	else
	{
		*ret = data.first[(data.second) / 2] == searchedValue ? searchedValue : 0.0;
	}
	return DataBlock(ret, 1);
}

template<typename Type>
auto BinarySearch<Type>::runGPU(DataBlock data) -> DataBlock
{
	auto* ret = new Type; // move to engine
	*ret = thrust::binary_search(data.first, data.first + data.second, searchedValue, thrust::less<Type>()) ? searchedValue : 0;
	cudaDeviceSynchronize();
	return DataBlock(ret, 1);
}

template <typename Type>
bool BinarySearch<Type>::isBase(DataBlock data)
{
	return data.first[(data.second)/2] == searchedValue || data.second == 1 || data.second == 2;
}

template <typename Type>
std::pair<Type*, std::uint64_t> BinarySearch<Type>::getChild(DataBlock data, std::uint32_t) {
	if (data.first[(data.second)/2] > searchedValue) {
		return DataBlock(data.first, (data.second) / 2);
	} else {
		return DataBlock(data.first + (data.second) / 2 + 1, data.second % 2 == 0 ? (data.second) / 2 - 1 : (data.second) / 2);
	}
}

template <typename Type>
auto BinarySearch<Type>::merge(std::vector<DataBlock> data) -> DataBlock
{
	Type* ret = new Type; //move to engine
	*ret = std::numeric_limits<Type>::min();
	for (int i = 0; i < data.size(); i++)
	{
		Type b = data[i].first[0];
		*ret = std::max(*ret, b); // either searchedValue or 0.0 (if not found)
	}
	return DataBlock(ret, 1);
}

