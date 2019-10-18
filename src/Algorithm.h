#pragma once
#include <utility>
#include <cstdint>
#include <vector>

template <typename Type>
class Algorithm
{
public:
	using DataBlock = std::pair<Type*, std::uint64_t>;
	
	virtual ~Algorithm() = default;
	
	virtual DataBlock runCPU(DataBlock data) = 0;
	virtual DataBlock runGPU(DataBlock data) = 0;
	virtual bool isBase(DataBlock data) = 0;
	virtual std::vector<DataBlock> divide(DataBlock data) = 0;
	virtual DataBlock merge(std::vector<DataBlock> data) = 0;
};
