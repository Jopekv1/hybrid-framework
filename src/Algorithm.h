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
	
	virtual DataBlock runBaseCPU(DataBlock data) = 0;
	virtual DataBlock runGPU(DataBlock data) const = 0;
	virtual void preDivision(DataBlock data) = 0;
	virtual bool isBase(DataBlock data) const = 0;
	virtual std::uint32_t getChildrenNum(DataBlock data) const = 0;
	virtual DataBlock getChild(DataBlock data, std::uint32_t num) const = 0;
	virtual DataBlock merge(std::vector<DataBlock> data) const = 0;
	virtual DataBlock mergeBlocks(std::vector<DataBlock> data) const = 0;
};
