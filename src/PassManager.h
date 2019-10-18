#pragma once
#include <utility>
#include "Algorithm.h"

template<typename Type>
class PassManager
{
public:
	using DataBlock = std::pair<Type*, std::uint64_t>;
	
	PassManager() = default;;
	~PassManager() = default;;

	DataBlock run(DataBlock data, Algorithm<Type> algorithm) {};

private:

};