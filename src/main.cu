#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdint>
#include "MaxReduction.cu"
#include "Sort.cu"
#include "BinarySearch.cu"
#include <iostream>
#include "PassManager.hpp"
#include <omp.h>

int main() {
	Engine engine;
	float* ptr;
	std::uint64_t size = std::uint64_t(1024) * 1024 * 1024 * 7 / 4;
	engine.allocateMemoryInCuda(ptr, size);
	for (std::uint64_t i = 0; i < size; i++)
	{
		ptr[i] = i;
	}
	ptr[12345] = std::uint64_t(3489660928) + 1000;
	using DataBlock = std::pair<float*, std::uint64_t>;
	LoadBalancer<float> lb;
	PassManager<float, MaxReduction<float>> pm(engine, lb);
	auto out = pm.run(DataBlock(ptr, size), DataBlock(0, 0));
	printf("%f\n", out.first[0]);
    return 0;
}