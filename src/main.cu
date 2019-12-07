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

template<class Callable, class... Args>
void timeWrapper(Callable f, Args... args) {
    auto start = std::chrono::steady_clock::now();
    f(args...);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    printf("%f\n", elapsed_seconds.count());
}

int main() {
	Engine engine;
	float* ptr;
	std::uint64_t size = std::uint64_t(1024) * 1024 * 1024 * 13 / 4;
	engine.allocateMemoryInCuda(ptr, size);
	for (std::uint64_t i = 0; i < size; i++)
	{
		ptr[i] = i;
	}
	ptr[12345] = std::uint64_t(3489660928) + 1000;
	using DataBlock = std::pair<float*, std::uint64_t>;
	LoadBalancer<float> lb;
	PassManager<float, MaxReduction<float>> pm(engine, lb);
	auto start = std::chrono::steady_clock::now();
	auto out = pm.run(DataBlock(ptr, size), DataBlock(0, 0));
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f %f\n", elapsed_seconds.count(), out.first[0]);
    return 0;
}