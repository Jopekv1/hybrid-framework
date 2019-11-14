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
	std::uint64_t inSize = std::uint64_t(1024) * 1024 * 1024 / sizeof(float);
	float* inData;
	cudaMallocManaged(&inData, inSize * sizeof(float));

	for (std::uint64_t i = 0; i < inSize; ++i)
	{
		inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
	}

	Sort<float> mr;

	auto start = std::chrono::steady_clock::now();

	using DataBlock = Sort<float>::DataBlock;
	PassManager<float, Sort<float>> pm{};
	auto x = pm.run(DataBlock(inData, inSize));
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f %d\n", elapsed_seconds.count(), std::is_sorted(inData, inData + inSize));
	//for (int i = 0; i < x.second; i++)
		//printf("%f\n", x.first[i]);
	return 0;
}