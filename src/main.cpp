#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdint>
#include "MaxReduction.cuh"
#include <iostream>
#include "PassManager.hpp"

template<class Callable, class... Args>
void timeWrapper(Callable f, Args... args) {
	auto start = std::chrono::steady_clock::now();
	f(args...);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f\n", elapsed_seconds.count());
}

int main() {
	std::uint64_t inSize = std::uint64_t(2048) * 1024 * 1024 / sizeof(float);
	float* inData;
	cudaMallocManaged(&inData, inSize * sizeof(float));

	for (std::uint64_t i = 0; i < inSize; ++i)
	{
		inData[i] = i;
	}

	MaxReduction<float> mr;

	auto start = std::chrono::steady_clock::now();

	using DataBlock = MaxReduction<float>::DataBlock;
	PassManager<float> pm(&mr);
	auto x = pm.run(DataBlock(inData, inSize));
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f %f\n", elapsed_seconds.count(), x.first[0]);
	cudaFree(inData);
	return 0;
}