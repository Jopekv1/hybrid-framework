#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdint>
#include "MaxReduction.h"
#include <iostream>
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
	DataBlock out, out2;
	#pragma omp parallel num_threads(2)
	{
		int tid = omp_get_thread_num();
		if (tid == 1)
			out2 = mr.runGPU(DataBlock(inData + inSize / 2, inSize / 2));
		else
			out = mr.runCPU(DataBlock(inData, inSize / 2));
	}
	auto x = mr.merge({ out, out2 });
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f %f\n", elapsed_seconds.count(), x.first[0]);
	cudaFree(inData);
	return 0;
}