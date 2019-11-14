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
	const int config_number = 5;
	const int config_start = 0;
	const int gpu = 0;
	int config = config_start;

	const int power_of_two_number = 10;
	const int power_of_two_start = 0;
	int power_of_two = power_of_two_start;

	const int multiplier_of_512_number = 8;
	const int multiplier_of_512_start = 2;
	int multiplier_of_512 = multiplier_of_512_start;

	const int small_test_number = 10;
	const int big_test_number = 4;
	int limit = big_test_number;


	printf("SORT - FLOAT\n");
	for (config = config_start; config < config_number; config++)
	{
		printf("Config: %i\n", config);
		for (power_of_two = power_of_two_start; power_of_two < power_of_two_number; power_of_two++)
		{
			printf("Size 2^%i MB\n", power_of_two);
			std::uint64_t inSize = std::uint64_t(int(pow(2, power_of_two))) * 1024 * 1024 / sizeof(float);
			for (int k = 0; k < small_test_number; k++)
			{
				float* inData;
				cudaMallocManaged(&inData, inSize * sizeof(float));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = Sort<float>::DataBlock;
				PassManager<float, Sort<float>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n");
		}
		for (multiplier_of_512 = multiplier_of_512_start; multiplier_of_512 < multiplier_of_512_number; multiplier_of_512++)
		{
			printf("Size 512*%i MB\n", multiplier_of_512);
			std::uint64_t inSize = std::uint64_t(512 * multiplier_of_512) * 1024 * 1024 / sizeof(float);
			limit = big_test_number;
			if (config == gpu) limit = small_test_number;
			for (int k = 0; k < limit; k++)
			{
				float* inData;
				cudaMallocManaged(&inData, inSize * sizeof(float));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = Sort<float>::DataBlock;
				PassManager<float, Sort<float>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n\n");
		}
	}

	printf("SORT - INT\n");
	for (config = config_start; config < config_number; config++)
	{
		printf("Config: %i\n", config);
		for (power_of_two = power_of_two_start; power_of_two < power_of_two_number; power_of_two++)
		{
			printf("Size 2^%i MB\n", power_of_two);
			std::uint64_t inSize = std::uint64_t(int(pow(2, power_of_two))) * 1024 * 1024 / sizeof(int);
			for (int k = 0; k < small_test_number; k++)
			{
				int* inData;
				cudaMallocManaged(&inData, inSize * sizeof(int));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <int> (std::rand()) / static_cast <int> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = Sort<int>::DataBlock;
				PassManager<int, Sort<int>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n");
		}
		for (multiplier_of_512 = multiplier_of_512_start; multiplier_of_512 < multiplier_of_512_number; multiplier_of_512++)
		{
			printf("Size 512*%i MB\n", multiplier_of_512);
			std::uint64_t inSize = std::uint64_t(512 * multiplier_of_512) * 1024 * 1024 / sizeof(int);
			limit = big_test_number;
			if (config == gpu) limit = small_test_number;
			for (int k = 0; k < limit; k++)
			{
				int* inData;
				cudaMallocManaged(&inData, inSize * sizeof(int));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <int> (std::rand()) / static_cast <int> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = Sort<int>::DataBlock;
				PassManager<int, Sort<int>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n\n");
		}
	}

	printf("REDUCE - FLOAT\n");
	for (config = config_start; config < config_number; config++)
	{
		printf("Config: %i\n", config);
		for (power_of_two = power_of_two_start; power_of_two < power_of_two_number; power_of_two++)
		{
			printf("Size 2^%i MB\n", power_of_two);
			std::uint64_t inSize = std::uint64_t(int(pow(2, power_of_two))) * 1024 * 1024 / sizeof(float);
			for (int k = 0; k < small_test_number; k++)
			{
				float* inData;
				cudaMallocManaged(&inData, inSize * sizeof(float));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = MaxReduction<float>::DataBlock;
				PassManager<float, MaxReduction<float>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n");
		}
		for (multiplier_of_512 = multiplier_of_512_start; multiplier_of_512 < multiplier_of_512_number; multiplier_of_512++)
		{
			printf("Size 512*%i MB\n", multiplier_of_512);
			std::uint64_t inSize = std::uint64_t(512 * multiplier_of_512) * 1024 * 1024 / sizeof(float);
			limit = big_test_number;
			if (config == gpu) limit = small_test_number;
			for (int k = 0; k < limit; k++)
			{
				float* inData;
				cudaMallocManaged(&inData, inSize * sizeof(float));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = MaxReduction<float>::DataBlock;
				PassManager<float, MaxReduction<float>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n\n");
		}
	}

	printf("REDUCE - INT\n");
	for (config = config_start; config < config_number; config++)
	{
		printf("Config: %i\n", config);
		for (power_of_two = power_of_two_start; power_of_two < power_of_two_number; power_of_two++)
		{
			printf("Size 2^%i MB\n", power_of_two);
			std::uint64_t inSize = std::uint64_t(int(pow(2, power_of_two))) * 1024 * 1024 / sizeof(int);
			for (int k = 0; k < small_test_number; k++)
			{
				int* inData;
				cudaMallocManaged(&inData, inSize * sizeof(int));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <int> (std::rand()) / static_cast <int> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = MaxReduction<int>::DataBlock;
				PassManager<int, MaxReduction<int>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n");
		}
		for (multiplier_of_512 = multiplier_of_512_start; multiplier_of_512 < multiplier_of_512_number; multiplier_of_512++)
		{
			printf("Size 512*%i MB\n", multiplier_of_512);
			std::uint64_t inSize = std::uint64_t(512 * multiplier_of_512) * 1024 * 1024 / sizeof(int);
			limit = big_test_number;
			if (config == gpu) limit = small_test_number;
			for (int k = 0; k < limit; k++)
			{
				int* inData;
				cudaMallocManaged(&inData, inSize * sizeof(int));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <int> (std::rand()) / static_cast <int> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = MaxReduction<int>::DataBlock;
				PassManager<int, MaxReduction<int>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n\n");
		}
	}

	printf("SEARCH - FLOAT\n");
	for (config = config_start; config < config_number; config++)
	{
		printf("Config: %i\n", config);
		for (power_of_two = power_of_two_start; power_of_two < power_of_two_number; power_of_two++)
		{
			printf("Size 2^%i MB\n", power_of_two);
			std::uint64_t inSize = std::uint64_t(int(pow(2, power_of_two))) * 1024 * 1024 / sizeof(float);
			for (int k = 0; k < small_test_number; k++)
			{
				float* inData;
				cudaMallocManaged(&inData, inSize * sizeof(float));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = BinarySearch<float>::DataBlock;
				PassManager<float, BinarySearch<float>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n");
		}
		for (multiplier_of_512 = multiplier_of_512_start; multiplier_of_512 < multiplier_of_512_number; multiplier_of_512++)
		{
			printf("Size 512*%i MB\n", multiplier_of_512);
			std::uint64_t inSize = std::uint64_t(512 * multiplier_of_512) * 1024 * 1024 / sizeof(float);
			limit = big_test_number;
			if (config == gpu) limit = small_test_number;
			for (int k = 0; k < limit; k++)
			{
				float* inData;
				cudaMallocManaged(&inData, inSize * sizeof(float));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = BinarySearch<float>::DataBlock;
				PassManager<float, BinarySearch<float>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n\n");
		}
	}

	printf("SEARCH - INT\n");
	for (config = config_start; config < config_number; config++)
	{
		printf("Config: %i\n", config);
		for (power_of_two = power_of_two_start; power_of_two < power_of_two_number; power_of_two++)
		{
			printf("Size 2^%i MB\n", power_of_two);
			std::uint64_t inSize = std::uint64_t(int(pow(2, power_of_two))) * 1024 * 1024 / sizeof(int);
			for (int k = 0; k < small_test_number; k++)
			{
				int* inData;
				cudaMallocManaged(&inData, inSize * sizeof(int));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <int> (std::rand()) / static_cast <int> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = BinarySearch<int>::DataBlock;
				PassManager<int, BinarySearch<int>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n");
		}
		for (multiplier_of_512 = multiplier_of_512_start; multiplier_of_512 < multiplier_of_512_number; multiplier_of_512++)
		{
			printf("Size 512*%i MB\n", multiplier_of_512);
			std::uint64_t inSize = std::uint64_t(512 * multiplier_of_512) * 1024 * 1024 / sizeof(int);
			limit = big_test_number;
			if (config == gpu) limit = small_test_number;
			for (int k = 0; k < limit; k++)
			{
				int* inData;
				cudaMallocManaged(&inData, inSize * sizeof(int));
				for (std::uint64_t i = 0; i < inSize; ++i)
				{
					inData[i] = static_cast <int> (std::rand()) / static_cast <int> (RAND_MAX);
				}

				auto start = std::chrono::steady_clock::now();
				using DataBlock = BinarySearch<int>::DataBlock;
				PassManager<int, BinarySearch<int>> pm(config);
				auto x = pm.run(DataBlock(inData, inSize));
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				printf("%f\n", elapsed_seconds.count());
				cudaFree(inData);
			}
			printf("\n\n");
		}
	}
	return 0;
}