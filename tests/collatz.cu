#include "kernel.h"
#include "load_balancer.h"
#include "configuration.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr uint64_t gpuAllocSize = 1073741824;

//void verifyCollatz(int* dst) {
//	std::cout << "Veryfying data..." << std::endl;
//	bool correct = true;
//	for (uint64_t i = 0; i < dataSize; i++) {
//		int counter = 0;
//		uint64_t value = i;
//		while (value > 1) {
//			if (value % 2 == 0) {
//				value = value / 2;
//			}
//			else {
//				value = 3 * value + 1;
//			}
//			counter++;
//		}
//		if (dst[i] != counter) {
//			correct = false;
//		}
//	}
//	if (correct) {
//		std::cout << "Results correct" << std::endl;
//	}
//	else {
//		std::cout << "!!!!! ERROR !!!!!" << std::endl;
//		throw std::exception();
//	}
//}

__global__
void collatz(uint64_t n, int* src, uint64_t offset) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		int counter = 0;
		uint64_t value = index + offset;
		while (value > 1) {
			if ((value % 2) == 0) {
				value = (value / 2);
			}
			else {
				value = (3 * value + 1);
			}
			counter = counter + 1;
		}
		src[index] = counter;
	}
}

class CollatzKernel : public Kernel {
public:

	CollatzKernel(uint64_t dataSize) {
		std::cout << "Initializing data..." << std::endl;

		cudaMallocHost(&srcHost, dataSize * sizeof(int));

		cudaMalloc(&src, gpuAllocSize * sizeof(int));

		cudaStreamCreate(&ownStream);

		std::cout << "Data initialized" << std::endl;
	}

	~CollatzKernel() {
		cudaFree(src);

		cudaFreeHost(srcHost);

		cudaStreamDestroy(ownStream);
	}

	void runCpu(uint64_t workItemId, uint64_t workGroupSize) override {
		for (uint64_t i = workItemId; i < workItemId + workGroupSize; i++) {
			int counter = 0;
			uint64_t value = i;
			while (value > 1) {
				if (value % 2 == 0) {
					value = value / 2;
				}
				else {
					value = 3 * value + 1;
				}
				counter++;
			}
			srcHost[i] = counter;
		}
	};

	void runGpu(uint64_t deviceId, uint64_t workItemId, uint64_t workGroupSize) override {
		uint64_t i = 0;
		while (i < workGroupSize) {
			auto size = gpuAllocSize;
			if (i + gpuAllocSize > workGroupSize) {
				size = workGroupSize - i;
			}

			int blockSize = 1024;
			int numBlocks = int((size + blockSize - 1) / blockSize);

			cudaMemcpyAsync(src, srcHost + workItemId + i, size * sizeof(int), cudaMemcpyHostToDevice, ownStream);
			collatz<<<numBlocks, blockSize, 0, ownStream>>>(size, src, workItemId + i);
			cudaMemcpyAsync(srcHost + workItemId + i, src, size * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
			cudaStreamSynchronize(ownStream);

			i += size;
		}
	};

	int* src = nullptr;
	int* srcHost = nullptr;

	cudaStream_t ownStream;
};

static uint64_t dataSizes[] = {
	1342177280,
	2684354560,
	5368709120,
	8053063680,
	10737418240,
	13421772800, };

class CollatzFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, uint64_t, int>> {
public:

	void SetUp() override {
		std::tie(dataSize, workGroupSize, gpuWorkGroups, numThreads) = GetParam();

		std::cout << "Test params: dataSize: " << dataSize << ", workGroupSize: " << workGroupSize << ", gpuWorkGroups: " << gpuWorkGroups << ", numThread: " << numThreads << std::endl;

		if (gpuWorkGroups * workGroupSize >= dataSize) {
			std::cout << "!!!!!!!!!!!!!!!!! GPU COVERS WHOLE DATA !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//GTEST_SKIP();
		}
		if (gpuWorkGroups * workGroupSize >= gpuAllocSize) {
			std::cout << "!!!!!!!!!!!!!!!!! GPU PACKAGE BIGGER THAN GPU ALLOC SIZE !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//GTEST_SKIP();
		}

		if (!Config::tunningMode) {
			if (!((workGroupSize == 10000 && gpuWorkGroups == 50000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 10000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 100000 && numThreads == 8) ||
				(workGroupSize == 100000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 100000 && numThreads == 8) ||
				(workGroupSize == 100000 && gpuWorkGroups == 10000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 1000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 10000 && numThreads == 8))) {
				GTEST_SKIP();
			}
		}

		if (Config::tunningMode) {
			if (dataSize != dataSizes[1]) {
				GTEST_SKIP();
			}
		}
	}

	uint64_t dataSize = 0;
	uint64_t workGroupSize = 0;
	uint64_t gpuWorkGroups = 0;
	int numThreads = 0;
};

TEST_P(CollatzFixture, hybrid) {
	CollatzKernel kernel(dataSize);

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	//verifyCollatz(kernel.srcHost);

	auto hybridFile = fopen("results_hybrid.txt", "a");
	fprintf(hybridFile, "Collatz %llu %llu %llu %d %f\n", dataSize, workGroupSize, gpuWorkGroups, numThreads, elapsed_seconds.count());
	fclose(hybridFile);
}

static uint64_t workGroupSizesValues[] = {
	10,
	100,
	1000,
	10000,
	100000 };

static uint64_t gpuWorkGroupsValues[] = {
	100,
	1000,
	10000,
	20000,
	50000,
	100000 };

static int numThreadsValues[] = {
	2,
	4,
	6,
	8 };

INSTANTIATE_TEST_SUITE_P(Collatz,
	CollatzFixture,
	::testing::Combine(
		::testing::ValuesIn(dataSizes),
		::testing::ValuesIn(workGroupSizesValues),
		::testing::ValuesIn(gpuWorkGroupsValues),
		::testing::ValuesIn(numThreadsValues)));

class CollatzGpuFixture : public ::testing::TestWithParam<uint64_t> {
public:

	void SetUp() override {
		dataSize = GetParam();

		if (Config::tunningMode) {
			if (dataSize != dataSizes[1]) {
				GTEST_SKIP();
			}
		}
	}

	uint64_t dataSize = 0;
};

TEST_P(CollatzGpuFixture, gpu) {
	CollatzKernel kernel(dataSize);

	auto start = std::chrono::steady_clock::now();
	kernel.runGpu(0, 0, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	//verifyCollatz(kernel.srcHost);

	auto gpuFile = fopen("results_gpu.txt", "a");
	fprintf(gpuFile, "Collatz %llu %f\n", dataSize, elapsed_seconds.count());
	fclose(gpuFile);
}

INSTANTIATE_TEST_SUITE_P(CollatzGpu,
	CollatzGpuFixture,
	::testing::ValuesIn(dataSizes));
