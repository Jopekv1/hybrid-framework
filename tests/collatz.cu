#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr uint64_t dataSize = 1000000000;

void verifyCollatz(int* dst) {
	std::cout << "Veryfying data..." << std::endl;
	bool correct = true;
	for (uint64_t i = 0; i < dataSize; i++) {
		int counter = 0;
		int value = i;
		while (value > 1) {
			if (value % 2 == 0) {
				value = value / 2;
			}
			else {
				value = 3 * value + 1;
			}
			counter++;
		}
		if (dst[i] != counter) {
			correct = false;
		}
	}
	if (correct) {
		std::cout << "Results correct" << std::endl;
	}
	else {
		std::cout << "!!!!! ERROR !!!!!" << std::endl;
		throw std::exception();
	}
}

__global__
void collatz(int n, int* src, int offset) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		int counter = 0;
		int value = index + offset;
		while (value > 1) {
			if ((value % 2) == 0) {
				value = (value / 2);
			}
			else {
				value = (3 * value + 1);
			}
			counter = counter +1;
		}
		src[index] = counter;
	}
}

class CollatzKernel : public Kernel {
public:

	CollatzKernel() {
		std::cout << "Initializing data..." << std::endl;

		cudaMallocHost(&srcHost, dataSize * sizeof(int));

		cudaMalloc(&src, dataSize * sizeof(int));

		for (uint64_t i = 0; i < dataSize; i++) {
			srcHost[i] = 0;
		}

		cudaStreamCreate(&ownStream);

		cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);

		std::cout << "Data initialized" << std::endl;
	}

	~CollatzKernel() {
		cudaFree(src);

		cudaFreeHost(srcHost);

		cudaStreamDestroy(ownStream);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			int counter = 0;
			int value = i;
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

	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		int blockSize = 1024;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		collatz << <numBlocks, blockSize, 0, ownStream >> > (workGroupSize, src + workItemId, workItemId);
		cudaMemcpyAsync(srcHost + workItemId, src + workItemId, workGroupSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
	};

	int* src = nullptr;
	int* srcHost = nullptr;

	cudaStream_t ownStream;
};

class CollatzFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>> {
public:

	void SetUp() override {
		std::tie(workGroupSize, gpuWorkGroups, numThreads) = GetParam();

		std::cout << "Test params: workGroupSize: " << workGroupSize << ", gpuWorkGroups: " << gpuWorkGroups << ", numThread: " << numThreads << std::endl;

		if (gpuWorkGroups * workGroupSize >= dataSize) {
			std::cout << "!!!!!!!!!!!!!!!!! GPU COVERS WHOLE DATA !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//GTEST_SKIP();
		}
	}

	uint64_t workGroupSize = 0;
	uint64_t gpuWorkGroups = 0;
	int numThreads = 0;
};

TEST_P(CollatzFixture, hybrid) {
	CollatzKernel kernel;

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	//verifyCollatz(kernel.srcHost);
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
		::testing::ValuesIn(workGroupSizesValues),
		::testing::ValuesIn(gpuWorkGroupsValues),
		::testing::ValuesIn(numThreadsValues)));

TEST(Collatz, gpu) {
	std::cout << "Initializing data..." << std::endl;

	int* src = nullptr;
	int* srcHost = nullptr;

	cudaStream_t ownStream;

	cudaMallocHost(&srcHost, dataSize * sizeof(int));

	cudaMalloc(&src, dataSize * sizeof(int));

	for (uint64_t i = 0; i < dataSize; i++) {
		srcHost[i] = 0;
	}

	cudaStreamCreate(&ownStream);

	cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);

	std::cout << "Data initialized" << std::endl;

	int blockSize = 1024;
	int numBlocks = (dataSize + blockSize - 1) / blockSize;

	auto start = std::chrono::steady_clock::now();
	collatz<<<numBlocks, blockSize, 0, ownStream>>>(dataSize, src, 0);
	cudaMemcpyAsync(srcHost, src, dataSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	//verifyCollatz(srcHost);

	cudaFree(src);

	cudaFreeHost(srcHost);

	cudaStreamDestroy(ownStream);
}