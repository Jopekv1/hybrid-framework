#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mutex>

constexpr uint64_t dataSize = 2684353186;
constexpr uint64_t gpuAllocSize = 1073741824;

void verifyMaxElement(thrust::host_vector<int>& host, int max) {
	std::cout << "Veryfying data..." << std::endl;
	auto maxExpected = std::max_element(host.begin(), host.end());
	if (max != *maxExpected) {
		std::cout << "!!!!! ERROR !!!!!" << std::endl;
		throw std::exception();
	}
	else {
		std::cout << "Results correct" << std::endl;
	}
}

class MaxElementKernel : public Kernel {
public:

	MaxElementKernel() {
		std::cout << "Initializing data..." << std::endl;

		srcHost.resize(dataSize);
		src.resize(gpuAllocSize);
		thrust::generate(thrust::host, srcHost.begin(), srcHost.end(), rand);

		cudaStreamCreate(&ownStream);
		thrust::cuda::par.on(ownStream);

		std::cout << "Data initialized" << std::endl;
	}

	~MaxElementKernel() {
		cudaStreamDestroy(ownStream);
	};

	void runCpu(uint64_t workItemId, uint64_t workGroupSize) override {
		auto max = std::max_element(srcHost.begin() + workItemId, srcHost.begin() + workItemId + workGroupSize);
		updateMax(*max);
	};

	void runGpu(uint64_t deviceId, uint64_t workItemId, uint64_t workGroupSize) override {
		uint64_t i = 0;
		while (i < workGroupSize) {
			auto size = gpuAllocSize;
			if (i + gpuAllocSize > workGroupSize) {
				size = workGroupSize - i;
			}

			int blockSize = 1024;
			int numBlocks = (size + blockSize - 1) / blockSize;

			cudaMemcpyAsync(thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(srcHost.data() + workItemId + i), size * sizeof(int), cudaMemcpyHostToDevice, ownStream);
			auto max = thrust::max_element(src.begin(), src.begin() + size);
			updateMax(*max);

			i += size;
		}
		//cudaMemcpyAsync(thrust::raw_pointer_cast(src.data() + workItemId), thrust::raw_pointer_cast(srcHost.data() + workItemId), workGroupSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
		//auto max = thrust::max_element(src.begin() + workItemId, src.begin() + workItemId + workGroupSize);
		//updateMax(*max);
	};

	void updateMax(int max) {
		std::lock_guard<std::mutex> lock(dstMutex);
		dst.push_back(max);
	}

	int merge() {
		return *std::max_element(dst.begin(), dst.end());
	}

	thrust::host_vector<int> srcHost;
	thrust::device_vector<int> src;

	cudaStream_t ownStream;

	std::vector<int> dst;
	std::mutex dstMutex;
};

class MaxElementFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>> {
public:

	void SetUp() override {
		std::tie(workGroupSize, gpuWorkGroups, numThreads) = GetParam();

		std::cout << "Test params: workGroupSize: " << workGroupSize << ", gpuWorkGroups: " << gpuWorkGroups << ", numThread: " << numThreads << std::endl;

		if (gpuWorkGroups * workGroupSize >= dataSize) {
			std::cout << "!!!!!!!!!!!!!!!!! GPU COVERS WHOLE DATA !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//GTEST_SKIP();
		}
		if (gpuWorkGroups * workGroupSize >= gpuAllocSize) {
			std::cout << "!!!!!!!!!!!!!!!!! GPU PACKAGE BIGGER THAN GPU ALLOC SIZE !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			GTEST_SKIP();
		}
	}

	uint64_t workGroupSize = 0;
	uint64_t gpuWorkGroups = 0;
	int numThreads = 0;
};

TEST_P(MaxElementFixture, hybrid) {
	MaxElementKernel kernel;

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();

	balancer.execute(&kernel, dataSize);
	auto max = kernel.merge();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	//verifyMaxElement(kernel.srcHost, max);

	auto hybridFile = fopen("results_hybrid.txt", "a");
	fprintf(hybridFile, "MaxElement %llu %llu %d %Lf\n", workGroupSize, gpuWorkGroups, numThreads, elapsed_seconds.count());
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

INSTANTIATE_TEST_SUITE_P(MaxElement,
	MaxElementFixture,
	::testing::Combine(
		::testing::ValuesIn(workGroupSizesValues),
		::testing::ValuesIn(gpuWorkGroupsValues),
		::testing::ValuesIn(numThreadsValues)));

TEST(MaxElement, gpu) {
	MaxElementKernel kernel;

	auto start = std::chrono::steady_clock::now();

	kernel.runGpu(0u, 0u, dataSize);
	auto max = kernel.merge();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	//verifyMaxElement(kernel.srcHost, max);

	auto gpuFile = fopen("results_gpu.txt", "a");
	fprintf(gpuFile, "MaxElement %Lf\n", elapsed_seconds.count());
	fclose(gpuFile);
}