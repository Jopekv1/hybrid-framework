#include "kernel.h"
#include "load_balancer.h"
#include "configuration.h"

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

	MaxElementKernel(uint64_t dataSize) {
		std::cout << "Initializing data..." << std::endl;

		int gpuCount;
		cudaGetDeviceCount(&gpuCount);

		srcHost.resize(dataSize);

		for (int i = 0; i < gpuCount; i++) {
			cudaSetDevice(i);

			cudaStream_t tOwnStream;
			thrust::device_vector<int> tSrc;
			src.push_back(tSrc);

			src[i].resize(gpuAllocSize);
			cudaStreamCreate(&tOwnStream);
			ownStream.push_back(tOwnStream);
			thrust::cuda::par.on(ownStream[i]);
		}

		cudaSetDevice(0);

		thrust::generate(thrust::host, srcHost.begin(), srcHost.end(), rand);


		std::cout << "Data initialized" << std::endl;
	}

	~MaxElementKernel() {
		int gpuCount;
		cudaGetDeviceCount(&gpuCount);
		
		for (int i = 0; i < gpuCount; i++) {
			cudaSetDevice(i);
			cudaStreamDestroy(ownStream[i]);
		}

		cudaSetDevice(0);
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

			cudaMemcpyAsync(thrust::raw_pointer_cast(src[deviceId].data()), thrust::raw_pointer_cast(srcHost.data() + workItemId + i), size * sizeof(int), cudaMemcpyHostToDevice, ownStream[deviceId]);
			auto max = thrust::max_element(src[deviceId].begin(), src[deviceId].begin() + size);
			updateMax(*max);

			i += size;
		}
	};

	void updateMax(int max) {
		std::lock_guard<std::mutex> lock(dstMutex);
		dst.push_back(max);
	}

	int merge() {
		return *std::max_element(dst.begin(), dst.end());
	}

	thrust::host_vector<int> srcHost;
	std::vector<thrust::device_vector<int>> src;

	std::vector<cudaStream_t> ownStream;

	std::vector<int> dst;
	std::mutex dstMutex;
};

static uint64_t dataSizes[] = {
	1342177280,
	2684354560,
	5368709120,
	8053063680,
	10737418240,
	13421772800, };

class MaxElementFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, uint64_t, int>> {
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
			if (!((workGroupSize == 10000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 10000 && numThreads == 8) ||
				(workGroupSize == 100000 && gpuWorkGroups == 1000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 100000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 100000 && numThreads == 8) ||
				(workGroupSize == 100000 && gpuWorkGroups == 10000 && numThreads == 8) ||
				(workGroupSize == 100000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 50000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 50000 && numThreads == 6) ||
				(workGroupSize == 1000 && gpuWorkGroups == 50000 && numThreads == 8))) {
				GTEST_SKIP();
			}
		}

		if (Config::tunningMode) {
			if (dataSize != dataSizes[1]) {
				GTEST_SKIP();
			}
		}

		if (Config::theoryMode) {
			GTEST_SKIP();
		}
	}

	uint64_t dataSize = 0;
	uint64_t workGroupSize = 0;
	uint64_t gpuWorkGroups = 0;
	int numThreads = 0;
};

TEST_P(MaxElementFixture, hybrid) {
	MaxElementKernel kernel(dataSize);

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();

	balancer.execute(&kernel, dataSize);
	auto max = kernel.merge();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	//verifyMaxElement(kernel.srcHost, max);

	auto hybridFile = fopen("results_hybrid.txt", "a");
	fprintf(hybridFile, "MaxElement %llu %llu %llu %d %f\n", dataSize, workGroupSize, gpuWorkGroups, numThreads, elapsed_seconds.count());
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
		::testing::ValuesIn(dataSizes),
		::testing::ValuesIn(workGroupSizesValues),
		::testing::ValuesIn(gpuWorkGroupsValues),
		::testing::ValuesIn(numThreadsValues)));

class MaxElementGpuFixture : public ::testing::TestWithParam<uint64_t> {
public:

	void SetUp() override {
		dataSize = GetParam();

		if (Config::tunningMode) {
			if (dataSize != dataSizes[1]) {
				GTEST_SKIP();
			}
		}

		if (Config::theoryMode) {
			dataSize = gpuAllocSize/4;
			static bool runInTheoryMode = false;
			if (runInTheoryMode) {
				GTEST_SKIP();
			}
			runInTheoryMode = true;
		}
	}

	uint64_t dataSize = 0;
};

TEST_P(MaxElementGpuFixture, gpu) {
	int gpuCount;
	cudaGetDeviceCount(&gpuCount);

	MaxElementKernel kernel(dataSize);
	LoadBalancer balancer(uint64_t(dataSize/gpuCount), 1, gpuCount);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto max = kernel.merge();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	//verifyMaxElement(kernel.srcHost, max);

	auto gpuFile = fopen("results_gpu.txt", "a");
	fprintf(gpuFile, "MaxElement %llu %f\n", dataSize, elapsed_seconds.count());
	fclose(gpuFile);
}

INSTANTIATE_TEST_SUITE_P(MaxElementGpu,
	MaxElementGpuFixture,
	::testing::ValuesIn(dataSizes));

TEST(MaxElementTheory, theoryCpu) {
	if (!Config::theoryMode) {
		GTEST_SKIP();
	}

	MaxElementKernel kernel(gpuAllocSize/4);

	auto start = std::chrono::steady_clock::now();
	kernel.runCpu(0,gpuAllocSize/4);
	auto max = kernel.merge();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "CPU time: " << elapsed_seconds.count() << "s\n";

	//verifyCollatz(kernel.srcHost);

	auto cpuFile = fopen("results_cpu.txt", "a");
	fprintf(cpuFile, "MaxElement %llu %f\n", gpuAllocSize/4, elapsed_seconds.count());
	fclose(cpuFile);
}
