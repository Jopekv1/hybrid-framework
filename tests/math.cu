#include "kernel.h"
#include "load_balancer.h"
#include "configuration.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cmath>

#define SERIES 50

constexpr uint64_t gpuAllocSize = 1073741824 / 2;

const double e = std::exp(1.0);

__global__
void math(uint64_t n, double* src, double e, uint64_t offset) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		uint64_t i = index + offset;
		double sum = 0.0;
		for (int j = 0; j < SERIES; j++) {
			double x = pow(-1.0, double(j)) * pow(double(i), 2.0 * j + 1) / (2 * i + 1);
			sum += x;
		}
		src[index] = sum;
	}
}

class MathKernel : public Kernel {
public:

	MathKernel(uint64_t dataSize) {
		std::cout << "Initializing data..." << std::endl;

		int gpuCount;
		cudaGetDeviceCount(&gpuCount);

		cudaMallocHost(&srcHost, dataSize * sizeof(double));

		for (int i = 0; i < gpuCount; i++) {
			cudaSetDevice(i);
			
			double* tSrc = nullptr;
			cudaStream_t tOwnStream;

			cudaMalloc(&tSrc, gpuAllocSize * sizeof(double));

			cudaStreamCreate(&tOwnStream);

			src.push_back(tSrc);
			ownStream.push_back(tOwnStream);
		}

		cudaSetDevice(0);

		std::cout << "Data initialized" << std::endl;
	}

	~MathKernel() {
		int gpuCount;
		cudaGetDeviceCount(&gpuCount);

		cudaFreeHost(srcHost);

		for (int i = 0; i < gpuCount; i++) {
			cudaSetDevice(i);
		
			cudaFree(src[i]);

			cudaStreamDestroy(ownStream[i]);
		}

		cudaSetDevice(0);
	}

	void runCpu(uint64_t workItemId, uint64_t workGroupSize) override {
		for (uint64_t i = workItemId; i < workItemId + workGroupSize; i++) {
			double sum = 0.0;
			for (int j = 0; j < SERIES; j++) {
				double x = pow(-1, j) * pow(i, 2 * j + 1) / (2 * i + 1);
				sum += x;
			}
			srcHost[i] = sum;
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

			math<<<numBlocks, blockSize, 0, ownStream[deviceId]>>>(size, src[deviceId], e, workItemId + i);
			cudaMemcpyAsync(srcHost + workItemId + i, src[deviceId], size * sizeof(double), cudaMemcpyDeviceToHost, ownStream[deviceId]);
			cudaStreamSynchronize(ownStream[deviceId]);

			i += size;
		}
	};

	double* srcHost = nullptr;
	std::vector<double*> src;

	std::vector<cudaStream_t> ownStream;
};

static uint64_t dataSizes[] = {
	1342177280 / 2,
	2684354560 / 2,
	5368709120 / 2,
	8053063680 / 2,
	10737418240 / 2,
	13421772800 / 2, };

class MathFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, uint64_t, int>> {
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
				(workGroupSize == 10000 && gpuWorkGroups == 100000 && numThreads == 8) ||
				(workGroupSize == 100000 && gpuWorkGroups == 100 && numThreads == 8) ||
				(workGroupSize == 100 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 1000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 10000 && numThreads == 8) ||
				(workGroupSize == 100 && gpuWorkGroups == 100000 && numThreads == 8) ||
				(workGroupSize == 10000 && gpuWorkGroups == 20000 && numThreads == 8) ||
				(workGroupSize == 1000 && gpuWorkGroups == 100000 && numThreads == 8))) {
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

TEST_P(MathFixture, hybrid) {
	MathKernel kernel(dataSize);

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	auto hybridFile = fopen("results_hybrid.txt", "a");
	fprintf(hybridFile, "Math %llu %llu %llu %d %f\n", dataSize, workGroupSize, gpuWorkGroups, numThreads, elapsed_seconds.count());
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

INSTANTIATE_TEST_SUITE_P(Math,
	MathFixture,
	::testing::Combine(
		::testing::ValuesIn(dataSizes),
		::testing::ValuesIn(workGroupSizesValues),
		::testing::ValuesIn(gpuWorkGroupsValues),
		::testing::ValuesIn(numThreadsValues)));

class MathGpuFixture : public ::testing::TestWithParam<uint64_t> {
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

TEST_P(MathGpuFixture, gpu) {
	int gpuCount;
	cudaGetDeviceCount(&gpuCount);

	MathKernel kernel(dataSize);
	LoadBalancer balancer(uint64_t(dataSize/gpuCount), 1, gpuCount);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	auto gpuFile = fopen("results_gpu.txt", "a");
	fprintf(gpuFile, "Math %llu %f\n", dataSize, elapsed_seconds.count());
	fclose(gpuFile);
}

INSTANTIATE_TEST_SUITE_P(MathGpu,
	MathGpuFixture,
	::testing::ValuesIn(dataSizes));

class MathCpuFixture : public ::testing::TestWithParam<uint64_t> {
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

TEST_P(MathCpuFixture, theoryCpu) {
	MathKernel kernel(dataSize);

	LoadBalancer balancer(uint64_t(dataSize / 8), 1, 8);
	balancer.forceDeviceCount(0);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "CPU time: " << elapsed_seconds.count() << "s\n";

	auto cpuFile = fopen("results_cpu.txt", "a");
	fprintf(cpuFile, "Math %llu %f\n", dataSize, elapsed_seconds.count());
	fclose(cpuFile);
}

INSTANTIATE_TEST_SUITE_P(MathCpu,
	MathCpuFixture,
	::testing::ValuesIn(dataSizes));
