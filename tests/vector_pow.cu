#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cmath>

constexpr uint64_t dataSize = 2684353186 / 4;
constexpr uint64_t gpuAllocSize = 1073741824 / 4;

const double e = std::exp(1.0);

void verifyVectorPow(double* dst, double* src) {
	std::cout << "Veryfying data..." << std::endl;
	bool correct = true;
	for (uint64_t i = 0; i < dataSize; i++) {
		auto expercted = pow(src[i], e);
		auto value = dst[i];
		if (value < expercted - 0.01 || value > expercted + 0.01) {
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
void pow(int n, double* src, double* dst, double e) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		dst[index] = pow(src[index], e);
	}
}

class VecPowKernel : public Kernel {
public:

	VecPowKernel() {
		std::cout << "Initializing data..." << std::endl;

		cudaMallocHost(&srcHost, dataSize * sizeof(double));
		cudaMallocHost(&dstHost, dataSize * sizeof(double));

		cudaMalloc(&src, gpuAllocSize * sizeof(double));
		cudaMalloc(&dst, gpuAllocSize * sizeof(double));

		double lower_bound = 0;
		double upper_bound = 100000;
		std::uniform_real_distribution<double> distribution(lower_bound, upper_bound);
		std::default_random_engine randomEngine;

		for (uint64_t i = 0; i < dataSize; i++) {
			srcHost[i] = distribution(randomEngine);
			dstHost[i] = 0;
		}

		cudaStreamCreate(&ownStream);

		std::cout << "Data initialized" << std::endl;
	}

	~VecPowKernel() {
		cudaFree(dst);
		cudaFree(src);

		cudaFreeHost(dstHost);
		cudaFreeHost(srcHost);

		cudaStreamDestroy(ownStream);
	}

	void runCpu(uint64_t workItemId, uint64_t workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dstHost[i] = pow(srcHost[i], e);
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
			int numBlocks = (size + blockSize - 1) / blockSize;

			cudaMemcpyAsync(src, srcHost + workItemId + i, size * sizeof(double), cudaMemcpyHostToDevice, ownStream);
			pow<<<numBlocks, blockSize, 0, ownStream>>>(size, src, dst, e);
			cudaMemcpyAsync(dstHost + workItemId + i, dst, size * sizeof(double), cudaMemcpyDeviceToHost, ownStream);
			cudaStreamSynchronize(ownStream);

			i += size;
		}
	};

	double* src = nullptr;
	double* dst = nullptr;
	double* srcHost = nullptr;
	double* dstHost = nullptr;

	cudaStream_t ownStream;
};

class VectorPowFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>> {
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
			//GTEST_SKIP();
		}
	}

	uint64_t workGroupSize = 0;
	uint64_t gpuWorkGroups = 0;
	int numThreads = 0;
};

TEST_P(VectorPowFixture, hybrid) {
	VecPowKernel kernel;

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	//verifyVectorPow(kernel.dstHost, kernel.srcHost);

	auto hybridFile = fopen("results_hybrid.txt", "a");
	fprintf(hybridFile, "VectorPow %llu %llu %d %Lf\n", workGroupSize, gpuWorkGroups, numThreads, elapsed_seconds.count());
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

INSTANTIATE_TEST_SUITE_P(VectorPow,
	VectorPowFixture,
	::testing::Combine(
		::testing::ValuesIn(workGroupSizesValues),
		::testing::ValuesIn(gpuWorkGroupsValues),
		::testing::ValuesIn(numThreadsValues)));

TEST(VectorPow, gpu) {
	VecPowKernel kernel;

	auto start = std::chrono::steady_clock::now();

	kernel.runGpu(0u, 0u, dataSize);

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	//verifyVectorPow(kernel.dstHost, kernel.srcHost);

	auto gpuFile = fopen("results_gpu.txt", "a");
	fprintf(gpuFile, "VectorPow %Lf\n", elapsed_seconds.count());
	fclose(gpuFile);
}