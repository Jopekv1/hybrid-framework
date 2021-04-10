#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr uint64_t dataSize = 100000000;

void verifyVectorPow(int* dst) {
	std::cout << "Veryfying data..." << std::endl;
	bool correct = true;
	for (uint64_t i = 0; i < dataSize; i++) {
		if (dst[i] != 5764801) {
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
void add(int n, int* src, int* dst) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		dst[index] = (int)pow((double)src[index], (double)dst[index]);
	}
}

class VecAddKernel : public Kernel {
public:

	VecAddKernel() {
		std::cout << "Initializing data..." << std::endl;

		cudaMallocHost(&srcHost, dataSize * sizeof(int));
		cudaMallocHost(&dstHost, dataSize * sizeof(int));

		cudaMalloc(&src, dataSize * sizeof(int));
		cudaMalloc(&dst, dataSize * sizeof(int));

		for (uint64_t i = 0; i < dataSize; i++) {
			srcHost[i] = 7;
			dstHost[i] = 8;
		}

		cudaStreamCreate(&ownStream);

		std::cout << "Data initialized" << std::endl;
	}

	~VecAddKernel() {
		cudaFree(dst);
		cudaFree(src);

		cudaFreeHost(dstHost);
		cudaFreeHost(srcHost);

		cudaStreamDestroy(ownStream);
	}

	void runCpu(uint64_t workItemId, uint64_t workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dstHost[i] = (int)pow((double)srcHost[i], (double)dstHost[i]);
		}
	};

	void runGpu(uint64_t deviceId, uint64_t workItemId, uint64_t workGroupSize) override {
		int blockSize = 1024;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		cudaMemcpyAsync(src + workItemId, srcHost + workItemId, workGroupSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
		cudaMemcpyAsync(dst + workItemId, dstHost + workItemId, workGroupSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
		add<<<numBlocks, blockSize, 0, ownStream>>>(workGroupSize, src + workItemId, dst + workItemId);
		cudaMemcpyAsync(dstHost + workItemId, dst + workItemId, workGroupSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
	};

	int* src = nullptr;
	int* dst = nullptr;
	int* srcHost = nullptr;
	int* dstHost = nullptr;

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
	}

	uint64_t workGroupSize = 0;
	uint64_t gpuWorkGroups = 0;
	int numThreads = 0;
};

TEST_P(VectorPowFixture, hybrid) {
	VecAddKernel kernel;

	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	//verifyVectorPow(kernel.dstHost);

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
	std::cout << "Initializing data..." << std::endl;

	int* src = nullptr;
	int* dst = nullptr;
	int* srcHost = nullptr;
	int* dstHost = nullptr;

	cudaStream_t ownStream;

	cudaMallocHost(&srcHost, dataSize * sizeof(int));
	cudaMallocHost(&dstHost, dataSize * sizeof(int));

	cudaMalloc(&src, dataSize * sizeof(int));
	cudaMalloc(&dst, dataSize * sizeof(int));

	for (uint64_t i = 0; i < dataSize; i++) {
		srcHost[i] = 7;
		dstHost[i] = 8;
	}

	cudaStreamCreate(&ownStream);

	std::cout << "Data initialized" << std::endl;

	int blockSize = 1024;
	int numBlocks = (dataSize + blockSize - 1) / blockSize;

	auto start = std::chrono::steady_clock::now();

	cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
	cudaMemcpyAsync(dst, dstHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
	add<<<numBlocks, blockSize, 0, ownStream>>>(dataSize, src, dst);
	cudaMemcpyAsync(dstHost, dst, dataSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
	cudaDeviceSynchronize();

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	//verifyVectorPow(dstHost);

	cudaFree(dst);
	cudaFree(src);

	cudaFreeHost(dstHost);
	cudaFreeHost(srcHost);

	cudaStreamDestroy(ownStream);

	auto gpuFile = fopen("results_gpu.txt", "a");
	fprintf(gpuFile, "VectorPow %Lf\n", elapsed_seconds.count());
	fclose(gpuFile);
}