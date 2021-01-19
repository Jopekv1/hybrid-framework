#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr uint64_t dataSize = 1800000000;

void verify(int* dst, int size) {
	std::cout << "Veryfying data..." << std::endl;
	bool correct = true;
	int errCnt = 0;
	for (uint64_t i = 0; i < size; i++) {
		if (dst[i] != 5764801) {
			correct = false;
			errCnt++;
			//std::cout << i << std::endl;
		}
	}
	if (correct) {
		std::cout << "Results correct" << std::endl;
	}
	else {
		std::cout << "!!!!! ERROR !!!!!" << std::endl;
	}
}

__global__
void add(int n, int* src, int* dst){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		dst[index] = (int)pow((double)src[index], (double)dst[index]);
	}
}

class VecAddKernel : public Kernel {
public:

	VecAddKernel() {
		std::cout << "Initializing data..." << std::endl;

		cudaMallocManaged(&src, dataSize * sizeof(int));
		cudaMallocManaged(&dst, dataSize * sizeof(int));
		for (uint64_t i = 0; i < dataSize; i++) {
			src[i] = 7;
			dst[i] = 8;
		}

		std::cout << "Data initialized" << std::endl;
	}

	~VecAddKernel() {
		cudaFree(dst);
		cudaFree(src);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dst[i] = (int)pow((double)src[i],(double)dst[i]);
		}
	};
	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		int blockSize = 1024;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		add<<<numBlocks, blockSize>>>(workGroupSize, src + workItemId, dst + workItemId);
	};

	int* src = nullptr;
	int* dst = nullptr;
};

constexpr uint64_t cpuPackageSize = 1000;
constexpr uint64_t gpuPackageSize = 10000;

TEST(vectorAdd, hybrid) {
	VecAddKernel kernel;

	LoadBalancer balancer(cpuPackageSize, gpuPackageSize);

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	verify(kernel.dst, dataSize);
}

TEST(vectorAdd, gpuSimulation) {
	std::cout << "Initializing data..." << std::endl;

	int* src = nullptr;
	int* dst = nullptr;

	cudaMallocManaged(&src, dataSize * sizeof(int));
	cudaMallocManaged(&dst, dataSize * sizeof(int));

	for (uint64_t i = 0; i < dataSize; i++) {
		src[i] = 7;
		dst[i] = 8;
	}

	std::cout << "Data initialized" << std::endl;

	int blockSize = 1024;
	int numBlocks = (cpuPackageSize* gpuPackageSize + blockSize - 1) / blockSize;

	auto partialSize = dataSize / (cpuPackageSize * gpuPackageSize);

	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < partialSize; i++) {
		add<<<numBlocks, blockSize>>>(cpuPackageSize * gpuPackageSize, src + i * cpuPackageSize * gpuPackageSize, dst + i * cpuPackageSize * gpuPackageSize);
	}
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	verify(dst, dataSize);

	cudaFree(dst);
	cudaFree(src);
}