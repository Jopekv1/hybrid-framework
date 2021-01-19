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

		cudaMallocHost(&srcHost, dataSize * sizeof(int));
		cudaMallocHost(&dstHost, dataSize * sizeof(int));

		cudaMalloc(&src, dataSize * sizeof(int));
		cudaMalloc(&dst, dataSize * sizeof(int));

		for (uint64_t i = 0; i < dataSize; i++) {
			srcHost[i] = 7;
			dstHost[i] = 8;
		}

		cudaStreamCreate(&ownStream);

		cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
		cudaMemcpyAsync(dst, dstHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);

		std::cout << "Data initialized" << std::endl;
	}

	~VecAddKernel() {
		cudaFree(dst);
		cudaFree(src);

		cudaFree(dstHost);
		cudaFree(srcHost);

		cudaStreamDestroy(ownStream);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dstHost[i] = (int)pow((double)srcHost[i],(double)dstHost[i]);
		}
	};

	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		int blockSize = 1024;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		add<<<numBlocks, blockSize,0, ownStream>>>(workGroupSize, src + workItemId, dst + workItemId);
		cudaMemcpyAsync(dstHost + workItemId, dst + workItemId, workGroupSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
	};

	int* src = nullptr;
	int* dst = nullptr;
	int* srcHost = nullptr;
	int* dstHost = nullptr;

	cudaStream_t ownStream;
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

	verify(kernel.dstHost, dataSize);
}

TEST(vectorAdd, gpuSimulation) {
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

	cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
	cudaMemcpyAsync(dst, dstHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);

	std::cout << "Data initialized" << std::endl;

	int blockSize = 1024;
	int numBlocks = (dataSize + blockSize - 1) / blockSize;

	auto start = std::chrono::steady_clock::now();
	add<<<numBlocks, blockSize, 0, ownStream>>>(dataSize, src, dst);
	cudaMemcpyAsync(dstHost, dst, dataSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	verify(dstHost, dataSize);

	cudaFree(dst);
	cudaFree(src);

	cudaFree(dstHost);
	cudaFree(srcHost);

	cudaStreamDestroy(ownStream);
}