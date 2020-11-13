#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>

constexpr uint64_t dataSize = 200000000;

void verify(int* dst, int size) {
	std::cout << "Veryfying data..." << std::endl;
	bool correct = true;
	int errCnt = 0;
	for (uint64_t i = 0; i < size; i++) {
		if (dst[i] != 3) {
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
void add(int n, int* src, int* dst)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		dst[index] = src[index] + dst[index];
	}
}

class VecAddKernel : public Kernel {
public:

	VecAddKernel() {
		cudaMallocManaged(&src, dataSize * sizeof(int));
		cudaMallocManaged(&dst, dataSize * sizeof(int));
		for (uint64_t i = 0; i < dataSize; i++) {
			src[i] = 1;
			dst[i] = 2;
		}
	}

	~VecAddKernel() {
		cudaFree(dst);
		cudaFree(src);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dst[i] += src[i];
		}
	};
	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		int blockSize = 1024;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		add<<<numBlocks, blockSize>>>(workGroupSize, src + workItemId, dst + workItemId);
		//cudaDeviceSynchronize();
	};

	int* src = nullptr;
	int* dst = nullptr;
};


TEST(vectorAdd, hybrid) {
	VecAddKernel kernel;

	LoadBalancer balancer;

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, 0, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	verify(kernel.dst, dataSize);
}

TEST(vectorAdd, gpu) {
	int* src = nullptr;
	int* dst = nullptr;

	cudaMallocManaged(&src, dataSize * sizeof(int));
	cudaMallocManaged(&dst, dataSize * sizeof(int));

	for (uint64_t i = 0; i < dataSize; i++) {
		src[i] = 1;
		dst[i] = 2;
	}

	int blockSize = 1024;
	int numBlocks = (dataSize + blockSize - 1) / blockSize;

	auto start = std::chrono::steady_clock::now();
	add<<<numBlocks, blockSize>>>(dataSize, src, dst);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	verify(dst, dataSize);

	cudaFree(dst);
	cudaFree(src);
}