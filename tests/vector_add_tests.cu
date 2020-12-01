#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>

constexpr uint64_t dataSize = 250000000;

void verify(int* dst, int size) {
	std::cout << "Veryfying data..." << std::endl;
	bool correct = true;
	int errCnt = 0;
	for (uint64_t i = 0; i < size; i++) {
		if (dst[i] != 21) {
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
void add(int n, int* src, int* src1, int* src2, int* src3, int* src4, int* dst)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		dst[index] = src[index] + src1[index] + src2[index] + src3[index] + src4[index] + dst[index];
	}
}

class VecAddKernel : public Kernel {
public:

	VecAddKernel() {
		cudaMallocManaged(&src, dataSize * sizeof(int));
		cudaMallocManaged(&src1, dataSize * sizeof(int));
		cudaMallocManaged(&src2, dataSize * sizeof(int));
		cudaMallocManaged(&src3, dataSize * sizeof(int));
		cudaMallocManaged(&src4, dataSize * sizeof(int));
		cudaMallocManaged(&dst, dataSize * sizeof(int));
		for (uint64_t i = 0; i < dataSize; i++) {
			src[i] = 1;
			src1[i] = 2;
			src2[i] = 3;
			src3[i] = 4;
			src4[i] = 5;
			dst[i] = 6;
		}
	}

	~VecAddKernel() {
		cudaFree(dst);
		cudaFree(src);
		cudaFree(src1);
		cudaFree(src2);
		cudaFree(src3);
		cudaFree(src4);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dst[i] += (src[i] + src1[i] + src2[i] + src3[i] + src4[i]);
		}
	};
	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		int blockSize = 1024;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		add<<<numBlocks, blockSize>>>(workGroupSize, src + workItemId, src1 + workItemId, src2 + workItemId, src3 + workItemId, src4 + workItemId, dst + workItemId);
		//cudaDeviceSynchronize();
	};

	int* src = nullptr;
	int* src1 = nullptr;
	int* src2 = nullptr;
	int* src3 = nullptr;
	int* src4 = nullptr;
	int* dst = nullptr;
};


TEST(vectorAdd, hybrid) {
	VecAddKernel kernel;

	LoadBalancer balancer;

	auto start = std::chrono::steady_clock::now();
	balancer.execute(&kernel, dataSize);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";

	verify(kernel.dst, dataSize);
}

TEST(vectorAdd, gpu) {
	int* src = nullptr;
	int* src1 = nullptr;
	int* src2 = nullptr;
	int* src3 = nullptr;
	int* src4 = nullptr;
	int* dst = nullptr;

	cudaMallocManaged(&src, dataSize * sizeof(int));
	cudaMallocManaged(&src1, dataSize * sizeof(int));
	cudaMallocManaged(&src2, dataSize * sizeof(int));
	cudaMallocManaged(&src3, dataSize * sizeof(int));
	cudaMallocManaged(&src4, dataSize * sizeof(int));
	cudaMallocManaged(&dst, dataSize * sizeof(int));

	for (uint64_t i = 0; i < dataSize; i++) {
		src[i] = 1;
		src1[i] = 2;
		src2[i] = 3;
		src3[i] = 4;
		src4[i] = 5;
		dst[i] = 6;
	}

	int blockSize = 1024;
	int numBlocks = (dataSize + blockSize - 1) / blockSize;

	auto start = std::chrono::steady_clock::now();
	add<<<numBlocks, blockSize>>>(dataSize, src, src1, src2, src3, src4, dst);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";

	verify(dst, dataSize);

	cudaFree(dst);
	cudaFree(src);
	cudaFree(src1);
	cudaFree(src2);
	cudaFree(src3);
	cudaFree(src4);
}