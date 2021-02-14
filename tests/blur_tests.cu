#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>

//constexpr uint64_t dataSize = 10000;
//constexpr uint64_t pictureSize = 1920 * 1080;
//
//__global__
//void filter(int n, int* src, int* dst) {
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//	if (index < n) {
//		int sum = 0;
//		for (int i = -5; i < 6; i++) {
//			int iterator = index + i;
//			if (iterator < 0) {
//				continue;
//			}
//			if (iterator > n) {
//				continue;
//			}
//			sum = sum + src[iterator];
//		}
//		dst[index] = sum;
//	}
//}
//
//class FilterKernel : public Kernel {
//public:
//
//	FilterKernel() {
//		std::cout << "Initializing data..." << std::endl;
//
//		cudaMallocHost(&srcHost, dataSize * sizeof(int*));
//		cudaMallocHost(&dstHost, dataSize * sizeof(int*));
//
//		cudaMalloc(&src, dataSize * sizeof(int*));
//		cudaMalloc(&dst, dataSize * sizeof(int*));
//
//		for (uint64_t i = 0; i < dataSize; i++) {
//		}
//
//		cudaStreamCreate(&ownStream);
//
//		cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
//		cudaMemcpyAsync(dst, dstHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
//
//		std::cout << "Data initialized" << std::endl;
//	}
//
//	~FilterKernel() {
//		cudaFree(dst);
//		cudaFree(src);
//
//		cudaFreeHost(dstHost);
//		cudaFreeHost(srcHost);
//
//		cudaStreamDestroy(ownStream);
//	}
//
//	void runCpu(int workItemId, int workGroupSize) override {
//		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
//			int sum = 0;
//			for (int j = -5; j < 6; j++) {
//				int iterator = i + j;
//				if (iterator < 0) {
//					continue;
//				}
//				if (iterator > workGroupSize) {
//					continue;
//				}
//				sum = sum + srcHost[iterator];
//			}
//			dstHost[i] = sum;
//		}
//	};
//
//	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
//		int blockSize = 1024;
//		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
//		add << <numBlocks, blockSize, 0, ownStream >> > (workGroupSize, src + workItemId, dst + workItemId);
//		cudaMemcpyAsync(dstHost + workItemId, dst + workItemId, workGroupSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
//	};
//
//	int** src = nullptr;
//	int** dst = nullptr;
//	int** srcHost = nullptr;
//	int** dstHost = nullptr;
//
//	cudaStream_t ownStream;
//};