//#include "kernel.h"
//#include "load_balancer.h"
//
//#include <gtest/gtest.h>
//#include <cuda_runtime.h>
//#include <chrono>
//#include <cstdlib>
//
//constexpr uint64_t dataSize = 100000000;
//constexpr uint64_t frameSize = 10000;
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
//		cudaMalloc(&dst1, dataSize * sizeof(int*));
//		cudaMalloc(&dst2, dataSize * sizeof(int*));
//
//		srand(time(0));
//		for (uint64_t i = 0; i < dataSize; i++) {
//			srcHost[i] = rand();
//		}
//
//		cudaStreamCreate(&ownStream);
//
//		std::cout << "Data initialized" << std::endl;
//	}
//
//	~FilterKernel() {
//		cudaFree(dst1);
//		cudaFree(dst2);
//		cudaFree(src);
//
//		cudaFreeHost(dstHost);
//		cudaFreeHost(srcHost);
//
//		cudaStreamDestroy(ownStream);
//	}
//
//	void runCpu(int workItemId, int workGroupSize) override {
//		for (int k = 0; k < 5; k++) {
//			for (int i = (workItemId * frameSize); i < ((workItemId + workGroupSize) * frameSize); i++) {
//				int sum = 0;
//				for (int j = -5; j < 6; j++) {
//					int iterator = i + j;
//					if (iterator < (workItemId * frameSize)) {
//						continue;
//					}
//					if (iterator >= ((workItemId + workGroupSize) * frameSize)) {
//						continue;
//					}
//					sum = sum + srcHost[iterator];
//				}
//				dstHost[i] = sum;
//			}
//		}
//	};
//
//	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
//		int blockSize = 1024;
//		int numBlocks = ((workGroupSize * frameSize) + blockSize - 1) / blockSize;
//		cudaMemcpyAsync(src + (workItemId * frameSize), srcHost + (workItemId * frameSize), (workGroupSize * frameSize) * sizeof(int), cudaMemcpyHostToDevice, ownStream);
//		filter << <numBlocks, blockSize, 0, ownStream >> > ((workGroupSize * frameSize), src + (workItemId * frameSize), dst1 + (workItemId * frameSize));
//		filter << <numBlocks, blockSize, 0, ownStream >> > ((workGroupSize * frameSize), dst1 + (workItemId * frameSize), dst2 + (workItemId * frameSize));
//		filter << <numBlocks, blockSize, 0, ownStream >> > ((workGroupSize * frameSize), dst2 + (workItemId * frameSize), dst1 + (workItemId * frameSize));
//		filter << <numBlocks, blockSize, 0, ownStream >> > ((workGroupSize * frameSize), dst1 + (workItemId * frameSize), dst2 + (workItemId * frameSize));
//		filter << <numBlocks, blockSize, 0, ownStream >> > ((workGroupSize * frameSize), dst2 + (workItemId * frameSize), dst1 + (workItemId * frameSize));
//		cudaMemcpyAsync(dstHost + (workItemId * frameSize), dst1 + (workItemId * frameSize), (workGroupSize * frameSize) * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
//	};
//
//	int* src = nullptr;
//	int* dst1 = nullptr;
//	int* dst2 = nullptr;
//	int* srcHost = nullptr;
//	int* dstHost = nullptr;
//
//	cudaStream_t ownStream;
//};
//
//class FilterFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>> {
//public:
//
//	void SetUp() override {
//		std::tie(workGroupSize, gpuWorkGroups, numThreads) = GetParam();
//
//		std::cout << "Test params: workGroupSize: " << workGroupSize << ", gpuWorkGroups: " << gpuWorkGroups << ", numThread: " << numThreads << std::endl;
//
//		if (gpuWorkGroups * workGroupSize >= dataSize) {
//			std::cout << "!!!!!!!!!!!!!!!!! GPU COVERS WHOLE DATA !!!!!!!!!!!!!!!!!!!!!!" << std::endl;
//			//GTEST_SKIP();
//		}
//	}
//
//	uint64_t workGroupSize = 0;
//	uint64_t gpuWorkGroups = 0;
//	int numThreads = 0;
//};
//
//TEST_P(FilterFixture, hybrid) {
//	FilterKernel kernel;
//
//	LoadBalancer balancer(workGroupSize, gpuWorkGroups, numThreads);
//
//	auto start = std::chrono::steady_clock::now();
//
//	balancer.execute(&kernel, dataSize);
//
//	auto end = std::chrono::steady_clock::now();
//
//	std::chrono::duration<double> elapsed_seconds = end - start;
//	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";
//}
//
//static uint64_t workGroupSizesValues[] = {
//	1,
//	2,
//	5,
//	10 };
//
//static uint64_t gpuWorkGroupsValues[] = {
//	10,
//	100,
//	1000,
//	2000,
//	5000,
//	10000 };
//
//static int numThreadsValues[] = {
//	2,
//	4,
//	6,
//	8 };
//
//INSTANTIATE_TEST_SUITE_P(Filter,
//	FilterFixture,
//	::testing::Combine(
//		::testing::ValuesIn(workGroupSizesValues),
//		::testing::ValuesIn(gpuWorkGroupsValues),
//		::testing::ValuesIn(numThreadsValues)));
//
//TEST(Filter, gpu) {
//	std::cout << "Initializing data..." << std::endl;
//
//	int* src = nullptr;
//	int* dst1 = nullptr;
//	int* dst2 = nullptr;
//	int* srcHost = nullptr;
//	int* dstHost = nullptr;
//
//	cudaStream_t ownStream;
//
//	cudaMallocHost(&srcHost, dataSize * sizeof(int));
//	cudaMallocHost(&dstHost, dataSize * sizeof(int));
//
//	cudaMalloc(&src, dataSize * sizeof(int));
//	cudaMalloc(&dst1, dataSize * sizeof(int));
//	cudaMalloc(&dst2, dataSize * sizeof(int));
//
//	srand(time(0));
//	for (uint64_t i = 0; i < dataSize; i++) {
//		srcHost[i] = rand();
//	}
//
//	cudaStreamCreate(&ownStream);
//
//	std::cout << "Data initialized" << std::endl;
//
//	int blockSize = 1024;
//	int numBlocks = (dataSize + blockSize - 1) / blockSize;
//
//	auto start = std::chrono::steady_clock::now();
//
//	cudaMemcpyAsync(src, srcHost, dataSize * sizeof(int), cudaMemcpyHostToDevice, ownStream);
//	filter << <numBlocks, blockSize, 0, ownStream >> > (dataSize, src, dst1);
//	filter << <numBlocks, blockSize, 0, ownStream >> > (dataSize, dst1, dst2);
//	filter << <numBlocks, blockSize, 0, ownStream >> > (dataSize, dst2, dst1);
//	filter << <numBlocks, blockSize, 0, ownStream >> > (dataSize, dst1, dst2);
//	filter << <numBlocks, blockSize, 0, ownStream >> > (dataSize, dst2, dst1);
//	cudaMemcpyAsync(dstHost, dst1, dataSize * sizeof(int), cudaMemcpyDeviceToHost, ownStream);
//	cudaDeviceSynchronize();
//
//	auto end = std::chrono::steady_clock::now();
//
//	std::chrono::duration<double> elapsed_seconds = end - start;
//	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";
//
//
//	cudaFree(dst1);
//	cudaFree(dst2);
//	cudaFree(src);
//
//	cudaFreeHost(dstHost);
//	cudaFreeHost(srcHost);
//
//	cudaStreamDestroy(ownStream);
//}