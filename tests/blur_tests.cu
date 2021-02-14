//#include "kernel.h"
//#include "load_balancer.h"
//
//#include <gtest/gtest.h>
//#include <cuda_runtime.h>
//#include <chrono>
//#include <cstdlib>
//
//__global__
//void collatz(int n, int* src) {
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//	if (index < n) {
//		int counter = 0;
//		int value = index;
//		while (value > 1) {
//			if (value % 2 == 0) {
//				value = value / 2;
//			}
//			else {
//				value = 3 * value + 1;
//			}
//			counter++;
//		}
//		src[index] = counter;
//	}
//}
//
//__global__
//void gaussianBlur(unsigned char* inputFrame, unsigned char* outputFrame, int imageWidth, int imageHight, float* filter) {
//	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
//	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
//	if (pixelX >= imageWidth || pixelY >= imageHight) {
//		return;
//	}
//
//	float result = 0.0f;
//
//	for (int i = -4; i < 5; i++) {
//		int pixelXI = pixelX + i;
//
//		if (pixelXI < 0) {
//			pixelXI = 0;
//		}
//		else if (pixelXI >= imageWidth) {
//			pixelXI = imageWidth - 1;
//		}
//
//		for (int j = -4; j < 5; j++) {
//			int pixelYJ = pixelY + j;
//
//			if (pixelYJ < 0) {
//				pixelYJ = 0;
//			}
//			else if (pixelYJ >= imageHight) {
//				pixelYJ = imageHight - 1;
//			}
//
//			result += (filter[(j + 4) * 9 + (i + 4)] * inputFrame[pixelYJ * imageWidth + pixelXI]);
//		}
//	}
//
//	outputFrame[pixelY * imageWidth + pixelX] = result;
//}
//
//class BlurKernel : public Kernel {
//public:
//	BlurKernel() = default;
//	~BlurKernel() = default;
//
//	void runCpu(int workItemId, int workGroupSize) override {
//		for (int k = workItemId; k < workItemId + workGroupSize; k++) {
//			auto inputFrame = inputFrames[k];
//			float result = 0.0f;
//
//			for (int i = 0; i < imageWidth; i++) {
//				for (int j = 0; j < imageHight; j++) {
//					for (int l = -4; i < 5; i++) {
//						int pixelXI = i + l;
//
//						if (pixelXI < 0) {
//							pixelXI = 0;
//						}
//						else if (pixelXI >= imageWidth) {
//							pixelXI = imageWidth - 1;
//						}
//
//						for (int m = -4; m < 5; m++) {
//							int pixelYJ = j + m;
//
//							if (pixelYJ < 0) {
//								pixelYJ = 0;
//							}
//							else if (pixelYJ >= imageHight) {
//								pixelYJ = imageHight - 1;
//							}
//
//							result += (filter[(m + 4) * 9 + (l + 4)] * inputFrame[pixelYJ * imageWidth + pixelXI]);
//						}
//					}
//					outputFramesHybrid[k][j * imageWidth + i] = static_cast<char>(result);
//				}
//			}
//		}
//	};
//	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
//		const dim3 blockSize(16, 16, 1);
//		const dim3 gridSize(imageWidth / blockSize.x + 1, imageHight / blockSize.y + 1, 1);
//
//		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
//			gaussianBlur<<<gridSize, blockSize>>>(inputFrames[i], outputFramesHybrid[i], imageWidth, imageHight, filter);
//		}
//	};
//};
//
////TEST(blur, hybrid) {
////	initializeData();
////
////	BlurKernel kernel;
////	LoadBalancer balancer(1,100);
////
////	auto start = std::chrono::steady_clock::now();
////	balancer.execute(&kernel, frameCount);
////	auto end = std::chrono::steady_clock::now();
////
////	std::chrono::duration<double> elapsed_seconds = end - start;
////	std::cout << "Hybrid time: " << elapsed_seconds.count() << "s\n";
////}
////
////TEST(blur, gpu) {
////	const dim3 blockSize(16, 16, 1);
////	const dim3 gridSize(imageWidth / blockSize.x + 1, imageHight / blockSize.y + 1, 1);
////
////	auto start = std::chrono::steady_clock::now();
////	for (int i = 0; i < frameCount; i++) {
////		gaussianBlur<<<gridSize, blockSize>>>(inputFrames[i], outputFramesGpu[i], imageWidth, imageHight, filter);
////	}
////	cudaDeviceSynchronize();
////	auto end = std::chrono::steady_clock::now();
////
////	std::chrono::duration<double> elapsed_seconds = end - start;
////	std::cout << "GPU time: " << elapsed_seconds.count() << "s\n";
////
////
////	//verifyData();
////	freeData();
////}