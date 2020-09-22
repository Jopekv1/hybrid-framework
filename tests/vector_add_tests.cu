#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

constexpr int dataSize = 20000000;

__global__
void add(int n, int* src, int* dst)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		dst[i] = src[i] + dst[i];
	}
}

class VecAddKernel : public Kernel {
public:

	VecAddKernel(int* src) : src(src) {
		cudaMallocManaged(&dst, dataSize * sizeof(int));
		memset(dst, 2, dataSize * sizeof(int));
	}

	~VecAddKernel() {
		cudaFree(dst);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		for (int i = workItemId; i < workItemId + workGroupSize; i++) {
			dst[i] += src[i];
		}
	};
	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		int blockSize = 256;
		int numBlocks = (workGroupSize + blockSize - 1) / blockSize;
		add<<<numBlocks, blockSize>>>(workGroupSize, src + workItemId, dst + workItemId);
		cudaDeviceSynchronize();
	};

	int* src = nullptr;
	int* dst = nullptr;
};


TEST(vectorAdd, vectorAdd) {
	int* src;
	cudaMallocManaged(&src, dataSize * sizeof(int));
	memset(src, 1, dataSize * sizeof(int));

	VecAddKernel kernel(src);

	LoadBalancer balancer;
	balancer.execute(&kernel, 0, dataSize);

	for (int i = 0; i < dataSize; i++) {
		if (50529027 == kernel.dst[i]) {
			int a=0;
			a++;
		}
		else {
			int b=0;
			b++;
		}
	}

	cudaFree(src);
}