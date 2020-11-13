#include "kernel.h"
#include "load_balancer.h"

#include <gtest/gtest.h>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

constexpr int dataSize = 100000000;

void gpuSort(int* p, int size) {
	//thrust::device_vector<int> deviceData();
	thrust::sort(p, p + size);
}

class QSortKernel : public Kernel {
public:

	QSortKernel() {
		cudaMallocManaged(&data, dataSize * sizeof(int));
	}

	~QSortKernel() {
		cudaFree(data);
	}

	void runCpu(int workItemId, int workGroupSize) override {
		std::sort(data + workItemId, data + workItemId + workGroupSize);
	};
	void runGpu(int deviceId, int workItemId, int workGroupSize) override {
		gpuSort(data + workItemId, workGroupSize);
	};

	int* data;
};
