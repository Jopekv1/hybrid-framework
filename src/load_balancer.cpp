#include "kernel.h"
#include "load_balancer.h"

#include <iostream>
#include <thread>
#include <utility>
#include <mutex>

LoadBalancer::LoadBalancer(uint64_t workGroupSize, uint64_t gpuWorkGroups, int numThreads) {
	this->getDeviceCount();
	this->workGroupSize = workGroupSize;
	this->gpuWorkGroups = gpuWorkGroups;
	this->numThreads = numThreads;
}

LoadBalancer::~LoadBalancer() = default;

struct threadData {
	int gpuCount;

	int threadId;
	Kernel* kernel;
	std::mutex* balancerMtx;

	uint64_t workItemsCnt;
	uint64_t cpuWorkGroupSize;
	uint64_t gpuWorkGroupSize;

	uint64_t* workCounter;
};

void threadExecute(threadData & data) {
	for (uint64_t i = 0; i < data.workItemsCnt; i++) {
		data.balancerMtx->lock();
		if (i < *data.workCounter) {
			i = *data.workCounter;
		}
		if (data.threadId == 0) {
			*data.workCounter += data.gpuWorkGroupSize;
		}
		else {
			*data.workCounter += data.cpuWorkGroupSize;
		}

		data.balancerMtx->unlock();

		if (i >= data.workItemsCnt) {
			break;
		}

		if (data.threadId == 0) {
			if (i + data.gpuWorkGroupSize > data.workItemsCnt) {
				data.gpuWorkGroupSize = data.workItemsCnt - i;
			}
			//printf("Thread %d run on GPU, items: %d - %d\n", data.threadId, i, i + data.gpuWorkGroupSize - 1);
			data.kernel->runGpu(0, i, data.gpuWorkGroupSize);
		}
		else {
			if (i + data.cpuWorkGroupSize > data.workItemsCnt) {
				data.cpuWorkGroupSize = data.workItemsCnt - i;
			}
			//printf("Thread %d run on CPU, items: %d - %d\n", data.threadId, i, i + data.cpuWorkGroupSize - 1);
			data.kernel->runCpu(i, data.cpuWorkGroupSize);
		}
	}
}

void LoadBalancer::execute(Kernel * kernel, uint64_t workItemsCnt) {
	//TODO:
	//-perform tuning

	constexpr int maxThreadCount = 8;
	const int threadCount = this->numThreads;

	std::thread threads[maxThreadCount - 1];
	threadData datas[maxThreadCount];

	uint64_t workCounter = 0u;
	std::mutex balancerMtx;

	for (int i = 0; i < threadCount; i++) {
		datas[i].gpuCount = this->gpuCount;
		datas[i].threadId = i;
		datas[i].kernel = kernel;
		datas[i].balancerMtx = &balancerMtx;
		datas[i].workItemsCnt = workItemsCnt;
		datas[i].cpuWorkGroupSize = this->workGroupSize;
		datas[i].gpuWorkGroupSize = this->workGroupSize * this->gpuWorkGroups;
		datas[i].workCounter = &workCounter;

		if (i < threadCount - 1) {
			threads[i] = std::thread(threadExecute, std::ref(datas[i]));
		}
	}

	threadExecute(std::ref(datas[threadCount - 1]));

	for (int i = 0; i < threadCount - 1; i++) {
		threads[i].join();
	}

	this->synchronize();
}
