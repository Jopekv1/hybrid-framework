#include "kernel.h"
#include "load_balancer.h"

#include <omp.h>
#include <iostream>

LoadBalancer::LoadBalancer() {
	this->getDeviceCount();
}

LoadBalancer::~LoadBalancer() = default;

void LoadBalancer::execute(Kernel* kernel, uint64_t wokrItemStartIndex, uint64_t workItemsCnt) {
	//TODO:
	//-perform tuning

	uint64_t id;
	bool nextIter;
	long long i = wokrItemStartIndex;
	uint64_t workCounter = wokrItemStartIndex;
	uint64_t workItemsCount = workItemsCnt;
	int numberOfGpus = this->gpuCount;
	uint64_t cpuWorkGroupSize = this->workGroupSize;
	uint64_t gpuWorkGroupSize = this->workGroupSize * this->gpuWorkGroups;

	#pragma omp parallel default(none) private(i, id, nextIter) shared(workCounter, workItemsCount, numberOfGpus, cpuWorkGroupSize, gpuWorkGroupSize, kernel) 
	{
		auto threadCount = omp_get_num_threads();
		if (threadCount == 1) {
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SINGLE THREAD RUNNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		}

		#pragma omp for schedule(dynamic, cpuWorkGroupSize)
		for (i = 0; i < workItemsCount; i++) {
			id = omp_get_thread_num();
			nextIter = false;

			#pragma omp critical
			{
				if (i < workCounter) {
					nextIter = true;
				}
				else {
					if (id < numberOfGpus) {
						workCounter += gpuWorkGroupSize;
						//printf("thread %d running on GPU work items: %d - %d\n",id,i, i + gpuWorkGroupSize - 1);

						if (workCounter >= workItemsCount) {
							gpuWorkGroupSize = workItemsCount - i;
						}
					}
					else {
						workCounter += cpuWorkGroupSize;
						//printf("thread %d running on CPU work items: %d - %d\n", id, i, i + cpuWorkGroupSize - 1);

						if (workCounter >= workItemsCount) {
							cpuWorkGroupSize = workItemsCount - i;
						}
					}
				}
			}
			if (nextIter) {
				continue;
			}

			if (id < numberOfGpus) {
				kernel->runGpu(id, i, gpuWorkGroupSize);
			}
			else {
				kernel->runCpu(i, cpuWorkGroupSize);
			}
		}
	}

	this->synchronize();
}
