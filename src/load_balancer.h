#include <stdint.h>

class Kernel;

class LoadBalancer {
public:
	LoadBalancer() = delete;
	LoadBalancer(uint64_t workGroupSize,uint64_t gpuWorkGroups);
	virtual ~LoadBalancer();

	void execute(Kernel* kernel, uint64_t workItemsCnt);

private:

	void getDeviceCount();
	void synchronize();

	int gpuCount = 1;
	uint64_t workGroupSize = 1000;
	uint64_t gpuWorkGroups = 100000;

};