#include <stdint.h>

class Kernel;

class LoadBalancer {
public:
	LoadBalancer();
	virtual ~LoadBalancer();

	void execute(Kernel* kernel, uint64_t wokrItemStartIndex, uint64_t workItemsCnt);

private:

	void getDeviceCount();
	void synchronize();

	int gpuCount = 1;
	uint64_t workGroupSize = 10000;
	uint64_t gpuWorkGroups = 1024*1000;

};