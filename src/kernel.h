#include <vector>

class Kernel {
public:
	Kernel() = default;
	virtual ~Kernel() = default;

	virtual void runCpu(int workItemId, int workGroupSize) = 0;
	virtual void runGpu(int deviceId, int workItemId, int workGroupSize) = 0;
};