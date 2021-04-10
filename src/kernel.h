#include <vector>

class Kernel {
public:
	Kernel() = default;
	virtual ~Kernel() = default;

	virtual void runCpu(uint64_t workItemId, uint64_t workGroupSize) = 0;
	virtual void runGpu(uint64_t deviceId, uint64_t workItemId, uint64_t workGroupSize) = 0;
};