class Kernel;

class LoadBalancer {
public:
	LoadBalancer();
	virtual ~LoadBalancer();

	void execute(Kernel* kernel, int wokrItemStartIndex, int workItemsCnt);

private:

	void getDeviceCount();
	void synchronize();

	int gpuCount = 1;
	int workGroupSize = 10000;
	int gpuWorkGroups = 100;

};