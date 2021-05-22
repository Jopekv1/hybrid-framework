#include <gtest/gtest.h>
#include <stdio.h>
#include <cstring>

namespace Config {
	bool tunningMode = false;
}

int main(int argc, char** argv) {
	auto gpuFile = fopen("results_gpu.txt", "w");
	auto hybridFile = fopen("results_hybrid.txt", "w");

	if (gpuFile == NULL || hybridFile == NULL) {
		return 0;
	}

	fprintf(gpuFile, "Testcase DataSize Time\n");
	fprintf(hybridFile, "Testcase DataSize WorkGroupSize GpuWorkGroups NumThreads Time\n");

	fclose(gpuFile);
	fclose(hybridFile);

	if (argc > 1) {
		auto param = argv[1];
		if (strcmp(param, "-t") == 0) {
			Config::tunningMode = true;
		}
	}

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}