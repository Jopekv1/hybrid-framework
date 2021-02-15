#include <gtest/gtest.h>
#include <stdio.h>

int main(int argc, char** argv) {
	auto gpuFile = fopen("results_gpu.txt", "w");
	auto hybridFile = fopen("results_hybrid.txt", "w");

	if (gpuFile == NULL || hybridFile == NULL) {
		return 0;
	}

	fprintf(gpuFile, "Testcase Time\n");
	fprintf(hybridFile, "Testcase WorkGroupSize GpuWorkGroups NumThreads Time\n");

	fclose(gpuFile);
	fclose(hybridFile);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}