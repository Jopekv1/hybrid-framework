#include "dynamic_executor.h"

#include <cstdio>

#define  NUM_LOOPS 50																
		
void mockKernel(int i, int size, void* ptr) {
	printf("I'm gpu xd\n");
}

int main() {

	int n = NUM_LOOPS;
	HYBRID_FOR(i=0, i < n, i++) {
		HYBRID_EXECUTE(mockKernel, nullptr);
		printf("Iteracja %d wykonana przez watek nr. %d.\n", i, id);
	}
	
	return 0;
}