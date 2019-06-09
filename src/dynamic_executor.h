#include <omp.h>

#define HYBRID_FOR(init,condition,increment)														\
		int i, id, chunk_size = 5;																	\
		__pragma(omp parallel default(shared) private(i,id) schedule(dynamic, chunk_size))			\
		for(init;condition;increment)		 														

#define HYBRID_EXECUTE(kernel, ...)																	\
		id = omp_get_thread_num();																	\
		if(id == 0) {																				\
			kernel(i, chunk_size, __VA_ARGS__);														\
			continue;																				\
		}																							