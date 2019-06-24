#include <omp.h>

//TODO:
//wyliczac chunk_size na podstawie ilosci threadow/pamieci gpu
#define HYBRID_FOR(init,condition,increment)														\
		int j, id, chunk_size = 10000;																\
		__pragma(omp parallel default(shared) private(j,id))										\
		__pragma(omp for schedule(dynamic, chunk_size))												\
		for(init;condition;increment)		


//TODO:
//faktycznie zawolac podany kernel
//znalezc workaround na zinkrementowanie "i" o chunk_size
//sprawic aby kernel liczyl dla chunk_size konretnych elementow poczawszy od indexu i
#define HYBRID_EXECUTE(kernel, ...)																	\
		id = omp_get_thread_num();																	\
		if(id == 0) {																				\
			if(j % chunk_size == 0) {																\
				kernel<<<chunk_size / 1000, 1000>>>(__VA_ARGS__);									\
				cudaDeviceSynchronize();															\
			}																						\
			continue;																				\
		}																							