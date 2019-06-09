#include <omp.h>

//TODO:
//wyliczac chunk_size na podstawie ilosci threadow/pamieci gpu
#define HYBRID_FOR(init,condition,increment)														\
		int i, id, chunk_size = 5;																	\
		__pragma(omp parallel default(shared) private(i,id) schedule(dynamic, chunk_size))			\
		for(init;condition;increment)		


//TODO:
//faktycznie zawolac podany kernel
//znalezc workaround na zinkrementowanie "i" o chunk_size
//sprawic aby kernel liczyl dla chunk_size konretnych elementow poczawszy od indexu i
#define HYBRID_EXECUTE(kernel, ...)																	\
		id = omp_get_thread_num();																	\
		if(id == 0) {																				\
			kernel(i, chunk_size, __VA_ARGS__);														\
			continue;																				\
		}																							