#include <omp.h>

#define OMP_FOR(init,condition,increment)						\
		_Pragma("omp parallel default(shared)")					\
		_Pragma("omp for schedule(dynamic, 1)")					\
		for(init;condition;increment)		


//TODO:
//faktycznie zawolac podany kernel
//znalezc workaround na zinkrementowanie "i" o chunk_size
//sprawic aby kernel liczyl dla chunk_size konretnych elementow poczawszy od indexu i
#define HYBRID_EXECUTE(kernel, ...)										\
		id = omp_get_thread_num();										\
		if(id == 0) {													\
			if(stop == 0) {												\
				kernel<<<chunk_size / 1000, 1000>>>(__VA_ARGS__);		\
				cudaDeviceSynchronize(); 								\
				moveMem << < rozmiarPlecaka / 1000, 1000 >> > (out, rozmiarPlecaka); \
				cudaDeviceSynchronize(); 								\
				cudaMemcpy(rozwiazanie.data() + rozmiarPlecaka + j, out + j + rozmiarPlecaka, chunk_size * sizeof(int), cudaMemcpyDeviceToHost); \
				cudaDeviceSynchronize(); 								\
			}															\
			++stop;														\
			if (stop == chunk_size)										\
				stop = 0;												\
			continue;													\
		}
