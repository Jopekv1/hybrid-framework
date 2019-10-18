#include "PassManager.h"
#include "LoadBalancer.h"

template <typename Type>
typename PassManager<Type>::DataBlock PassManager<Type>::run(DataBlock data)
{
	LoadBalancer<Type> lb(algorithm);
	auto dividedData = lb.calculate();

	std::vector<DataBlock> outs;
	#pragma omp parallel for
	for (int i = 0; i < dividedData.size(); i++)
	{
		DataBlock out;
		//if (dividedData[i].second.type == cpu)
		//	out = engine.runAlgorithm(dividedData[i].first);
		//else
		//	out = engine.runAlgorithm(dividedData[i].first);
		#pragma omp critical
		outs.push_back(out);
	}
	//engine.free(data);
	DataBlock ret = algorithm.merge(outs);
	return ret;
}
