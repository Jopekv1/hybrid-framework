#pragma once

#include "Algorithm.h"
#include <algorithm>
#include <cuda_runtime.h>

template<typename Type>
class MaxReduction : public Algorithm<Type>
{
public:
	using DataBlock = typename Algorithm<Type>::DataBlock;
	
	DataBlock runCPU(DataBlock data) override;
	DataBlock runGPU(DataBlock data) override;
	bool isBase(DataBlock data) override;
	std::vector<DataBlock> divide(DataBlock data) override { return {data}; };
	DataBlock merge(std::vector<DataBlock> data) override;

	virtual ~MaxReduction() {};
};



template<typename Type>
auto MaxReduction<Type>::runCPU(DataBlock data) -> DataBlock
{
	auto* ret = new float; // move to engine
	*ret = *std::max_element(data.first, data.first + data.second);
	return DataBlock(ret, 1);
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize, typename Type, typename = std::enable_if<std::is_same<Type, float>::value, int>>
__global__ void reduce(float* g_idata, float* g_odata, unsigned int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] = fmaxf(g_idata[i], g_idata[i + blockSize]); i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<typename Type>
auto MaxReduction<Type>::runGPU(DataBlock data) -> DataBlock
{
	const int threads = 512;
	unsigned int numPerThread = round(log2(data.second));
	while ((data.second / threads / 2) % numPerThread != 0)
		--numPerThread;
	const std::uint32_t outSize = data.second / numPerThread / threads / 2;
	const dim3 dimGrid = { outSize, 1, 1 };
	const dim3 dimBlock = { threads, 1, 1 };
	const int sharedMemSize = threads * 4;

	Type* outData;

	cudaMallocManaged(&outData, outSize * sizeof(Type)); //move this to engine

	switch (threads)
	{
	case 512:
		reduce<512, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 256:
		reduce<256, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 128:
		reduce<128, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 64:
		reduce< 64, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 32:
		reduce< 32, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 16:
		reduce< 16, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 8:
		reduce<  8, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 4:
		reduce<  4, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 2:
		reduce<  2, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	case 1:
		reduce<  1, Type> << <dimGrid, dimBlock, sharedMemSize >> > (data.first, outData, numPerThread * threads * 2 * outSize); break;
	}
	cudaDeviceSynchronize();

	return DataBlock(outData, outSize);
}

template <typename Type>
bool MaxReduction<Type>::isBase(DataBlock data)
{
	return data.second < 512;
}

template <typename Type>
auto MaxReduction<Type>::merge(std::vector<DataBlock> data) -> DataBlock
{
	Type* max = new Type; //move to engine
	*max = std::numeric_limits<Type>::min();
#pragma omp parallel for
	for (int i = 0; i < data.size(); i++)
	{
		Type b = *std::max_element(data[i].first, data[i].first + data[i].second);
		#pragma omp critical
		*max = std::max(*max, b);
	}
	return DataBlock(max, 1);
}

