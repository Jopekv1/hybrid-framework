#include "hybrid_executor.h"

#include <cstdio>
#include <chrono>
#include <ctime>
#include <memory>
#include <vector>
#include <cassert>

constexpr int problemPlecakowySize = 100000;
constexpr int rozmiarPlecaka = 50000;

int test[rozmiarPlecaka];

template<class Callable, class... Args>
void timeWrapper(Callable f, Args... args) {
	auto start = std::chrono::steady_clock::now();
	f(args...);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f\n", elapsed_seconds.count());
}

void problemPlecakowyCPU(int* waga, int* wartosc) {
	auto rozwiazanie = new int*[2];
	for (int i = 0; i < 2; i++) {
		rozwiazanie[i] = new int [rozmiarPlecaka];
	}
	for (int j = 0; j < rozmiarPlecaka; j++) {
		rozwiazanie[0][j] = 0;
	}

	for (int i = 0; i < 2; i++) {
		rozwiazanie[i][0] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		for (int j = 1; j < rozmiarPlecaka; j++) {
			if (j > waga[i] && rozwiazanie[0][j] < rozwiazanie[0][j - waga[i]] + wartosc[i]) {
				rozwiazanie[1][j] = rozwiazanie[0][j - waga[i]] + wartosc[i];
			}
			else {
				rozwiazanie[1][j] = rozwiazanie[0][j];
			}
		}
		for (int j = 1; j < rozmiarPlecaka; j++)
		{
			rozwiazanie[0][j] = rozwiazanie[1][j];
		}
	}
	for (int i = 0; i < 2; i++) {
		if (i == 1)
		{
			for (int j = 0; j < rozmiarPlecaka; j++)
				test[j] = rozwiazanie[i][j];
		}
		delete[] rozwiazanie[i];
	}
	delete[] rozwiazanie;
}

void problemPlecakowyOMP(int* waga, int* wartosc) {
	auto rozwiazanie = new int*[2];
	for (int i = 0; i < 2; i++) {
		rozwiazanie[i] = new int[rozmiarPlecaka];
	}
	for (int j = 0; j < rozmiarPlecaka; j++) {
		rozwiazanie[0][j] = 0;
	}

	for (int i = 0; i < 2; i++) {
		rozwiazanie[i][0] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		HYBRID_FOR(j = 1, j < rozmiarPlecaka, j++) {
			if (j > waga[i] && rozwiazanie[0][j] < rozwiazanie[0][j - waga[i]] + wartosc[i]) {
				rozwiazanie[1][j] = rozwiazanie[0][j - waga[i]] + wartosc[i];
			}
			else {
				rozwiazanie[1][j] = rozwiazanie[0][j];
			}
		}
		for (int j = 1; j < rozmiarPlecaka; j++)
		{
			rozwiazanie[0][j] = rozwiazanie[1][j];
		}
	}
	for (int i = 0; i < 2; i++) {
		if (i == 1)
		{
			for (int j = 0; j < rozmiarPlecaka; j++)
			{
				if (test[j] != rozwiazanie[1][j])
					printf("%d ---------- %d\n", test[j], rozwiazanie[1][j]);
			}
		}
		delete[] rozwiazanie[i];
	}
	delete[] rozwiazanie;
}

__global__ void mockKernel(int* waga, int* wartosc, int* rozwiazanie, int i, int size) {
	int j = threadIdx.x + blockIdx.x * 1000;
	if (j == 0)
		return;
	long long rozwIdx = size + j;
	if (j > waga[i] && rozwiazanie[j] < rozwiazanie[j - waga[i]] + wartosc[i]) {
		rozwiazanie[rozwIdx] = rozwiazanie[j - waga[i]] + wartosc[i];
	}
	else {
		rozwiazanie[rozwIdx] = rozwiazanie[j];
	}
}

__global__ void moveMem(int* mem, int size)
{
	int j = threadIdx.x + blockIdx.x * 1000;
	mem[j] = mem[j + size];
}

void problemPlecakowyGPU(int* waga, int* wartosc) {
	std::vector<int> rozwiazanie(2 * rozmiarPlecaka, 0);

	int* weights, * values, * out;
	cudaMalloc(&weights, problemPlecakowySize * sizeof(int));
	cudaMalloc(&values, problemPlecakowySize * sizeof(int));
	cudaMalloc(&out, static_cast<long long>(2 * rozmiarPlecaka * sizeof(int)));
	cudaMemcpy(weights, waga, problemPlecakowySize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(values, wartosc, problemPlecakowySize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(out, rozwiazanie.data(), 2 * rozmiarPlecaka * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 1; i < problemPlecakowySize; i++) {
		mockKernel << < rozmiarPlecaka/1000, 1000 >> > (weights, values, out, i, rozmiarPlecaka);
		cudaDeviceSynchronize();
		moveMem << < rozmiarPlecaka / 1000, 1000 >> > (out, rozmiarPlecaka);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(rozwiazanie.data(), out, 2 * rozmiarPlecaka * sizeof(int), cudaMemcpyDeviceToHost);
	for (int j = 0; j < rozmiarPlecaka; j++)
	{
		if(test[j] != rozwiazanie[rozmiarPlecaka + j])
			printf("%d ---------- %d\n", test[j], rozwiazanie[rozmiarPlecaka + j]);
	}
	cudaFree(weights);
	cudaFree(values);
	cudaFree(out);
}

__global__ void mockKernelHybrid(int* waga, int* wartosc, int* rozwiazanie, int i, int size, int start_j) {
	int j = threadIdx.x + blockIdx.x * 1000 + start_j;
	if (j == 0)
		return;
	long long rozwIdx = size + j;
	if (j > waga[i] && rozwiazanie[j] < rozwiazanie[j - waga[i]] + wartosc[i]) {
		rozwiazanie[rozwIdx] = rozwiazanie[j - waga[i]] + wartosc[i];
	}
	else {
		rozwiazanie[rozwIdx] = rozwiazanie[j];
	}
}

void problemPlecakowyHybrid(int* waga_ref, int* wartosc_ref) {
	int* waga, *wartosc, *rozwiazanie;
	cudaMallocManaged(&waga, problemPlecakowySize * sizeof(int));
	cudaMallocManaged(&wartosc, problemPlecakowySize * sizeof(int));
	cudaMallocManaged(&rozwiazanie, rozmiarPlecaka * 2 * sizeof(int));
	for (int i = 0; i < problemPlecakowySize; i++) {
		waga[i] = waga_ref[i];
		wartosc[i] = wartosc_ref[i];
	}

	for (int i = 0; i < rozmiarPlecaka * 2; i++) {
		rozwiazanie[i] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		HYBRID_FOR(j = 1, j < rozmiarPlecaka, j++) {
			HYBRID_EXECUTE(mockKernelHybrid, waga, wartosc, rozwiazanie, i, rozmiarPlecaka, j);
			//kernel ktory realizuje logike hybrid for'a
			//!!!!!!!!!!!!!!!!jako pierwszy parametr bierze indeks ok ktorego liczy, a jako drugi parametr, ile elemntow ma obliczyc!!!!!!!!!!!!!!!
			if (j > waga[i] && rozwiazanie[j] < rozwiazanie[j - waga[i]] + wartosc[i]) {
				rozwiazanie[rozmiarPlecaka + j] = rozwiazanie[j - waga[i]] + wartosc[i];
			}
			else {
				rozwiazanie[rozmiarPlecaka + j] = rozwiazanie[j];
			}
		}
		for (int j = 1; j < rozmiarPlecaka; j++)
		{
			rozwiazanie[j] = rozwiazanie[j + rozmiarPlecaka];
		}
		printf("%d\n", i);
	}
	cudaFree(waga);
	cudaFree(wartosc);
	cudaFree(rozwiazanie);
}

void problemPlecakowy() {
	srand(time(NULL));
	auto waga = new int[problemPlecakowySize];
	auto wartosc = new int[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		waga[i] = rand() % 200;
		wartosc[i] = rand() % 200;
	}
	printf("PROBLEM PLECAKOWY\n");

	printf("Wersja proceduralna: ");
	timeWrapper(problemPlecakowyCPU, waga, wartosc);

	printf("Wersja wielowatkowa(OMP): ");
	//timeWrapper(problemPlecakowyOMP, waga, wartosc);

	printf("Wersja GPU: ");
	//timeWrapper(problemPlecakowyGPU, waga, wartosc);

	printf("Wersja hybrydowa: ");
	timeWrapper(problemPlecakowyHybrid, waga, wartosc);
}


int main() {
	problemPlecakowy();
	return 0;
}