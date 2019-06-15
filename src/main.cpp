#include "hybrid_executor.h"

#include <cstdio>
#include <chrono>
#include <ctime>
#include <memory>

constexpr int problemPlecakowySize = 100000;
constexpr int rozmiarPlecaka = 50000;

template<class Callable>
void timeWrapper(Callable f) {
	auto start = std::chrono::steady_clock::now();
	f();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("%f\n", elapsed_seconds.count());
}

void problemPlecakowyCPU() {
	srand(time(NULL));
	auto waga = new int[problemPlecakowySize];
	auto warotsc = new int[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		waga[i] = rand() % 200;
		warotsc[i] = rand() % 200;
	}

	auto rozwiazanie = new int*[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i] = new int [rozmiarPlecaka];
	}
	for (int j = 0; j < rozmiarPlecaka; j++) {
		rozwiazanie[0][j] = 0;
	}

	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i][0] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		for (int j = 1; j < rozmiarPlecaka; j++) {
			if (j > waga[i] && rozwiazanie[i - 1][j] < rozwiazanie[i][j - waga[i]] + warotsc[i]) {
				rozwiazanie[i][j] = rozwiazanie[i][j - waga[i]] + warotsc[i];
			}
			else {
				rozwiazanie[i][j] = rozwiazanie[i - 1][j];
			}
		}
	}
	delete[] waga;
	delete[] warotsc;
	for (int i = 0; i < problemPlecakowySize; i++) {
		delete[] rozwiazanie[i];
	}
	delete[] rozwiazanie;
}

void problemPlecakowyOMP() {
	srand(time(NULL));
	auto waga = new int[problemPlecakowySize];
	auto warotsc = new int[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		waga[i] = rand() % 200;
		warotsc[i] = rand() % 200;
	}

	auto rozwiazanie = new int*[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i] = new int[rozmiarPlecaka];
	}
	for (int j = 0; j < rozmiarPlecaka; j++) {
		rozwiazanie[0][j] = 0;
	}

	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i][0] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		HYBRID_FOR(j = 1, j < rozmiarPlecaka, j++) {
			if (j > waga[i] && rozwiazanie[i - 1][j] < rozwiazanie[i][j - waga[i]] + warotsc[i]) {
				rozwiazanie[i][j] = rozwiazanie[i][j - waga[i]] + warotsc[i];
			}
			else {
				rozwiazanie[i][j] = rozwiazanie[i - 1][j];
			}
		}
	}
	delete[] waga;
	delete[] warotsc;
	for (int i = 0; i < problemPlecakowySize; i++) {
		delete[] rozwiazanie[i];
	}
	delete[] rozwiazanie;
}

void mockKernel(void* exampleParam1, void* exampleParam2, void* exampleParam3, void* exampleParam4) {
	//napisac faktyczny kernel
}

void problemPlecakowyGPU() {
	srand(time(NULL));
	auto waga = new int[problemPlecakowySize];
	auto warotsc = new int[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		waga[i] = rand() % 200;
		warotsc[i] = rand() % 200;
	}

	auto rozwiazanie = new int*[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i] = new int[rozmiarPlecaka];
	}
	for (int j = 0; j < rozmiarPlecaka; j++) {
		rozwiazanie[0][j] = 0;
	}

	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i][0] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		//mockKernel(params...) kernel ktory realizuje logike hybrid for'a
	}
	delete[] waga;
	delete[] warotsc;
	for (int i = 0; i < problemPlecakowySize; i++) {
		delete[] rozwiazanie[i];
	}
	delete[] rozwiazanie;
}

void mockKernelHybrid(int i, int size, void* exampleParam1, void* exampleParam2, void* exampleParam3, void* exampleParam4) {
	//napisac faktyczny kernel
}

void problemPlecakowyHybrid() {
	srand(time(NULL));
	auto waga = new int[problemPlecakowySize];
	auto warotsc = new int[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		waga[i] = rand() % 200;
		warotsc[i] = rand() % 200;
	}

	auto rozwiazanie = new int*[problemPlecakowySize];
	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i] = new int[rozmiarPlecaka];
	}
	for (int j = 0; j < rozmiarPlecaka; j++) {
		rozwiazanie[0][j] = 0;
	}

	for (int i = 0; i < problemPlecakowySize; i++) {
		rozwiazanie[i][0] = 0;
	}

	for (int i = 1; i < problemPlecakowySize; i++) {
		HYBRID_FOR(j = 1, j < rozmiarPlecaka, j++) {
			//HYBRID_EXECUTE(mockKernelHybrid, nullptr, nullptr, nullptr, nullptr);
			//kernel ktory realizuje logike hybrid for'a
			//!!!!!!!!!!!!!!!!jako pierwszy parametr bierze indeks ok ktorego liczy, a jako drugi parametr, ile elemntow ma obliczyc!!!!!!!!!!!!!!!
			if (j > waga[i] && rozwiazanie[i - 1][j] < rozwiazanie[i][j - waga[i]] + warotsc[i]) {
				rozwiazanie[i][j] = rozwiazanie[i][j - waga[i]] + warotsc[i];
			}
			else {
				rozwiazanie[i][j] = rozwiazanie[i - 1][j];
			}
		}
	}
	delete[] waga;
	delete[] warotsc;
	for (int i = 0; i < problemPlecakowySize; i++) {
		delete[] rozwiazanie[i];
	}
	delete[] rozwiazanie;
}

void problemPlecakowy() {
	printf("PROBLEM PLECAKOWY\n");

	printf("Wersja proceduralna: ");
	timeWrapper(problemPlecakowyCPU);

	printf("Wersja wielowatkowa(OMP): ");
	timeWrapper(problemPlecakowyOMP);

	printf("Wersja GPU: ");
	timeWrapper(problemPlecakowyGPU);

	printf("Wersja hybrydowa: ");
	timeWrapper(problemPlecakowyHybrid);
}


int main() {
	problemPlecakowy();
	return 0;
}