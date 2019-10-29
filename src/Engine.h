#pragma once
constexpr auto gigaByte = 1073741824.0f;
#include <vector>


class Engine
{
public:
	Engine();
	int getDeviceCount() { return deviceCount; }
	std::vector <float> getMemories() { return memorySizes; }

private:
	int deviceCount;
	std::vector <float> memorySizes;
};

