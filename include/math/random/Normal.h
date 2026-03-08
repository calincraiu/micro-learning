#ifndef NORMAL_H
#define NORMAL_H

#include <vector>
#include <random>

class Normal {
public:
	/// <summary>
	/// Generate one standard normal sample using Box-Muller.
	/// </summary>
	float sampleStandardNormal();
	float sample(float mean, float std);
	std::vector<float> sample(float mean, float std, size_t numSamples);
private:
	std::mt19937 m_mtEngine{ std::random_device()() };
	std::uniform_real_distribution<float> m_uniform{ 0.0f, 1.0f }; // uniform generator
};


#endif // NORMAL_H