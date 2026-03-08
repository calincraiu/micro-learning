#include <cmath>

#include "math/random/Normal.h"
#include "math/Constants.h"


float Normal::sampleStandardNormal() {
	float u1, u2;
	do {
		u1 = m_uniform(m_mtEngine);  // Uniform [0,1)
		u2 = m_uniform(m_mtEngine);  // Uniform [0,1)
	} while (u1 == 0.0);  // Avoid log(0)

	float r = sqrt(-2.0 * log(u1));
	float theta = 2.0 * M_PI<float> * u2;
	float z0 = r * cos(theta);

	return z0;
}

float Normal::sample(float mean, float std) {
	return sampleStandardNormal() * std + mean;
}

std::vector<float> Normal::sample(float mean, float std, size_t numSamples) {
	std::vector<float> samples;
	samples.reserve(numSamples);
	for (size_t i = 0; i < numSamples; i++) {
		samples.push_back(sample(mean, std));
	}
	return samples;
}