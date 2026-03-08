#include <assert.h>
#include <sstream>
#include <random>

#include "nn/layers/Linear.h"

Linear::Linear(Tape* tape, size_t features_in, size_t features_out, bool use_bias) : 
	Layer(), m_tape(tape), m_features_in(features_in), m_features_out(features_out)
{
	m_weights.reserve(features_out * features_in);
	if (use_bias) {
		m_biases.reserve(features_out);
	}

	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> uniform(-1.0, 1.0);

	for (size_t o = 0; o < features_out; o++)  {
		for (size_t i = 0; i < features_in; i++) {
			float val = uniform(engine); // Random number from uniform distribution
			m_weights.emplace_back(m_tape->create_leaf(val), m_tape);
		}

		if (use_bias) { 
			m_biases.emplace_back(m_tape->create_leaf(0.0f), m_tape);
		}
	}
}

std::vector<Value> Linear::operator()(const std::vector<Value>& input) {
	assert(input.size() == m_features_in);

	std::vector<Value> out_values;
	out_values.reserve(m_features_out);

	for (size_t o = 0; o < m_features_out; o++) {
		Value sum = m_weights[o * m_features_in + 0] * input[0];  // Start with first weight * first input

		for (size_t i = 1; i < m_features_in; i++) {
			size_t w_idx = o * m_features_in + i;
			sum = sum + m_weights[w_idx] * input[i];
		}

		if (!m_biases.empty()) {
			sum = sum + m_biases[o];
		}

		out_values.push_back(sum);
	}
	return out_values;
}

std::vector<Value> Linear::parameters() const {
	std::vector<Value> params = m_weights;
	params.insert(params.end(), m_biases.begin(), m_biases.end());
	return params;
}

std::string Linear::description() const {
	std::ostringstream oss;

	oss << "Linear("
		<< "in_features=" << m_features_in
		<< ", out_features=" << m_features_out
		<< ", bias=" << (!m_biases.empty() ? "true" : "false")
		<< ")";

	return oss.str();	
}