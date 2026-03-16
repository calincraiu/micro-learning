#include "nn/losses/MSE.h"
#include <vector>
#include <cassert>

MSE::MSE(Tape* tape) : m_tape(tape)
{
}

Value MSE::operator()(const Value& predicted, const Value& target)
{
	Value diff = predicted - target;
	return diff.pow(2.0f);
}

Value MSE::operator()(const std::vector<Value>& predicted, const std::vector<Value>& target)
{
	assert(predicted.size() == target.size());
	assert(!predicted.empty());
	
	Value diff = predicted[0] - target[0];
	Value sum = diff.pow(2.0f);
	
	for (size_t i = 1; i < predicted.size(); ++i) {
		Value d = predicted[i] - target[i];
		sum = sum + (d.pow(2.0f));
	}
	
	float scale = 1.0f / static_cast<float>(predicted.size());
	Value mse = sum * Value(m_tape->create_leaf(scale), m_tape);
	
	return mse;
}
