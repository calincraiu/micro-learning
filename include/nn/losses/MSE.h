#ifndef MSE_H
#define MSE_H

#include <vector>

#include "core/Value.h" 
#include "core/Tape.h" 

class MSE {
public:
	MSE(Tape* tape);

	Value operator()(const Value& predicted, const Value& target);
	Value operator()(const std::vector<Value>& predicted, const std::vector<Value>& target);

private:
	Tape* m_tape;
};

#endif // MSE_H