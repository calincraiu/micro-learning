#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "core/Tape.h"

class Optimizer {
public:
	virtual void step() = 0;

	virtual ~Optimizer() = default;
};
	
#endif // OPTIMIZER_H