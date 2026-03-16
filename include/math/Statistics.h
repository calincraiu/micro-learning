#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>

namespace math::Statistics {
	float mean(const std::vector<float>& data);
	float std(const std::vector<float>& data);
}


#endif // STATISTICS_H