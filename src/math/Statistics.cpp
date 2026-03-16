#include <numeric>
#include <cmath>

#include "math/Statistics.h"

namespace math::Statistics {

    float mean(const std::vector<float>& data) {
        if (data.empty()) return 0.0;
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }
    
    float std(const std::vector<float>& data) {
        
        float mean = math::Statistics::mean(data);
        float varianceSum = 0;
        for (float x : data) {
            varianceSum += std::pow(x - mean, 2);
        }

        float variance = varianceSum / data.size();
        float stdDev = std::sqrt(variance);

        return stdDev;
    }
}