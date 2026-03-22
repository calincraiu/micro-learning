#include <assert.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "util/data/Dataset.h"
#include "util/data/CSV.h"

Dataset::Dataset(const std::vector<std::vector<float>>& features,
    const std::vector<float>& targets)
    : m_features(features), m_targets(targets)
{
    assert(features.size() == targets.size());
}

std::pair<std::vector<float>, float> Dataset::operator[](size_t idx) const {
    return { m_features[idx], m_targets[idx] };
}

Dataset Dataset::from_csv(const std::string& filename, bool has_header = true, char delimiter = ',') {
    std::vector<std::vector<float>> features;
    std::vector<float> targets;

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    bool first_row = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue; // skip empty lines

        auto tokens = util::data::CSV::split_csv(line, delimiter);

        if (tokens.empty()) continue;

        // Skip header if requested
        if (first_row && has_header) {
            first_row = false;
            continue;
        }

        if (tokens.size() < 2) {
            throw std::runtime_error("Row has too few columns: " + line);
        }

        std::vector<float> row_features;
        row_features.reserve(tokens.size() - 1);

        // Parse all but last column as features
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            try {
                row_features.push_back(std::stof(tokens[i]));
            }
            catch (...) {
                throw std::runtime_error("Invalid float in features: " + tokens[i]);
            }
        }

        // Last column = target
        try {
            float target = std::stof(tokens.back());
            features.push_back(std::move(row_features));
            targets.push_back(target);
        }
        catch (...) {
            throw std::runtime_error("Invalid float in target: " + tokens.back());
        }

        first_row = false;
    }

    if (features.empty()) {
        throw std::runtime_error("No valid data rows found in " + filename);
    }

    return Dataset(std::move(features), std::move(targets));
}