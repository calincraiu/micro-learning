#include "util/data/CSV.h"

namespace util::data::CSV {

    std::vector<std::string> split_csv(const std::string& line, char delim = ',') {
        std::vector<std::string> result;
        std::string field;
        bool in_quotes = false;

        for (char c : line) {
            if (c == '"') {
                in_quotes = !in_quotes;
            }
            else if (c == delim && !in_quotes) {
                result.push_back(field);
                field.clear();
            }
            else {
                field += c;
            }
        }
        if (!field.empty()) {
            result.push_back(field);
        }

        for (auto& field : result) {
            // Remove leading whitespace
            size_t start = field.find_first_not_of(" \t\r\n");
            // Remove trailing whitespace
            size_t end = field.find_last_not_of(" \t\r\n");
            
            if (start != std::string::npos && end != std::string::npos) {
                field = field.substr(start, end - start + 1);
            } else {
                field = "";
            }
        }

        return result;
    }

}
