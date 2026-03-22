#ifndef CSV_H
#define CSV_H

#include <vector>
#include <string>

namespace util::data::CSV {
	std::vector<std::string> split_csv(const std::string& line, char delim);
}


#endif // CSV_H