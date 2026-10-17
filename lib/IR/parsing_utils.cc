#include "parsing_utils.h"
#include <cerrno>
#include <cstdlib>

namespace causalflow::avelang::ir {

std::optional<long long> ParseConstantInteger(const std::string &value) {
    if (value.empty())
        return std::nullopt;

    char *end;
    errno = 0;
    long long result = std::strtoll(value.c_str(), &end, 10);

    // Check if the entire string was consumed and conversion was successful
    if (end != value.c_str() && *end == '\0' && errno != ERANGE) {
        return result;
    }

    return std::nullopt;
}

std::optional<double> ParseConstantFloat(const std::string &value) {
    if (value.empty())
        return std::nullopt;

    char *end;
    double result = std::strtod(value.c_str(), &end);

    // Check if the entire string was consumed and conversion was successful
    if (end != value.c_str() && *end == '\0') {
        return result;
    }

    return std::nullopt;
}

} // namespace causalflow::avelang::ir
