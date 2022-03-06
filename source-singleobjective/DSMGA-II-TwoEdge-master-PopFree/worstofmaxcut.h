#ifndef WOMX_FN
#define WOMX_FN

#include <cstddef>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

struct MaxCutInstance
{
	size_t l;
	std::vector<std::tuple<size_t, size_t, long>> edges;
};

struct WorstOfMaxCutInstance
{
    size_t l;
    std::vector<MaxCutInstance> instances;
};

MaxCutInstance load_maxcut(std::istream& stream);
MaxCutInstance load_maxcut(std::filesystem::path& path);

#endif