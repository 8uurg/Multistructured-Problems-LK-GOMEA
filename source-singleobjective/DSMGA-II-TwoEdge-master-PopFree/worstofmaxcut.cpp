#include "worstofmaxcut.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <stdexcept>

MaxCutInstance load_maxcut(std::istream& stream)
{
    size_t l = 0;
    size_t num_edges = 0;
    std::vector<std::tuple<size_t, size_t, long>> edges;
    stream >> l >> num_edges;
    if (stream.fail()) throw std::invalid_argument("invalid instance - start");
    while (!stream.eof())
    {
        size_t i = 0;
        size_t j = 0;
        long w = 0;
        stream >> i >> j >> w;
        if (!stream.eof() && stream.fail()) throw std::invalid_argument("invalid instance - edges");
        if (stream.eof()) break;
        // Note: vertices are 1-indexed, so remove 1!
        edges.push_back({i - 1, j - 1, w});
    }
    assert(edges.size() == num_edges);
    return MaxCutInstance { l , edges }; 
}

MaxCutInstance load_maxcut(std::filesystem::path& path)
{
    std::ifstream file(path);
    // std::cout << "Loading " << path << "\n";
    return load_maxcut(file);
}
