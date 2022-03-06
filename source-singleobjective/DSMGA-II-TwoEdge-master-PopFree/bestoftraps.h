#ifndef BOT_FN
#define BOT_FN

#include <cassert>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <limits>
#include <vector>

// Definition
struct PermutedRandomTrap
{
    int number_of_parameters;
    int block_size;
    std::vector<size_t> permutation;
    std::vector<char> optimum;
};

struct BestOfTraps
{
    std::vector<PermutedRandomTrap> permutedRandomTraps;
};

// Evaluation
int trapFunctionBOT(int unitation, int size);
template<typename T>
int evaluateConcatenatedPermutedTrap( PermutedRandomTrap &permutedRandomTrap, T&& getBoolAtIndex );

template<typename T>
int evaluateBestOfTraps( BestOfTraps &bestOfTraps, T&& getBoolAtIndex, int &best_fn );

// Generation
PermutedRandomTrap generatePermutedRandomTrap(std::mt19937 &rng, int n, int k);

BestOfTraps generateBestOfTrapsInstance(int64_t seed, int n, int k, int fns);

void writeBestOfTraps(std::filesystem::path outpath, BestOfTraps &bot);
// Loading
BestOfTraps readBestOfTraps(std::filesystem::path inpath);

// Load or Generate from instance string.
BestOfTraps loadBestOfTraps(std::string &instance, int number_of_variables);

#endif