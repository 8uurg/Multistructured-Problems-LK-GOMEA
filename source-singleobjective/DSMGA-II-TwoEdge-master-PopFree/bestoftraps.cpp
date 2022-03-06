#include "bestoftraps.h"

// Evaluation
int trapFunctionBOT(int unitation, int size)
{
    if (unitation == size) return size;
    return size - unitation - 1;
}

template<typename T>
int evaluateConcatenatedPermutedTrap( PermutedRandomTrap &permutedRandomTrap, T&& getBoolAtIndex )
{
    int l = permutedRandomTrap.number_of_parameters;
    // int number_of_blocks = l / block_size;
    
    int objective = 0;
    for (int block_start = 0; block_start < l; block_start += permutedRandomTrap.block_size)
    {
        int unitation = 0;
        int current_block_size = std::min(permutedRandomTrap.block_size, l - block_start);
        for (int i = 0; i < current_block_size; ++i)
        {
            int idx = permutedRandomTrap.permutation[block_start + i];
            unitation += getBoolAtIndex(idx) == permutedRandomTrap.optimum[idx];
        }
        objective += trapFunctionBOT(unitation, current_block_size);
    }
    return objective;
}

template<typename T>
int evaluateBestOfTraps( BestOfTraps &bestOfTraps, T&& getBoolAtIndex, int &best_fn )
{
    int result = std::numeric_limits<int>::min();
    for (size_t fn = 0; fn < bestOfTraps.permutedRandomTraps.size(); ++fn)
    {
        int result_subfn = evaluateConcatenatedPermutedTrap(bestOfTraps.permutedRandomTraps[fn], getBoolAtIndex);
        if (result_subfn > result)
        {
            best_fn = fn;
            result = result_subfn;
        }
    }
    return result;
}

// Generation
PermutedRandomTrap generatePermutedRandomTrap(std::mt19937 &rng, int n, int k)
{
    // Generate permutation
    std::vector<size_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    // Generate optimum
    std::vector<char> optimum(n);
    std::uniform_int_distribution<char> binary_dist(0, 1);
    std::generate(optimum.begin(), optimum.end(), [&rng, &binary_dist](){return binary_dist(rng);});

    return PermutedRandomTrap {
        /* .number_of_parameters = */ n,
        /* .block_size = */ k,
        /* .permutation = */ permutation,
        /* .optimimum = */ optimum
    };
}

BestOfTraps generateBestOfTrapsInstance(int64_t seed, int n, int k, int fns)
{
    std::vector<PermutedRandomTrap> randomPermutedTraps(fns);
    std::mt19937 rng(seed);

    for (int fn = 0; fn < fns; ++fn)
    {
        randomPermutedTraps[fn] = generatePermutedRandomTrap(rng, n, k);
    }
    
    return BestOfTraps {
        randomPermutedTraps
    };
}

void writeBestOfTraps(std::filesystem::path outpath, BestOfTraps &bot)
{
    std::ofstream file(outpath);
    file << bot.permutedRandomTraps.size() << '\n';
    for (PermutedRandomTrap subfunction: bot.permutedRandomTraps)
    {
        file << subfunction.number_of_parameters << ' ';
        file << subfunction.block_size << '\n';
        // optimum
        for (int o = 0; o < subfunction.number_of_parameters; ++o)
        {
            file << static_cast<int>(subfunction.optimum[o]);
            if (o == subfunction.number_of_parameters - 1)
                file << '\n';
            else
                file << ' ';
        }
        // permutation
        for (int o = 0; o < subfunction.number_of_parameters; ++o)
        {
            file << subfunction.permutation[o];
            if (o == subfunction.number_of_parameters - 1)
                file << '\n';
            else
                file << ' ';
        }
    }
}

// Loading
void stopInvalidInstanceBOT(std::ifstream &stream, std::string expected)
{
    std::cerr << "Instance provided for BOT is invalid.\n";
    std::cerr << "Invalid character at position " << stream.tellg() << ".\n";
    std::cerr << expected << std::endl;
    exit(1);
}
void stopFileMissingBOT(std::filesystem::path file)
{
    std::cerr << "Instance provided for BOT is invalid.\n";
    std::cerr << "File " << file << " does not exist." << std::endl;
    exit(1);
}

BestOfTraps readBestOfTraps(std::filesystem::path inpath)
{
    if (! std::filesystem::exists(inpath)) stopFileMissingBOT(inpath);
    std::ifstream file(inpath);
    size_t number_of_subfunctions = 0;
    file >> number_of_subfunctions;
    if (file.fail()) stopInvalidInstanceBOT(file, "expected number_of_subfunctions"); 

    std::vector<PermutedRandomTrap> subfunctions;
    subfunctions.reserve(number_of_subfunctions);

    for (int fn = 0; fn < static_cast<int>(number_of_subfunctions); ++fn)
    {
        int number_of_parameters = -1;
        int block_size = -1;
        file >> number_of_parameters;
        if (file.fail()) stopInvalidInstanceBOT(file, "expected number_of_parameters"); 
        file >> block_size;
        if (file.fail()) stopInvalidInstanceBOT(file, "expected block_size");
        std::string current_line;
        // Skip to the next line.
        if(! std::getline(file, current_line)) stopInvalidInstanceBOT(file, "expected newline");
        // optimum
        std::vector<char> optimum;
        optimum.reserve(number_of_parameters);
        if(! std::getline(file, current_line)) stopInvalidInstanceBOT(file, "expected optimum");
        
        {
            std::stringstream linestream(current_line);
            int v = 0;
            while (!linestream.fail())
            {
                linestream >> v;
                optimum.push_back(static_cast<char>(v));
            }
        }
        std::vector<size_t> permutation;
        permutation.reserve(number_of_parameters);
        if(! std::getline(file, current_line)) stopInvalidInstanceBOT(file, "expected permutation");
        {
            std::stringstream linestream(current_line);
            int v = 0;
            while (!linestream.fail())
            {
                linestream >> v;
                permutation.push_back(v);
            }
        }
        
        PermutedRandomTrap prt = PermutedRandomTrap {
            number_of_parameters,
            block_size,
            permutation,
            optimum
        };

        subfunctions.push_back(prt);
    }
    return BestOfTraps {
        subfunctions
    };
}

void stopInvalidInstanceSpecifierBOT(std::stringstream &stream, std::string expected)
{
    int fail_pos = stream.tellg();
    std::string s;
    stream >> s;
    std::cerr << "While loading Best of Traps instance string was invalid at position " << fail_pos << ".\n";
    std::cerr << expected << ". Remainder is `" << s << "`.\n";
    std::cerr << "Expected format is `g(e?)_<n>_<k>_<fns>_<seed>` for generating a bot instance (include e to export to file).\n";
    std::cerr << "and `l_<path>` for loading from a file.\n";
    std::cerr << "Each instance is their own dimension, and each instance/dimension is separated by a `;`" << std::endl; //  or `f_<path>`
    exit(1);
}

// Load or Generate from instance string.
BestOfTraps loadBestOfTraps(std::string &instance, int number_of_variables)
{
    std::stringstream instance_stream(instance);

    while (! instance_stream.eof())
    {
        std::string t;
        if(! std::getline(instance_stream, t, '_')) stopInvalidInstanceSpecifierBOT(instance_stream, "expected <str>_\n");

        if (t[0] == 'g') // Generate
        {
            bool write_to_file = t.length() >= 2 && t[1] == 'e';
            
            // Parameters, initialized to silence warnings.
            int n = -1, k = -1, fns = -1; size_t seed = 0U;
            instance_stream >> n;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> k;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> fns;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> seed;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");

            BestOfTraps bot = generateBestOfTrapsInstance(seed, n, k, fns);
            if (write_to_file)
            {
                std::filesystem::path botoutdirectory = "./bestoftraps/";
                if (!std::filesystem::exists(botoutdirectory))
                {
                    std::filesystem::create_directories(botoutdirectory);
                }
                std::string filename = "bot_n" + std::to_string(n) + "k" + std::to_string(k) + "fns" + std::to_string(fns) + "s"  + std::to_string(seed) + ".txt";
                std::filesystem::path botoutpath = botoutdirectory / filename;
                writeBestOfTraps(botoutpath, bot);
            }

            for (PermutedRandomTrap subfunction: bot.permutedRandomTraps)
            {
                assert(number_of_variables >= subfunction.number_of_parameters);
            }

            return bot;

            // if(instance_stream.get() != ';') break;
        }
        else if (t[0] == 'f')
        {
            std::string botinpathstr;
            if(! std::getline(instance_stream, botinpathstr, ';')) stopInvalidInstanceSpecifierBOT(instance_stream, "expected path");
            std::filesystem::path botinpath = botinpathstr;
            BestOfTraps bot = readBestOfTraps(botinpath);
            
            return bot;
        }
        else stopInvalidInstanceSpecifierBOT(instance_stream, "expected one of {`g`, `f`}");
    }

    stopInvalidInstanceSpecifierBOT(instance_stream, "expected one of {`g`, `f`}");
}