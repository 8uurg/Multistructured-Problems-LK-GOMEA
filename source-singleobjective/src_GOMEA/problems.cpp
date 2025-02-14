//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "problems.hpp"
#include <filesystem>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <vector>

void createProblemInstance(int problemIndex, int numberOfVariables, Config *config, Problem **problemInstance, string &instancePath, int k, int s)
{
    switch (problemIndex)
    {
        case 0: *problemInstance = new oneMax(); break;
        case 1: *problemInstance = new concatenatedDeceptiveTrap(config->k, config->s, false); break;
        case 2: *problemInstance = new ADF(instancePath); break;
        case 3: *problemInstance = new maxCut(instancePath); break;     
        case 4: *problemInstance = new hierarchialIfAndOnlyIf(); break;
        case 6: *problemInstance = new hierarchialDeceptiveTrap(3); break;
        case 7: *problemInstance = new concatenatedDeceptiveTrap(config->k, config->s, true); break;
        case 8: *problemInstance = new MAXSAT(instancePath); break;
        case 9: *problemInstance = new SpinGlass(instancePath); break;
        case 10: *problemInstance = new PythonFunction(config->functionName, instancePath); break;

        case 11: *problemInstance = new zeroOneMax(); break;
        case 12: *problemInstance = new closestDistanceToGridPoints(config->k); break;
        case 13: *problemInstance = new bimodalModifiedTrap(config->k, config->s); break;
        case 14: *problemInstance = new maxConcatenatedPermutedTraps(config->k, config->s, config->fn); break;
        case 15: *problemInstance = new BestOfTrapsProblem(instancePath); break;
        case 16: *problemInstance = new WorstOfTrapsProblem(instancePath); break;
        case 17: *problemInstance = new WorstOfMaxcutProblem(instancePath); break;

        default:
        {
            cerr << "No problem with index #" << problemIndex << " installed!\n";
            exit(0);
        };
    }
    (*problemInstance)->initializeProblem(config, numberOfVariables);
}

bool problemNameByIndex(Config *config, string &problemName)
{
    switch (config->problemIndex)
    {
        case 0: problemName = "OneMax"; break;
        case 1: problemName = "Concatenated Deceptive Trap with k=" + to_string(config->k) + " s=" + to_string(config->s); break;
        case 2: problemName = "ADF with k=" + to_string(config->k) + " s=" + to_string(config->s); break;
        case 3: problemName = "MaxCut"; break;
        case 4: problemName = "Hierarhical If-And-Only-If"; break;
        case 5: problemName = "Leading Ones"; break;        
        case 6: problemName = "Hierarhical Deceptive Trap-3"; break;
        case 7: problemName = "Concatenated Bimodal Deceptive Trap with k=" + to_string(config->k) + " s=" + to_string(config->s); break;
        case 8: problemName = "MAXSAT"; break;
        case 9: problemName = "SpinGlass"; break;
        case 10: problemName = "custom PythonFunction "; break;

        case 11: problemName = "ZeroOneMax"; break;
        case 12: problemName = "Closest Point Distance with k=" + to_string(config->k); break;
        case 13: problemName = "Bimodal Modified Trap with k=" + to_string(config->k) + " s=" + to_string(config->s); break;
        case 14: problemName = "Maximum over random permuted & flipped traps with fn=" + to_string(config->fn) + " k=" + to_string(config->k) + " seed=" + to_string(config->s); break;
        case 15: problemName = "Best of Traps"; break;
        case 16: problemName = "Worst of Traps"; break;
        case 17: problemName = "Worst of MaxCuts"; break;

        default: return false; break;
    }
    return true;
}

double Problem::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    return 0.0;
}

double oneMax::calculateFitness(Individual *solution)
{
    double res = 0.0;
    for (size_t i = 0; i < solution->genotype.size(); ++i)
        res += solution->genotype[i];

    solution->fitness = res;
    return res;
}

double oneMax::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    double res = fitnessBefore;
    for (size_t i = 0; i < touchedGenes.size(); ++i)
    {
        int touchedGene = touchedGenes[i];
        res -= solutionBefore->genotype[touchedGene];
        res += solution->genotype[touchedGene];
    }
    solution->fitness = res;
    return res;
}

double zeroOneMax::calculateFitness(Individual *solution)
{
    double res = 0.0;
    for (size_t i = 0; i < solution->genotype.size(); ++i)
    {
        res += solution->genotype[i];
    }
    res = max(res, solution->genotype.size() - res);
    solution->fitness = res;
    return res;
}

double zeroOneMax::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    // TODO: Caching is required to identify which of the two branches (zeroMax, oneMax) we are in.
    // For now: just reevaluate completely.
    return calculateFitness(solution);
}

double closestDistanceToGridPoints::calculateFitness(Individual *solution)
{
    // This problem consists of 2k points, equidistantly spread. For k=1 (default) this is equivalent to zeroOneMax.
    vector<double> distances;
    distances.resize(k);
    fill(distances.begin(), distances.end(), 0.0);
    for (size_t i = 0; i < solution->genotype.size(); ++i)
    {
        for(size_t j = 0; j < k; ++j)
        {
            // distances[j] += abs((double) (i/(1 + j) % 2) - ((double) solution->genotype[i]));
            distances[j] += abs((double) (i/(1 << j) % 2) - ((double) solution->genotype[i]));
        }
    }
    double res = distances[0];
    res = max(res, solution->genotype.size() - res);
    for(size_t j = 1; j < k; ++j)
    {
        res = max(res, max(distances[j], solution->genotype.size() - distances[j]));
    }
    solution->fitness = res;
    return res;
}

double closestDistanceToGridPoints::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    // TODO: Caching is required to identify which point is the closest.
    // For now: just reevaluate completely.
    return calculateFitness(solution);
}

double deceptiveTrap(int unitation, int k)
{
    double result = 0.0;
    if (unitation != k)
        result = k - 1 - unitation;
    else
        result = unitation;
    
    return (double)result;
}

double bimodalDeceptiveTrap(int unitation, int k)
{
    double result = 0.0;
    if (k == 10)
    {
        if (unitation == 0 || unitation == 10)
            result = 10;
        else if (unitation == 5)
            result = 9;
        else if (unitation == 1 || unitation == 9)
            result = 5;
        else if (unitation == 2 || unitation == 8)
            result = 0;
        else if (unitation == 3 || unitation == 7)
            result = 3;
        else if (unitation == 4 || unitation == 6)
            result = 6;
    }
    else if (k == 6)
    {
        if (unitation == 0 || unitation == 6)
            result = 6;
        else if (unitation == 1 || unitation == 5)
            result = 0;
        else if (unitation == 2 || unitation == 4)
            result = 2;
        else if (unitation == 3)
            result = 5;
    }   
    else
    {
        cout << "Bimodal Trap with size " << k << " not implemented!";
        exit(0);
    }

    return (double)result;
}

void bimodalModifiedTrap::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    // TODO: Partial evaluation
    // trapsForVariable_a.resize(numberOfVariables);
    // trapsForVariable_b.resize(numberOfVariables);

    // for (int i = 0; i < numberOfVariables; i += s)
    // {
    //     for (int j = i; j < i + k; j++)         
    //         trapsForVariable_a[j % numberOfVariables].push_back(i);
    // }
};
    
double bimodalModifiedTrap::calculateFitness(Individual *solution)
{
    // Function 1 is tightly encoded.
    double res_a = 0.0;
    for (int i = 0; i < numberOfVariables; i += s)
    {
        //cout << i << s << endl;
        int unitation = 0;
        int n_vars = 0;
        for (int j = i; j < i + k; j++)
        {
            size_t idx = j%numberOfVariables;
            // unitation += solution->genotype[idx];
            // To avoid overlapping the local optimum and global
            // optimum of the other subfunction, perform an xor with
            // 01010101 repeating.
            unitation += abs((int) (idx % 2) - (int) solution->genotype[idx]);
            n_vars += 1;
        }

        res_a += deceptiveTrap(unitation, n_vars);
    }

    // Function 2 is loosely encoded and is inverted.
    double res_b = 0.0;
    for (int i = 0; i < k; i++)
    {
        //cout << i << s << endl;
        int unitation = 0;
        int n_vars = 0;
        for (int j = i; j < numberOfVariables; j += s)
        {
            unitation += solution->genotype[j%numberOfVariables];
            n_vars += 1;
        }

        res_b += deceptiveTrap(n_vars - unitation, n_vars);
    }

    // double res = res_a;
    double res = max(res_a, res_b);

    solution->fitness = res;
    return res;
}

double bimodalModifiedTrap::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    // As with the previous function, caching is required due to the max function.
    return calculateFitness(solution);
}

bool bimodalModifiedTrap::isKeySolution(vector<char> &genotype, double fitness)
{
    // Note, this performs standard evaluation.
    double res_a = 0.0;
    double res_a_best = 0.0;
    for (int i = 0; i < numberOfVariables; i += s)
    {
        //cout << i << s << endl;
        int unitation = 0;
        int n_vars = 0;
        for (int j = i; j < i + k; j++)
        {
            size_t idx = j%numberOfVariables;
            // unitation += solution->genotype[idx];
            // To avoid overlapping the local optimum and global
            // optimum of the other subfunction, perform an xor with
            // 01010101 repeating.
            unitation += abs((int) (idx % 2) - (int) genotype[idx]);
            n_vars += 1;
        }

        res_a += deceptiveTrap(unitation, n_vars);
        res_a_best += n_vars;
    }

    double res_b = 0.0;
    double res_b_best = 0.0;
    for (int i = 0; i < k; i++)
    {
        //cout << i << s << endl;
        int unitation = 0;
        int n_vars = 0;
        for (int j = i; j < numberOfVariables; j += s)
        {
            unitation += genotype[j%numberOfVariables];
            n_vars += 1;
        }

        res_b += deceptiveTrap(n_vars - unitation, n_vars);
        res_b_best += n_vars;
    }

    return res_a == res_a_best || 
           res_b == res_b_best;
}


void maxConcatenatedPermutedTraps::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    
    auto problem_rng = mt19937(this->s);

    permutations.resize(this->fn);
    optima.resize(this->fn);

    for (size_t sfn_idx = 0; sfn_idx < this->fn; ++sfn_idx)
    {
        // Generate random permutation
        vector<size_t> permutation(numberOfVariables);
        iota(permutation.begin(), permutation.end(), 0);
        shuffle(permutation.begin(), permutation.end(), problem_rng);
        
        // Generate random bits for optimum.
        uniform_int_distribution<char> rand_bit(0, 1);
        vector<char> optimum(numberOfVariables);
        generate(optimum.begin(), optimum.end(), [&rand_bit, &problem_rng]() { return rand_bit(problem_rng); });

        // Store
        permutations[sfn_idx] = permutation;
        optima[sfn_idx] = optimum;
    }
};


vector<double> maxConcatenatedPermutedTraps::calculateFitnessSubfunctions(vector<char> &genotype)
{
    vector<double> res(this->fn);

    for (size_t sfn_idx = 0; sfn_idx < this->fn; ++sfn_idx)
    {
        double res_sfn = 0.0;
        for ( size_t ssfn_idx = 0; ssfn_idx < numberOfVariables; ssfn_idx += k )
        {
            int unitation = 0;
            for (size_t i = ssfn_idx; i < min(ssfn_idx + k, (size_t) numberOfVariables); ++i)
            {
                size_t v_idx = this->permutations[sfn_idx][i];
                unitation += genotype[v_idx] == this->optima[sfn_idx][i];
            }
            int trap_value = deceptiveTrap(unitation, this->k);
            res_sfn += (double) trap_value;
        }
        res[sfn_idx] = res_sfn;
    }

    return res;
}

double maxConcatenatedPermutedTraps::calculateFitness(Individual *solution)
{
    vector<double> subfitnesses = maxConcatenatedPermutedTraps::calculateFitnessSubfunctions(solution->genotype);

    assert(subfitnesses.size() > 0);

    double res = subfitnesses[0];

    for (double f: subfitnesses)
    {
        res = max(res, f);
    }

    solution->fitness = res;
    return res;
}

bool maxConcatenatedPermutedTraps::isKeySolution(vector<char> &genotype, double fitness)
{
    // vector<double> subfitnesses = calculateFitnessSubfunctions(genotype);

    return fitness == numberOfVariables;
}

double maxConcatenatedPermutedTraps::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    // As with the previous function, caching is required due to the max function.
    return calculateFitness(solution);
}

vector<vector<int>> maxConcatenatedPermutedTraps::getProblemFOS()
{
    const size_t blocks_per_function = this->numberOfVariables / this->k;
    const size_t total_number_of_blocks = blocks_per_function * this->fn;

    vector<vector<int>> fos(total_number_of_blocks);

    for (size_t fn_idx = 0; fn_idx < this->fn; ++fn_idx)
    {
        for (size_t block_idx = 0; block_idx < blocks_per_function; ++block_idx)
        {
            for (size_t i = block_idx*this->k; i < min((size_t) numberOfVariables, (block_idx + 1)*this->k); ++i)
            {
                fos[fn_idx*blocks_per_function + block_idx].push_back(permutations[fn_idx][i]);
            }
        }
    }

    return fos;
}

void concatenatedDeceptiveTrap::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    trapsForVariable.resize(numberOfVariables);

    for (int i = 0; i < numberOfVariables; i += s)
    {
        for (int j = i; j < i + k; j++)         
            trapsForVariable[j % numberOfVariables].push_back(i);
    }
};
    
double concatenatedDeceptiveTrap::calculateFitness(Individual *solution)
{
    double res = 0.0;
    for (int i = 0; i < numberOfVariables; i += s)
    {
        //cout << i << s << endl;
        int unitation = 0;
        for (int j = i; j < i + k; j++)
            unitation += solution->genotype[j%numberOfVariables];

        if (!bimodal)
            res += deceptiveTrap(unitation, k);
        else if (bimodal)
            res += bimodalDeceptiveTrap(unitation, k);
    }

    solution->fitness = res;
    return res;
}

double concatenatedDeceptiveTrap::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    double res = fitnessBefore;

    unordered_set<int> touchedTraps; //starting positions of all touched traps
    for (int i = 0; i < touchedGenes.size(); i++)
    {
        int touchedGene = touchedGenes[i];
        for (int j = 0; j < trapsForVariable[touchedGene].size(); ++j)
        {
            touchedTraps.insert(trapsForVariable[touchedGene][j]);
        }
    }

    for (unordered_set<int>::iterator it = touchedTraps.begin(); it != touchedTraps.end(); ++it)
    {
        int unitation = 0, unitationBefore = 0;

        for (int j = *it; j < *it + k; j++)
        {
            unitationBefore += solutionBefore->genotype[j];
            unitation += solution->genotype[j];
        }
        res -= deceptiveTrap(unitationBefore, k);
        res += deceptiveTrap(unitation, k);
    }

    solution->fitness = res;
    return res;
}


void ADF::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    subfunctionsForVariable.resize(numberOfVariables);

    ifstream inFile(problemInstanceFilename, ifstream::in);
    if (inFile.fail())
    {
        cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
        exit(0);
    }
    int N, numFunctions;
    inFile >> N >> numFunctions;
    //cout << N << " " << numFunctions << endl;

    for (int i = 0; i < numFunctions; ++i)
    {
        NKSubfunction subfunction;

        int k;
        inFile >> k;

        for (int j = 0; j < k; ++j)
        {
            int var;
            inFile >> var;
            subfunction.variablesPositions.push_back(var);
        }
        int numCombinations = 1 << k;
        subfunction.valuesTable.resize(numCombinations);
        for (int j = 0; j < numCombinations; ++j)
        {
            string combination;
            inFile >> combination;
            int index = 0, pow = 1;
            for (size_t p = combination.size()-1; p > 0; p--)
            {
                if (combination[p] == '0' || combination[p] == '1')
                {
                    index += ((int)(combination[p] == '1') * pow);
                    pow *= 2;
                }
            }
            inFile >> subfunction.valuesTable[index];           

        }
        subfunctions.push_back(subfunction);

        //needed for partial evaluations
        for (size_t p = 0; p < subfunction.variablesPositions.size(); ++p)
        {
            subfunctionsForVariable[subfunction.variablesPositions[p]].push_back(subfunctions.size()-1);
        }

        //VIG.push_back(subfunction.variablesPositions);

    }
    inFile.close();
};


double ADF::calculateFitness(Individual *solution)
{
    long double res = 0.0;
    
    for (size_t i = 0; i < subfunctions.size(); ++i)
    {
        int index = 0, pow = 1;
        for (int j = subfunctions[i].variablesPositions.size()-1; j >= 0; --j)
        {
            int pos = subfunctions[i].variablesPositions[j];
            index += solution->genotype[pos] * pow;
            pow *= 2;
        }
        res += subfunctions[i].valuesTable[index];// * multiplier;
    }
    
    solution->fitness = (double)res;
    return (double)res;
}

double ADF::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    double res = fitnessBefore;

    unordered_set<int> touchedSubfunctions;
    for (int i = 0; i < touchedGenes.size(); i++)
    {
        int touchedGene = touchedGenes[i];
        for (int j = 0; j < subfunctionsForVariable[touchedGene].size(); ++j)
        {
            touchedSubfunctions.insert(subfunctionsForVariable[touchedGene][j]);
        }
    }

    for (unordered_set<int>::iterator it = touchedSubfunctions.begin(); it != touchedSubfunctions.end(); ++it)
    {
        int index = 0, indexBefore = 0, pow = 1;
        for (int j = subfunctions[*it].variablesPositions.size()-1; j >= 0; --j)
        {
            int pos = subfunctions[*it].variablesPositions[j];
            index += solution->genotype[pos] * pow;
            indexBefore += solutionBefore->genotype[pos] * pow;         
            pow *= 2;
        }
        
        res -= subfunctions[*it].valuesTable[indexBefore];      
        res += subfunctions[*it].valuesTable[index];

    }
    //cout << res << endl;

    solution->fitness = res;
    return res;
}

vector<vector<int>> ADF::getProblemFOS()
{
    vector<vector<int>> fos(this->subfunctions.size());

    for (size_t idx = 0; idx < this->subfunctions.size(); ++idx)
    {
        auto sf = this->subfunctions[idx];
        
        fos[idx].resize(sf.variablesPositions.size());
        copy(sf.variablesPositions.begin(), sf.variablesPositions.end(), fos[idx].begin());
    }

    return fos;
}

double hierarchialDeceptiveTrap::generalTrap(int unitation, double leftPeak, double rightPeak)
{
    if (unitation == -1)
        return 0; 

    if (unitation == k)
        return rightPeak;
    
    return leftPeak * (1.0 - unitation / (k - 1.0));
};

double hierarchialDeceptiveTrap::calculateFitness(Individual *solution)
{
    double res = 0.0;
    int blockSize = k;
    double leftPeak, rightPeak;
    for (int i = 0; i < numberOfVariables; ++i)
        transform[i] = (int)solution->genotype[i];

    while (blockSize <= numberOfVariables)
    {
        for (int i = 0; i < numberOfVariables; i += blockSize)
        {
            int unitation = 0;
            for (int j = i; j < i + blockSize; j += blockSize / k)
            {
                if (transform[j] != 0 && transform[j] != 1)
                {
                    unitation = -1;
                    break;
                }
                unitation += transform[j];
            }
            
            if (unitation == 0)
                transform[i] = 0;
            else if (unitation == k)
                transform[i] = 1;
            else
                transform[i] = -1;

            if (blockSize < numberOfVariables)
            {
                leftPeak = 1.0;
                rightPeak = 1.0;
            }
            else
            {
                leftPeak = 0.9;
                rightPeak = 1.0;
            }
            res += generalTrap(unitation, leftPeak, rightPeak) * (blockSize / 3);
        }
        blockSize *= k;
    }
    solution->fitness = res;
    return res;
}


double hierarchialIfAndOnlyIf::calculateFitness(Individual *solution)
{
    double res = 0.0;
    int blockSize = 1;
    while (blockSize <= numberOfVariables)
    {
        for (int i = 0; i < numberOfVariables; i+=blockSize)
        {
            int unitation = 0;
            for (int j = i; j < i + blockSize; ++j)
                unitation += solution->genotype[j];

            if (unitation == blockSize || unitation == 0)
                res += blockSize;
            //cout << blockSize << "  " << res << endl;
        }
        blockSize *= 2;
    }
    solution->fitness = res;
    return res;
}


void maxCut::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    edgesForVariable.resize(numberOfVariables);

    ifstream inFile(problemInstanceFilename, ifstream::in);
    if (inFile.fail())
    {
        cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
        exit(0);
    }
    int N, numEdges;
    inFile >> N >> numEdges;
    
    assert(N <= numberOfVariables);

    //cout << problemInstanceFilename << " " << N << "  " << numEdges << endl;

    for (int i = 0; i < numEdges; ++i)
    {
        int v1, v2;
        double w;
        inFile >> v1 >> v2 >> w;
        edges.push_back(make_pair(make_pair(v1-1, v2-1), w));
        //cout << v1 << " " << v2 << " " << w << endl;
        edgesForVariable[v1-1].push_back(i);
        edgesForVariable[v2-1].push_back(i);        
    }

    inFile.close();
}


double maxCut::calculateFitness(Individual *solution)
{
    double res = 0.0;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        int v1 = edges[i].first.first;
        int v2 = edges[i].first.second;
        double w = edges[i].second;

        if (solution->genotype[v1] != solution->genotype[v2])
            res += w;
    }

    solution->fitness = res;
    return res;
}

double maxCut::calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore)
{
    double res = fitnessBefore;

    unordered_set<int> touchedEdges;
    for (int i = 0; i < touchedGenes.size(); i++)
    {
        int touchedGene = touchedGenes[i];
        for (int j = 0; j < edgesForVariable[touchedGene].size(); ++j)
        {
            touchedEdges.insert(edgesForVariable[touchedGene][j]);
        }
    }

    for (unordered_set<int>::iterator it = touchedEdges.begin(); it != touchedEdges.end(); ++it)
    {
        int v1 = edges[*it].first.first;
        int v2 = edges[*it].first.second;
        double w = edges[*it].second;

        if (solutionBefore->genotype[v1] != solutionBefore->genotype[v2])
            res -= w;

        if (solution->genotype[v1] != solution->genotype[v2])
            res += w;
    }

    solution->fitness = res;
    return res;
}

void Clustering::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    
    ifstream inFile(problemInstanceFilename, ifstream::in);
    if (inFile.fail())
    {
        cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
        exit(0);
    }
    int N;
    inFile >> N >> Dim;
    points.resize(N);
    for (int i = 0; i < N; ++i)
    {
        points[i].resize(Dim);
        for (int j = 0; j < Dim; ++j)
            inFile >> points[i][j];
        //cout << points[i][0];
    }

    inFile.close();
    distances.resize(numberOfVariables);
    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        distances[i].resize(numberOfVariables);

        for (size_t j = i+1; j < numberOfVariables; ++j)
        {
            double dist = 0.0;

            for (int k = 0; k < Dim; ++k)
                dist += (points[i][k] - points[j][k])*(points[i][k] - points[j][k]);
        
            dist = pow(dist, 1.0/float(Dim));

            distances[i][j] = dist;
        }
    }
}

double Clustering::calculateFitness(Individual *solution)
{
    double res = 0.0;
    double intra_cluster_dist = 0.0, inter_cluster_dist = 0.0;
    int counter1=0, counter2=0;

    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        for (size_t j = i+1; j < numberOfVariables; ++j)
        {
            if (solution->genotype[i] != solution->genotype[j])
            {
                inter_cluster_dist += distances[i][j];
                counter1 += 1;
                //if (distances[i][j] < min_dist)
                //  min_dist = distances[i][j];
            }
            else
            {
                intra_cluster_dist += distances[i][j];
                counter2 += 1;
                //if (distances[i][j] > max_dist)
                //  max_dist = distances[i][j];
            }
        }
    }

    inter_cluster_dist /= counter1;
    intra_cluster_dist /= counter2;
    
    if (counter1 == 0 or counter2 == 0)
        res = -1;
    else
        res = inter_cluster_dist / intra_cluster_dist;

    solution->fitness = res;
    return res;
}

void MAXSAT::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    subfunctionsForVariable.resize(numberOfVariables);

    ifstream inFile(problemInstanceFilename, ifstream::in);
    if (inFile.fail())
    {
        cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
        exit(0);
    }
    
    string line;
    for (int i = 0; i < 8; ++i)
        getline(inFile, line);

    while(true)
    {
        vector<int> subfunction;
        vector<int> subfunctionSigns;
        int var;
        inFile >> var;
        if (inFile.fail())
            break;
        if (var < 0)
        {
            subfunctionSigns.push_back(-1);
            var++;
        }
        else if (var > 0)
        {
            var--;
            subfunctionSigns.push_back(1);
        }
        var = abs(var);
        subfunction.push_back(var);

        while (true)
        {
            inFile >> var;
            if (var == 0)
                break;

            if (var < 0)
            {
                subfunctionSigns.push_back(-1);
                var++;
            }
            else if (var > 0)
            {
                var--;
                subfunctionSigns.push_back(1);
            }

            var = abs(var);
            subfunction.push_back(var);
        }
        for (int i = 0; i < subfunction.size(); ++i)
        {
            //cout << subfunction[i] << " | " << subfunctionSigns[i] << " ";
            subfunctionsForVariable[subfunction[i]].push_back(subfunctions.size());
        }
        //cout << endl;
        subfunctions.push_back(subfunction);
        signs.push_back(subfunctionSigns);

    }
    inFile.close();
};


double MAXSAT::calculateFitness(Individual *solution)
{
    long double res = 0.0;
    
    for (size_t i = 0; i < subfunctions.size(); ++i)
    {
        bool b = false;
        for (int j = 0; j < subfunctions[i].size(); ++j)
        {
            int var = subfunctions[i][j];
            int sign = signs[i][j];
            if (sign > 0 && solution->genotype[var] == 1)
                b = true;
            else if (sign < 0 && solution->genotype[var] == 0)
                b = true;           
        }
        if (b == false)
            res -= 1;       
    }
    
    solution->fitness = (double)res;
    return (double)res;
}

vector<vector<int>> MAXSAT::getProblemFOS()
{
    return this->subfunctions;
}



void SpinGlass::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    subfunctionsForVariable.resize(numberOfVariables);

    ifstream inFile(problemInstanceFilename, ifstream::in);
    if (inFile.fail())
    {
        cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
        exit(0);
    }
    
    double tmp1, tmp2;
    inFile >> tmp1 >> tmp2;
    
    while(true)
    {
        vector<int> subfunction;
        int var1, var2, sign;
        inFile >> var1 >> var2 >> sign;
        if (inFile.fail())
            break;
        
        var1--;
        var2--;

        subfunction.push_back(var1);
        subfunction.push_back(var2);
        subfunction.push_back(sign);        
        subfunctionsForVariable[var1].push_back(subfunctions.size());
        subfunctionsForVariable[var2].push_back(subfunctions.size());       
        subfunctions.push_back(subfunction);
        //cout << subfunction.size() << " " << var1 << " " << var2 << " " << sign << endl;
        
    }
    inFile.close();
};


double SpinGlass::calculateFitness(Individual *solution)
{
    long double res = 0.0;
    
    for (size_t i = 0; i < subfunctions.size(); ++i)
    {
        int cur_value = 1;
        for (int j = 0; j < 2; ++j)
        {
            if (solution->genotype[subfunctions[i][j]] == 1)
                cur_value *= 1;
            else
                cur_value *= -1;
        }
        cur_value *= subfunctions[i][2];
        res += cur_value;
    }
    
    res /= (double)numberOfVariables;
    solution->fitness = res;
    return (double)res;
}

// BEGIN Best-of-Traps

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
    assert(l > 0);
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

    assert(subfunctions.size() > 0);

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

void BestOfTrapsProblem::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    bot = loadBestOfTraps(config->problemInstancePath, numberOfVariables);
}

double BestOfTrapsProblem::calculateFitness(Individual *solution)
{
    int best_fn = 0;
    double result = evaluateBestOfTraps(bot, [&solution](int i) { return solution->genotype[i]; }, best_fn);
    solution->fitness = result;
    return result;
}

double BestOfTrapsProblem::calculateFitnessPartialEvaluations(Individual *solution, Individual */*solutionBefore*/, vector<int> &/*touchedGenes*/, double /*fitnessBefore*/)
{
    // Note: this simply performs a normal evaluation.
    int best_fn = 0;
    double result = evaluateBestOfTraps(bot, [&solution](int i) { return solution->genotype[i]; }, best_fn);
    solution->fitness = result;
    return result;
}

// END Best-of-Traps
// BEGIN Worst-of-Traps
template<typename T>
int worstBestOfTraps( BestOfTraps &bestOfTraps, T&& getBoolAtIndex, int &worst_fn )
{
    int result = std::numeric_limits<int>::max();
    for (size_t fn = 0; fn < bestOfTraps.permutedRandomTraps.size(); ++fn)
    {
        int result_subfn = evaluateConcatenatedPermutedTrap(bestOfTraps.permutedRandomTraps[fn], getBoolAtIndex);
        if (result_subfn < result)
        {
            worst_fn = fn;
            result = result_subfn;
        }
    }
    return result;
}

void WorstOfTrapsProblem::initializeProblem(Config *config, int numberOfVariables_)
{
    numberOfVariables = numberOfVariables_;
    bot = loadBestOfTraps(config->problemInstancePath, numberOfVariables);
}

double WorstOfTrapsProblem::calculateFitness(Individual *solution)
{
    int best_fn = 0;
    double result = worstBestOfTraps(bot, [&solution](int i) { return solution->genotype[i]; }, best_fn);
    solution->fitness = result;
    return result;
}

double WorstOfTrapsProblem::calculateFitnessPartialEvaluations(Individual *solution, Individual */*solutionBefore*/, vector<int> &/*touchedGenes*/, double /*fitnessBefore*/)
{
    // Note: this simply performs a normal evaluation.
    int best_fn = 0;
    double result = evaluateBestOfTraps(bot, [&solution](int i) { return solution->genotype[i]; }, best_fn);
    solution->fitness = result;
    return result;
}
// END Worst-of-Traps
// BEGIN Worst-of-MaxCuts
MaxCutInstance load_maxcut(istream& stream)
{
    size_t l = 0;
    size_t num_edges = 0;
    std::vector<std::tuple<size_t, size_t, long>> edges;
    stream >> l >> num_edges;
    if (stream.fail()) throw invalid_argument("invalid instance - start");
    while (!stream.eof())
    {
        size_t i = 0;
        size_t j = 0;
        long w = 0;
        stream >> i >> j >> w;
        if (!stream.eof() && stream.fail()) throw invalid_argument("invalid instance - edges");
        if (stream.eof()) break;
        // Note: vertices are 1-indexed, so remove 1!
        edges.push_back({i - 1, j - 1, w});
    }
    assert(edges.size() == num_edges);
    return MaxCutInstance { l , edges }; 
}

MaxCutInstance load_maxcut(std::filesystem::path& path)
{
    ifstream file(path);
    std::cout << "Loading " << path << "\n";
    return load_maxcut(file);
}

long evaluate_maxcut(MaxCutInstance &instance, std::vector<char> &genotype)
{
    long v = 0;
    for (auto &[i, j, w] : instance.edges)
    {
        v += w * static_cast<long>(genotype[i] != genotype[j]);
    }
    return v;
}

WorstOfMaxcutProblem::WorstOfMaxcutProblem(std::string instance_str)
{
    // Load instances
    maxcut_instances.clear();
    l = 0;
    size_t pos = 0;
    while (pos < instance_str.size()) {
        size_t next_pos = instance_str.find(';', pos);
        // if not found: use end
        if (next_pos == instance_str.npos) next_pos = instance_str.size();
        std::filesystem::path filename(instance_str.substr(pos, next_pos - pos));
        MaxCutInstance instance = load_maxcut(filename);
        l = std::max(l, instance.l);
        maxcut_instances.push_back(instance);
        pos = next_pos + 1; // + 1 to skip ';'
    }
}

double WorstOfMaxcutProblem::calculateFitnessPartialEvaluations(
    Individual *solution, Individual * /* solutionBefore */, vector<int> &/* touchedGenes */,
    double /* fitnessBefore */) {
  return calculateFitness(solution);
}
double WorstOfMaxcutProblem::calculateFitness(Individual *solution) {
  double f = std::numeric_limits<double>::infinity();
  for (MaxCutInstance &instance : maxcut_instances) {
    double f_m = evaluate_maxcut(instance, solution->genotype);
    if (f_m < f) {
      f = f_m;
    }
  }
  solution->fitness = f;
  return f;
}
void WorstOfMaxcutProblem::initializeProblem(Config * /* config */,
                                             int numberOfVariables_) {
  assert(l == static_cast<size_t>(numberOfVariables_));
};

// END Worst-of-MaxCuts


void PythonFunction::initializeProblem(Config *config, int numberOfVariables_)
{
  numberOfVariables = numberOfVariables_;
  string pwd = filesystem::current_path().string();
  cout << "current_path: " << pwd << endl;
  char moduleName[1000];
  sprintf(moduleName, "import sys; sys.path.insert(0, \"%s/py_src\")", pwd.c_str());  
  cout << moduleName << endl;
  PyRun_SimpleString(moduleName);
  //PyRun_SimpleString ("import sys; print (sys.path)");

  PyObject* module = PyImport_ImportModule("fitnessFunctions");
  if (module == NULL) {cout << "Module import failed!\n";}

  functionClass = PyObject_GetAttrString(module, functionName.c_str());   /* fetch module.class */
  if (functionClass == NULL) {cout << "Class import failed!\n";}

  PyObject *pargs  = Py_BuildValue("(s,s,i,s)", config->folder.c_str(), instancePath.c_str(), config->numberOfVariables, config->alphabet.c_str());
  functionInstance  = PyEval_CallObject(functionClass, pargs);        /* call class(  ) */
  if (functionInstance == NULL) {cout << "Function init failed!\n";}

  fitnessFunction  = PyObject_GetAttrString(functionInstance, "fitness"); /* fetch bound method */
  if (fitnessFunction == NULL) {cout << "Fitness function retrieval failed!\n";}    

  getEvalsFunction  = PyObject_GetAttrString(functionInstance, "nEvals"); /* fetch bound method */
  if (getEvalsFunction == NULL) {cout << "Get evals function retrieval failed!\n";}   
};


double PythonFunction::calculateFitness(Individual *solution)
{
    PyObject *pySolution = PyTuple_New(numberOfVariables);
    for (Py_ssize_t i = 0; i < numberOfVariables; i++)
    {
        PyTuple_SET_ITEM(pySolution, i, PyLong_FromLong((int)solution->genotype[i]));
    }
    PyObject *arglist = Py_BuildValue("(O)", pySolution);
    PyObject *result = PyEval_CallObject(fitnessFunction, arglist);
    if (result == NULL) {cout << "Fitness calculation failed!\n";}

    solution->fitness = PyFloat_AsDouble(result);

    Py_DECREF(result);
    Py_DECREF(arglist);
    Py_DECREF(pySolution);

    return solution->fitness;
}

int PythonFunction::getEvals()
{
    PyObject *arglist = NULL;
    PyObject *result = PyEval_CallObject(getEvalsFunction, arglist);
    if (result == NULL) {cout << "Get evals function failed!\n";}

    int evals = PyLong_AsLong(result);
    //cout << "#evals:" << evals << endl;
    Py_DECREF(result);
    
    return evals;
}
