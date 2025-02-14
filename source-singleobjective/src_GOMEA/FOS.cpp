//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "FOS.hpp"

bool FOSNameByIndex(size_t FOSIndex, string &FOSName)
{
    switch (FOSIndex)
    {
        case 1: FOSName = "Linkage Tree"; break;
        case 2: FOSName = "Fixed Linkage Tree"; break;  
        case 3: FOSName = "Filtered Linkage Tree"; break;
        case 4: FOSName = "Random Tree"; break;
        // case 4: FOSName = "ILS"; break;
        // case 5: FOSName = "Dynamic LTFOS"; break;

        case 10: FOSName = "Fixed Problem FOS"; break;
            
        default: return false; break;
    }
    return true;
}

void createFOSInstance(size_t FOSIndex, FOS **FOSInstance, size_t numberOfVariables, vector<int> &alphabetSize, int similarityMeasure, Problem *problem)
{
    switch (FOSIndex)
    {
        case 10: *FOSInstance = new ProblemFOS(numberOfVariables, alphabetSize, problem); break;
        case 1: *FOSInstance = new LTFOS(numberOfVariables, alphabetSize, similarityMeasure); break;
        case 3: *FOSInstance = new LTFOS(numberOfVariables, alphabetSize, similarityMeasure, true); break;
        case 4: *FOSInstance = new LTFOS(numberOfVariables, alphabetSize, 2); break;
        
        default:
        {
            cerr << "No FOS for index " << FOSIndex << " is installed." << endl;
            exit(0);
        }
        break;
    }
}

void FOS::writeToFileFOS(string folder, int populationIndex, int generation)
{
    ofstream outFile;
    outFile.open(folder + "/fos/" + to_string(populationIndex) + "_" + to_string(generation) + ".txt");
    outFile << "FOS size = " << FOSStructure.size() << endl;
    for (size_t i = 0; i < FOSStructure.size(); ++i)
    {
        outFile << "[ ";
        for (size_t j = 0; j < FOSStructure[i].size(); ++j)
            outFile << FOSStructure[i][j] << " "; 
        outFile << "]" << endl;
    }
    outFile.close();
}

void FOS::setCountersToZero()
{
    improvementCounters.resize(FOSStructure.size());
    usageCounters.resize(FOSStructure.size());
    for (int i = 0; i < FOSStructure.size(); ++i)
    {
        improvementCounters[i] = 0;
        usageCounters[i] = 0;
    }
}

void FOS::writeFOSStatistics(string folder, int populationIndex, int generation)
{
    ofstream outFile;
    outFile.open(folder + "/fos/statistics_" + to_string(populationIndex) + "_" + to_string(generation) + ".txt");
    outFile << "FOS_element improvement_counter usage_counter" << endl;
    for (size_t i = 0; i < FOSStructure.size(); ++i)
    {
        for (size_t j = 0; j < FOSStructure[i].size(); ++j)
            outFile << FOSStructure[i][j] << "_"; 
        outFile << " " << improvementCounters[i] << " " << usageCounters[i] << endl;
    }
    outFile.close();
}

void FOS::shuffleFOS(vector<int> &indices, mt19937 *rng)
{
    indices.resize(FOSSize());
    iota(indices.begin(), indices.end(), 0);   
    shuffle(indices.begin(), indices.end(), *rng);
}

void FOS::sortFOSAscendingOrder(vector<int> &indices)
{
    vector<pair<int, int> > fos_goodness;
    indices.resize(FOSSize());

    for (size_t i = 0; i < FOSSize(); i++)
        fos_goodness.push_back(make_pair(FOSElementSize(i), i));
    
    sort(fos_goodness.begin(), fos_goodness.end(), [](const pair<double, int>& lhs, const pair<double, int>& rhs)
    {
        return lhs.first < rhs.first;
    });

    for (size_t i = 0; i < FOSSize(); i++)
        indices[i] = fos_goodness[i].second;
}

void FOS::sortFOSDescendingOrder(vector<int> &indices)
{
    vector<pair<int, int> > fos_goodness;
    indices.resize(FOSSize());

    for (size_t i = 0; i < FOSSize(); i++)
        fos_goodness.push_back(make_pair(FOSElementSize(i), i));
    
    sort(fos_goodness.begin(), fos_goodness.end(), [](const pair<double, int>& lhs, const pair<double, int>& rhs)
    {
        return lhs.first > rhs.first;
    });

    for (size_t i = 0; i < FOSSize(); i++)
        indices[i] = fos_goodness[i].second;
}

void FOS::orderFOS(int orderingType, vector<int> &indices, mt19937 *rng)
{
    if (orderingType == 0)
        shuffleFOS(indices, rng);
    else if (orderingType == 1)
        sortFOSAscendingOrder(indices);
    // else if (orderingType == 2)
    //     sortFOSDescendingOrder(indices);       
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

ProblemFOS::ProblemFOS(size_t numberOfVariables_, vector<int> &alphabetSize_, Problem *problem): FOS(numberOfVariables_, alphabetSize_)
{
    FOSStructure = problem->getProblemFOS();
    usageCounters.resize(FOSStructure.size());
    improvementCounters.resize(FOSStructure.size());
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

LTFOS::LTFOS(size_t numberOfVariables_, vector<int> &alphabetSize_, int similarityMeasure_, bool filtered_): FOS(numberOfVariables_, alphabetSize_)
{
    similarityMeasure = similarityMeasure_;
    filtered = filtered_;
}

void LTFOS::prepareMatrices()
{
    MI_Matrix.resize(numberOfVariables);        
    S_Matrix.resize(numberOfVariables);

    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        MI_Matrix[i].resize(numberOfVariables);         
        S_Matrix[i].resize(numberOfVariables);
    }
}

void LTFOS::learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG, mt19937 *rng)
{
    FOSStructure.clear();
    vector<int> mpmFOSMap;
    vector<int> mpmFOSMapNew;

    prepareMatrices();
    
    /* Compute Mutual Information matrix */
    if (similarityMeasure == 0) // MI
        computeMIMatrix(population);
    else if (similarityMeasure == 1) // normalized MI
        computeNMIMatrix(population);
    else if (similarityMeasure == 2) // Random Similarity (Random Tree)
        computeRandomMatrix(rng);
    
    #ifdef DEBUG
        for (int i = 0; i < numberOfVariables; ++i)
        {
            for (int j = 0; j < numberOfVariables; ++j)
                cout << MI_Matrix[i][j] << " ";
            cout << endl;
        }
    #endif

    /* Initialize MPM to the univariate factorization */
    vector <int> order(numberOfVariables);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), *rng);

    vector< vector<int> > mpm(numberOfVariables);
    vector< vector<int> > mpmNew(numberOfVariables);
    
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        mpm[i].push_back(order[i]);  
    }

    /* Initialize LT to the initial MPM */
    int FOSLength = 2 * numberOfVariables - 1;
    FOSStructure.resize(FOSLength);

    vector<int> useFOSElement(FOSStructure.size(), true);

    int FOSsIndex = 0;
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        FOSStructure[i] = mpm[i];
        mpmFOSMap.push_back(i);
        FOSsIndex++;
    }

    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        for(size_t j = 0; j < numberOfVariables; j++ )
            S_Matrix[i][j] = MI_Matrix[mpm[i][0]][mpm[j][0]];//((*rng)()%10000)/10000.0

        S_Matrix[i][i] = 0;
    }

    vector<int> NN_chain;
    NN_chain.resize(numberOfVariables+2);
    size_t NN_chain_length = 0;
    bool done = false;
    while (!done)
    {
        if (NN_chain_length == 0)
        {
            NN_chain[NN_chain_length] = (*rng)() % mpm.size();
            //std::cout << NN_chain[NN_chain_length] << " | " << mpm.size() << std::endl;
      
            NN_chain_length++;
        }

        while (NN_chain_length < 3)
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length-1], mpm);
            NN_chain_length++;
        }

        while (NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1])
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length-1], mpm);
            if( ((S_Matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length]] == S_Matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
            
            NN_chain_length++;
            if (NN_chain_length > numberOfVariables)
                break;
        }

        size_t r0 = NN_chain[NN_chain_length-2];
        size_t r1 = NN_chain[NN_chain_length-1];
        if (S_Matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]] >= 1-(1e-6))
        {
            // for (int j = 0; j < mpm[r0].size(); ++j)
            //     cout << mpm[r0][j] << " ";
            // cout << "1:"<<endl;

            // for (int j = 0; j < mpm[r1].size(); ++j)
            //     cout << mpm[r1][j] << " ";
            // cout << "2:"<<endl;
            // cout << endl;
            // cout << mpmFOSMap[r0] << " " <<FOSStructure[mpmFOSMap[r0]].size() << endl;
            // for (int j = 0; j < FOSStructure[mpmFOSMap[r0]].size(); ++j)
            //     cout << FOSStructure[mpmFOSMap[r0]][j] << " ";
            // cout << endl;
            // for (int j = 0; j < FOSStructure[mpmFOSMap[r1]].size(); ++j)
            //     cout << FOSStructure[mpmFOSMap[r1]][j] << " ";
            // cout << endl;
            
            useFOSElement[mpmFOSMap[r0]] = false;
            useFOSElement[mpmFOSMap[r1]] = false;            
        }

        if (r0 > r1)
        {
            int rswap = r0;
            r0 = r1;
            r1 = rswap;
        }
        NN_chain_length -= 3;


        if (r1 < mpm.size()) 
        {
            vector<int> indices(mpm[r0].size() + mpm[r1].size());

            size_t i = 0;
            for (size_t j = 0; j < mpm[r0].size(); j++)
            {
                indices[i] = mpm[r0][j];
                i++;
            }

            for (size_t j = 0; j < mpm[r1].size(); j++)
            {
                indices[i] = mpm[r1][j];
                i++;
            }

            FOSStructure[FOSsIndex] = indices;
            FOSsIndex++;

            double mul0 = (double)mpm[r0].size() / (double)(mpm[r0].size() + mpm[r1].size());
            double mul1 = (double)mpm[r1].size() / (double)(mpm[r0].size() + mpm[r1].size());
            for (size_t i = 0; i < mpm.size(); i++)
            {
                if ((i != r0) && (i != r1))
                {
                    S_Matrix[i][r0] = mul0 * S_Matrix[i][r0] + mul1 * S_Matrix[i][r1];
                    S_Matrix[r0][i] = S_Matrix[i][r0];
                }
            }
    
            mpmNew.resize(mpm.size() - 1);
            mpmFOSMapNew.resize(mpmFOSMap.size()-1);
            for (size_t i = 0; i < mpmNew.size(); i++)
            {
                mpmNew[i] = mpm[i];
                mpmFOSMapNew[i] = mpmFOSMap[i];
            }

            mpmNew[r0] = indices;
            mpmFOSMapNew[r0] = FOSsIndex-1;

            if (r1 < mpm.size() - 1)
            {
                mpmNew[r1] = mpm[mpm.size() - 1];
                mpmFOSMapNew[r1] = mpmFOSMap[mpm.size() - 1];

                for (int i = 0; i < r1; i++)
                {
                  S_Matrix[i][r1] = S_Matrix[i][mpm.size() - 1];
                  S_Matrix[r1][i] = S_Matrix[i][r1];
                }

                for (int j = r1 + 1; j < mpmNew.size(); j++)
                {
                  S_Matrix[r1][j] = S_Matrix[j][mpm.size() - 1];
                  S_Matrix[j][r1] = S_Matrix[r1][j];
                }
            }

            for (i = 0; i < NN_chain_length; i++)
            {
                if (NN_chain[i] == mpm.size() - 1)
                {
                    NN_chain[i] = r1;
                    break;
                }
            }

            mpm = mpmNew;
            mpmFOSMap = mpmFOSMapNew;

            if (mpm.size() == 1)
                done = true;
        }
  }

  if (filtered)
  {
    for (int i = 0; i < useFOSElement.size(); ++i)
    {
        if (!useFOSElement[i])
        {
            // cout << "filtered out:\n";
            // for (int j = 0 ; j < FOSStructure[i].size(); ++j)
            //     cout << FOSStructure[i][j] << " ";
            // cout << endl;
            FOSStructure[i].clear();
        }
        else
        {
            // for (int j = 0 ; j < FOSStructure[i].size(); ++j)
            //     cout << FOSStructure[i][j] << " ";
            // cout << endl;
            
        }
    }
  }

  int size = FOSStructure.size();

  for (int i = 0; i < size; ++i)
  {
    if (FOSStructure[i].size()==0)
        continue;
}
  //   double sum = 0;
  //   for (int j =0; j < FOSStructure[i].size(); ++j)
  //   {
  //       sum += alphabetSize[FOSStructure[i][j]];
  //   }
  //   sum /= FOSStructure[i].size();
  //   sum /= 2;
  //   int repeats = floor(sum);

  //   cout << sum << " " << "repeat:" << repeats << endl;
  //   for (int j = 0; j < repeats; ++j)
  //       FOSStructure.push_back(FOSStructure[i]);
  // }
}

void LTFOS::shrinkMemoryUsage()
{
    MI_Matrix.resize(0);
    MI_Matrix.shrink_to_fit();
    S_Matrix.resize(0);
    S_Matrix.shrink_to_fit();
}

/**
 * Determines nearest neighbour according to similarity values.
 */
int LTFOS::determineNearestNeighbour(int index, vector<vector< int> > &mpm)
{
    int result = 0;

    if (result == index)
        result++;

    for (size_t i = 1; i < mpm.size(); i++)
    {
        //std::cout << index << " " << i << " " << result << " " << S_Matrix[index][i] << " " << S_Matrix[index][result] << " " << mpm[i].size() << " " << mpm[result].size() << std::endl;

        if (i != index)
        {
            if ((S_Matrix[index][i] > S_Matrix[index][result]) || ((S_Matrix[index][i] == S_Matrix[index][result]) && (mpm[i].size() < mpm[result].size())))
            {
                result = i;
                //std::cout << (int)(S_Matrix[index][i] > S_Matrix[index][result])<< " result=i" << std::endl;
            }
        }
    }
    //std::cout << "nearest-neighbor " << result << endl;
    return result;
}

void LTFOS::computeMIMatrix(vector<Individual*> &population)
{
    size_t factorSize;
    double p;
    
    /* Compute joint entropy matrix */
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            vector<size_t> indices{i, j};
            vector<double> factorProbabilities;
            estimateParametersForSingleBinaryMarginal(population, indices, factorSize, factorProbabilities);

            MI_Matrix[i][j] = 0.0;
            for(size_t k = 0; k < factorSize; k++)
            {
                p = factorProbabilities[k];
                if (p > 0)
                    MI_Matrix[i][j] += -p * log2(p);
            }
            MI_Matrix[j][i] = MI_Matrix[i][j];
        }

        vector<size_t> indices{i};
        vector<double> factorProbabilities;
        estimateParametersForSingleBinaryMarginal(population, indices, factorSize, factorProbabilities);

        MI_Matrix[i][i] = 0.0;
        for (size_t k = 0; k < factorSize; k++)
        {
            p = factorProbabilities[k];
            if (p > 0)
                MI_Matrix[i][i] += -p * log2(p);
        }

    }

    /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y) */
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            MI_Matrix[i][j] = MI_Matrix[i][i] + MI_Matrix[j][j] - MI_Matrix[i][j];
            MI_Matrix[j][i] = MI_Matrix[i][j];
        }
    }

    // for (int i = 0; i < numberOfVariables; ++i)
    // {
    //  for (int j = 0; j < numberOfVariables; ++j)
    //      cout << MI_Matrix[i][j] << " ";
    //  cout << endl;
    // }
}

void LTFOS::computeNMIMatrix(vector<Individual*> &population)
{
    double p;
    
    /* Compute joint entropy matrix */
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            vector<double> factorProbabilities_joint;
            vector<double> factorProbabilities_i;
            vector<double> factorProbabilities_j;
            size_t factorSize_joint, factorSize_i, factorSize_j;

            vector<size_t> indices_joint{i, j};
            estimateParametersForSingleBinaryMarginal(population, indices_joint, factorSize_joint, factorProbabilities_joint);
            
            vector<size_t> indices_i{i};
            estimateParametersForSingleBinaryMarginal(population, indices_i, factorSize_i, factorProbabilities_i);
            
            vector<size_t> indices_j{j};
            estimateParametersForSingleBinaryMarginal(population, indices_j, factorSize_j, factorProbabilities_j);

            MI_Matrix[i][j] = 0.0;
            
            double separate = 0.0, joint = 0.0;

            for(size_t k = 0; k < factorSize_joint; k++)
            {
                p = factorProbabilities_joint[k];
                //cout << i << " " << j << " " << p << endl;
                if (p > 0)
                    joint += (-p * log2(p));
            }

            for(size_t k = 0; k < factorSize_i; k++)
            {
                p = factorProbabilities_i[k];
                if (p > 0)
                    separate += (-p * log2(p));
            }

            for(size_t k = 0; k < factorSize_j; k++)
            {
                p = factorProbabilities_j[k];
                if (p > 0)
                    separate += (-p * log2(p));
            }
            //cout << separate << " " << joint << endl;
            MI_Matrix[i][j] = 0.0;
            if (joint)
                MI_Matrix[i][j] = separate / joint - 1;
            MI_Matrix[j][i] = MI_Matrix[i][j];

        }

    }

    /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y) */
    // for (int i = 0; i < numberOfVariables; ++i)
    // {
    //  for (int j = 0; j < numberOfVariables; ++j)
    //      cout << MI_Matrix[i][j] << " ";
    //  cout << endl;
    // }
}

void LTFOS::computeRandomMatrix(mt19937 *rng)
{
    uniform_real_distribution unif(0.0, 1.0);
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            double r = unif(*rng);
            MI_Matrix[i][j] = r;
            MI_Matrix[j][i] = r; 
        }
    }
}


void LTFOS::writeMIMatrixToFile(string folder, int populationIndex, int generation)
{
    ofstream outFile;
    outFile.open(folder + "/fos/MI_" + to_string(populationIndex) + "_" + to_string(generation) + ".txt");
    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        for (size_t j = 0; j < numberOfVariables; ++j)
        {       
            outFile << MI_Matrix[i][j] << " "; 
        }
        outFile << endl;
    }
    outFile.close();
}
/**
 * Estimates the cumulative probability distribution of a
 * single binary marginal.
 */
void LTFOS::estimateParametersForSingleBinaryMarginal(vector<Individual*> &population, vector<size_t> &indices, size_t &factorSize, vector<double> &result)
{
    size_t numberOfIndices = indices.size();

    factorSize = 1;
    for (int i = 0; i < numberOfIndices; ++i)
        factorSize *= alphabetSize[indices[i]];

    result.resize(factorSize);
    fill(result.begin(), result.end(), 0.0);

    for (size_t i = 0; i < population.size(); i++)
    {
        int index = 0;
        int power = 1;
        for (int j = numberOfIndices-1; j >= 0; j--)
        {
            int var = indices[j];
            //cout << "var:" << var << " " << power << " " << +population[i]->genotype[var] << endl;
            index += (int)population[i]->genotype[var] * power;
            power *= alphabetSize[var];
        }

        result[index] += 1.0;
        //cout << "index:" << index << " " << factorSize << endl;
    }   
    for (size_t i = 0; i < factorSize; i++)
        result[i] /= (double)population.size();
}

void LTFOS::buildGraph(double thresholdValue, mt19937 *rng)
{
    vector<vector<int> > adjacencyMatrix(numberOfVariables);
    for (int i = 0; i < numberOfVariables; ++i)
    {
        adjacencyMatrix[i].resize(numberOfVariables);
        fill(adjacencyMatrix[i].begin(), adjacencyMatrix[i].end(), 0);
    }

    graph.clear();
    graph.resize(numberOfVariables);

    for (int i = 0; i < numberOfVariables; ++i)
    {
        vector<pair<double, int> > MI_for_variable(numberOfVariables);
        double sum = 0.0;
        for (int j = 0; j < numberOfVariables; ++j)
        {
            MI_for_variable[j] = make_pair(MI_Matrix[i][j], j);
            if (j == i)
                MI_for_variable[j].first = 0.0;
            sum += MI_for_variable[j].first;
        }

        if (sum == 0.0)
            continue;
        //cout << endl;
        //MI_for_variable[i].first = 0.0;
        
        sort(MI_for_variable.begin(), MI_for_variable.end(), [&](const pair<double, int> l, const pair<double,int> r){return l.first > r.first;});
        
        // random ordering in case of MI scores ties
        int l = 0;
        for (int j = 0; j < numberOfVariables; ++j)
        {
            if (MI_for_variable[j].first != MI_for_variable[l].first)
            {
                shuffle(MI_for_variable.begin()+l, MI_for_variable.begin()+j, *rng);
                l = j;
            }
        }
        shuffle(MI_for_variable.begin()+l, MI_for_variable.end(), *rng); // the last chunk

        double cur_sum = 0;
        double max_mi = MI_for_variable[0].first;
        //cout << thresholdValue;
        int cnt = 0;
        for (int j = 0; j < numberOfVariables; ++j)
        {
            if (i == j)
                continue;

            if (MI_for_variable[j].first == 0.0)
                break;

            if (thresholdValue < 1)
            {
                if ((double)MI_for_variable[j].first / (double)max_mi < thresholdValue)
                    break;
            }
            else
            {
                if (cnt == (int)thresholdValue)
                    break;
            }

            if (adjacencyMatrix[i][MI_for_variable[j].second] == 0)
                graph[i].push_back(MI_for_variable[j].second);

            //if (adjacencyMatrix[MI_for_variable[j].second][i] == 0)
            //  graph[MI_for_variable[j].second].push_back(i);

            adjacencyMatrix[i][MI_for_variable[j].second] = 1;
            //adjacencyMatrix[MI_for_variable[j].second][i] = 1;

            cur_sum += MI_for_variable[j].first;
            cnt++;
        }
        //cout << endl;
    }
    
    // for (int i = 0; i < numberOfVariables; ++i)
    // {
    //  for (int j = 0; j < numberOfVariables; ++j)
    //  {
    //      if (adjacencyMatrix[i][j] == 1 && adjacencyMatrix[j][i] == 1)
    //      {
    //          graph[i].push_back(j);
    //          graph[j].push_back(i);
    //      }
    //  }
    // }
    // for (int i = 0; i < numberOfVariables; ++i)
    // {
    //  cout << i << ": ";
    //  for (int j = 0; j < graph[i].size(); ++j)
    //      cout << graph[i][j] << " ";
    //  cout << endl;
    // }
}

void LTFOS::buildGraphGlobal(double thresholdValue)
{
    
    vector<double> flattened_MI;
    for (int i = 0; i < numberOfVariables; ++i)
    {
        for (int j = i+1; j < numberOfVariables; ++j)
        {
            flattened_MI.push_back(MI_Matrix[i][j]);
        }
    }

    sort(flattened_MI.begin(), flattened_MI.end());
    int index = ceil(thresholdValue * flattened_MI.size());
    double cutoffValue = thresholdValue;
    cout << cutoffValue << endl;

    vector<vector<int> > adjacencyMatrix(numberOfVariables);
    for (int i = 0; i < numberOfVariables; ++i)
    {
        adjacencyMatrix[i].resize(numberOfVariables);
        fill(adjacencyMatrix[i].begin(), adjacencyMatrix[i].end(), 0);
    }

    graph.clear();
    graph.resize(numberOfVariables);


    for (int i = 0; i < numberOfVariables; ++i)
    {
        for (int j = 0; j < numberOfVariables; ++j)
        {
            if (MI_Matrix[i][j] >= cutoffValue && MI_Matrix[i][j] > 0 && i != j)
            {
                if (adjacencyMatrix[i][j] == 0)
                    graph[i].push_back(j);

                if (adjacencyMatrix[j][i] == 0)
                    graph[j].push_back(i);

                adjacencyMatrix[i][j] = 1;
                adjacencyMatrix[j][i] = 1;
            }
        }
    }
        
    //for (int i = 0; i < numberOfVariables; ++i)
    //  cout << graph[i].size() << endl;
    // {
    //  cout << i << ": ";
    //  for (int j = 0; j < graph[i].size(); ++j)
    //      cout << graph[i][j] << " ";
    //  cout << endl;
    // }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////


