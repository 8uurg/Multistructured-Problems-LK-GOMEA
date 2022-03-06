/***************************************************************************
 *   Copyright (C) 2015 by TEIL                                            *
 ***************************************************************************/

#include <cstdio>
#include <cstring>
#include "spin.h"
#include "chromosome.h"
#include "nk-wa.h"
#include "sat.h"
#include <sys/time.h>

#define TRAP_K 5

Chromosome::Chromosome () {
    length = 0;
    lengthLong = 0;
    this->populationSize = 0;
    gene = NULL;
    evaluated = false;
}

Chromosome::Chromosome (int n_length, int populationSize) {
    gene = NULL;
    init (n_length, populationSize);
}


Chromosome::~Chromosome () {
    if (gene != NULL) delete []gene;
}

void Chromosome::init (int _length, int populationSize) {
    length = _length;
    lengthLong = quotientLong(length)+1;
    this->populationSize = populationSize;

    if (gene != NULL)
        delete []gene;

    gene = new unsigned long [lengthLong];
    gene[lengthLong-1] = 0;

    evaluated = false;
}

void Chromosome::init0 (int _length, int populationSize) {
    length = _length;
    lengthLong = quotientLong(length)+1;
    this->populationSize = populationSize;

    if (gene != NULL)
        delete []gene;

    gene = new unsigned long [lengthLong];

    for (int i=0; i<lengthLong; ++i)
        gene[i] = 0;

    key = 0;
    evaluated = false;
}

void Chromosome::initR (int _length, int populationSize) {
    length = _length;
    lengthLong = quotientLong(length)+1;
    this->populationSize = populationSize;

    if (gene != NULL)
        delete []gene;

    gene = new unsigned long [lengthLong];
    gene[lengthLong-1] = 0;

    key = 0;
    for (int i=0; i<length; ++i) {

        int val = myRand.flip();
        setValF(i, val);
        if (val == 1)
            key ^= zKey[i];
    }

    evaluated = false;
}

double Chromosome::getFitness () {

    checkTimeLimit(startTimestamp, timeLimitMilliseconds);

    if (evaluated)
        return fitness;
    else {
        fitness = evaluate();
        if (fitness > elitistFitness)
        {
            elitistFitness = fitness;
            for (int i = 0; i < length; ++i)
                elitist[i] = getVal(i);
            //cout << elitistFitness << endl;
            long long cur_time = getMilliSecondsRunningSinceTimeStamp(startTimestamp);
            bool is_key_solution = fitness >= getMaxFitness();
            
            writeElitistSolutionToFile(folder, nfe, cur_time, elitist, length, elitistFitness, is_key_solution, populationSize);
        }
        //cout << elitistFitness << " " << getMaxFitness() << endl;

        if (!hit && fitness >= getMaxFitness()) {
            hit = true;
            hitnfe = nfe;
        }
        return fitness;
    }
}

bool Chromosome::isEvaluated () const {
    return evaluated;
}

bool Chromosome::hasSeen() const {

    unordered_map<unsigned long, double>::iterator it = cache.find(key);
    if (it != cache.end())
        return true;
    return false;
}

double Chromosome::evaluate () {


    if (CACHE)
        if (hasSeen()) {
            evaluated = true;
            return cache[key];
        }

    ++nfe;

    if (nfe > Chromosome::maxnfe)
    {
        cout << "EVALUATION BUDGET REACHED!" << endl;
        throw customException("evaluations");
    }
    // if (nfe%100000 == 0)
        // cout <<"nfe: " << nfe << endl;

    evaluated = true;
    double accum = 0.0;

    accum = problemInstance->calculateFitness(gene);
    // switch (function) {
    //     case ONEMAX:
    //         accum = oneMax();
    //         break;
    //     case MKTRAP:
    //         accum = mkTrap(1, 0.8);
    //         break;
    //     case CYCTRAP:
    //         accum = cycTrap(1, 0.8);
    //         break;
    //     case FTRAP:
    //         accum = fTrap();
    //         break;
    //     case SPINGLASS:
    //         accum = spinGlass();
    //         break;
    //     case NK:
    //         accum = nkFitness();
    //         break;
    //     case SAT:
    //         accum = satFitness();
    //         break;
    //     default:
    //         accum = mkTrap(1, 0.8);
    //         break;
    // }

    if (CACHE)
        cache[key]=accum;

    return accum;

}



double
Chromosome::spinGlass () const {

    int *x = new int[length];
    double result;

    for (int i=0; i<length; i++)
        if (getVal(i) == 1)
            x[i] = 1;
        else
            x[i] = -1;

    result = evaluateSPIN(x, &mySpinGlassParams);

    delete []x;

    return result;
}

double Chromosome::nkFitness() const {
    char *x = new char[length];

    for ( int i = 0; i < length; ++i) {
        x[i] = (char) getVal(i);
    }

    double result = evaluateNKProblem(x, &nkwa);
    //double result = evaluateNKWAProblem(x, &nkwa);
    delete []x;
    return result;
}

// OneMax
double Chromosome::oneMax () const {

    double result = 0;

    for (int i = 0; i < length; ++i)
        result += getVal(i);

    return result;
}

bool Chromosome::operator== (const Chromosome& c) const {
    if (length != c.length)
        return false;

    for (int i=0; i<lengthLong; i++)
        if (gene[i] != c.gene[i])
            return false;

    return true;
}

Chromosome& Chromosome::operator= (const Chromosome& c) {

    if (length != c.length) {
        length = c.length;
        init (length, c.populationSize);
    }

    evaluated = c.evaluated;
    fitness = c.fitness;
    lengthLong = c.lengthLong;
    key = c.key;

    memcpy(gene, c.gene, sizeof(long) * lengthLong);

    return *this;
}

double Chromosome::trap (int unitary, double fHigh, double fLow, int trapK) const {
    if (unitary > trapK)
        return 0;

    if (unitary == trapK)
        return fHigh;
    else
        return fLow - unitary * fLow / (trapK-1);
}


double Chromosome::fTrap() const {

    double result = 0.0;

    for (int i=0; i<length/6; ++i) {
        int u=0;
        for (int j=0; j<6; ++j)
            u += getVal(i*6+j);

        if (u==0)
            result += 1.0;
        else if (u==1)
            result += 0.0;
        else if (u==2)
            result += 0.4;
        else if (u==3)
            result += 0.8;
        else if (u==4)
            result += 0.4;
        else if (u==5)
            result += 0.0;
        else // u == 6
            result += 1.0;
    }

    return result;
}

double Chromosome::cycTrap(double fHigh, double fLow) const {
    int i, j;
    int u;
    int TRAP_M = length / (TRAP_K-1);
    if (length % (TRAP_K-1) != 0)
        outputErrMsg ("TRAP_k doesn't divide length for Cyclic Setting");
    double result = 0;
    for (i = 0; i < TRAP_M; i++) {
        u = 0;
        int idx = i * TRAP_K - i;
        for (j = 0; j < TRAP_K; j++) {
            int pos = idx + j;
            if (pos == length)
                pos = 0;
            else if (pos > length)
                outputErrMsg ("CYCLIC BUG");
            //
            u += getVal(pos);
        }
        result += trap (u, fHigh, fLow, TRAP_K);
    }
    return result;
}



double Chromosome::mkTrap (double fHigh, double fLow) const {
    int i, j;
    int u;

    int TRAP_M = length / TRAP_K;

    if (length % TRAP_K != 0)
        outputErrMsg ("TRAP_K doesn't divide length");

    double result = 0;

    for (i = 0; i < TRAP_M; i++) {
        u = 0;
        for (j = 0; j < TRAP_K; j++)
            u += getVal(i * TRAP_K + j);

        result += trap (u, fHigh, fLow, TRAP_K);
    }

    return result;
}


int Chromosome::getLength () const {
    return length;
}

double Chromosome::getMaxFitness () const {

    //cout << vtr << endl;
    return vtr;

    // double maxF;

    // switch (function) {
    //     case ONEMAX:
    //         maxF = length;
    //         break;
    //     case MKTRAP:
    //         maxF = length/TRAP_K;
    //         break;
    //     case FTRAP:
    //         maxF = length/6;
    //         break;
    //     case CYCTRAP:
    //         maxF =  length/(TRAP_K - 1);
    //         break;
    //     case SPINGLASS:
    //         maxF = mySpinGlassParams.opt;
    //         break;
    //     case NK:
    //         maxF = nkwa.maxF;
    //         break;
    //     case SAT:
    //         maxF = 0;
    //         break;
    //     default:
    //         // Never converge
    //         maxF = INF;
    // }

    // return maxF - EPSILON;

}

// contribute to lsnfe
bool Chromosome::tryFlipping(int index) {

    //int oldNFE = nfe;

    double oldF = getFitness();
    flip(index);

    //2016-10-21
    if (getFitness() - EPSILON <= oldF) {
    //if (getFitness() <= oldF) {
        flip(index);
        evaluated = true;
        fitness = oldF;

        //lsnfe += nfe - oldNFE;
        //nfe = oldNFE;

        return false;
    } else {

        //lsnfe += nfe - oldNFE;
        //nfe = oldNFE;

        return true;
    }


}

bool Chromosome::GHC() {

    int* order = new int [length];
    myRand.uniformArray(order, length, 0, length-1);

    bool flag = false;
    for (int i=0; i<length; ++i) {
        if (tryFlipping(order[i])) flag = true;
    }

    delete []order;
    return flag;

}

double Chromosome::satFitness() const {
    int *x = new int[length];

    for ( int i = 0; i < length; ++i) {
        x[i] = getVal(i);
    }

    double result = evaluateSAT(x, &mySAT);
    delete []x;
    return result;
}

//////////////////////////////////////////////////////////////////////////////////

void writeElitistSolutionToFile(string &folder, long long numberOfEvaluations, long long time, unsigned long *solution, int numberOfVariables, double fitness, bool is_key_solution, long populationSize)
{
    ofstream outFile(folder + "/elitists.txt", ofstream::app);
    if (outFile.fail())
    {
        cerr << "Problems with opening file " << folder + "/elitists.txt!\n";
        exit(0);
    }

    outFile << 
        (int)numberOfEvaluations << " " << 
        time << " " << 
        fixed << setprecision(6) << fitness << " " <<
        is_key_solution << " " <<
        populationSize << " ";
    for (size_t i = 0; i < numberOfVariables; ++i)
        outFile << solution[i];
    outFile << endl;

    outFile.close();
}

void checkTimeLimit(long long startTimestamp, long long timeLimitMilliseconds)
{
  if (getMilliSecondsRunningSinceTimeStamp(startTimestamp) > timeLimitMilliseconds)
  {
    cout << "TIME LIMIT REACHED!" << endl;
    throw customException("time");
  }
}

long long getCurrentTimeStampInMilliSeconds()
{
  struct timeval tv;
  long long result;

  gettimeofday (&tv, NULL);
  result = (tv.tv_sec * 1000) + (tv.tv_usec / 1000);

  return  result;
}

long long getMilliSecondsRunningSinceTimeStamp (long long startTimestamp)
{
  long long timestamp_now, difference;

  timestamp_now = getCurrentTimeStampInMilliSeconds();

  difference = timestamp_now-startTimestamp;

  return difference;
}
