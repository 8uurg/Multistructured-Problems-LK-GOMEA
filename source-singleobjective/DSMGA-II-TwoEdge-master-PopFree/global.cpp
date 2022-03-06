/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include "myrand.h"
#include "statistics.h"
#include "doublelinkedlistarray.h"
#include "zkey.h"
#include "chromosome.h"
#include "sat.h"

int maxMemory = 0;

bool GHC = true;
bool SELECTION = true;
bool CACHE = false;
bool SHOW_BISECTION = true;

char outputFilename[100];
Chromosome::Function Chromosome::function;
int Chromosome::nfe;
int Chromosome::lsnfe;
int Chromosome::hitnfe;
int Chromosome::maxnfe;
bool Chromosome::hit;
unordered_map<unsigned long, double> Chromosome::cache;

double Chromosome::elitistFitness;
unsigned long *Chromosome::elitist;
string Chromosome::folder;
long long Chromosome::timeLimitMilliseconds;
Problem *Chromosome::problemInstance = NULL;
string Chromosome::instancePath = "";
int Chromosome::k;
int Chromosome::s;
double Chromosome::vtr;
long long Chromosome::startTimestamp;
int Chromosome::generationsWithoutImprovement;
int Chromosome::minimumGOMEAIndex;

ZKey zKey;
MyRand myRand;
BitwiseDistance myBD;
SPINinstance mySpinGlassParams;
NKWAProblem nkwa;
SATinstance mySAT;


void outputErrMsg(const char *errMsg) {
    printf("%s\n", errMsg);
    exit(1);
}

int pow2(int x) {
    return (1 << x);
}

