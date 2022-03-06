/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/

#include <list>
#include <vector>
#include <algorithm>
#include <iterator>

#include <iostream>
#include "chromosome.h"
#include "dsmga2.h"
#include "fastcounting.h"
#include "statistics.h"

#include <iomanip>
using namespace std;


DSMGA2::DSMGA2 (int n_ell, int n_nInitial, int n_maxGen, int n_maxFe, int fffff) {


    previousFitnessMean = -INF;
    ell = n_ell;
    nCurrent = (n_nInitial/2)*2;  // has to be even


    selectionPressure = 2;
    maxGen = n_maxGen;
    maxFe = n_maxFe;

    graph.init(ell);
    graph_size.init(ell);
     
    bestIndex = -1;
    masks = new list<int>[ell];
    selectionIndex = new int[nCurrent];
    orderN = new int[nCurrent];
    orderELL = new int[ell];
    population = new Chromosome[nCurrent];
    fastCounting = new FastCounting[ell];

    for (int i = 0; i < ell; i++)
        fastCounting[i].init(nCurrent);


    pHash.clear();
    for (int i=0; i<nCurrent; ++i) {
        population[i].initR(ell, nCurrent);
        double f = population[i].getFitness();
        pHash[population[i].getKey()] = f;
    }

    if (GHC) {
        for (int i=0; i < nCurrent; i++)
            population[i].GHC();
    }

    double max = -INF;
    stFitness.reset ();

    for (int i = 0; i < nCurrent; ++i) {
        double fitness = population[i].getFitness();
        //cout << i << " " << fitness << endl;
        if (fitness > max) {
            max = fitness;
            bestIndex = i;
        }
        stFitness.record (fitness);

    }

    //showStatistics ();

}


DSMGA2::~DSMGA2 () {
    delete []masks;
    delete []orderN;
    delete []orderELL;
    delete []selectionIndex;
    delete []population;
    delete []fastCounting;
}



// bool DSMGA2::isSteadyState () {

//     if (stFitness.getNumber () <= 0)
//         return false;

//     if (previousFitnessMean < stFitness.getMean ()) {
//         previousFitnessMean = stFitness.getMean () + EPSILON;
//         return false;
//     }

//     return true;
// }

// bool DSMGA2::converged() {
//     if (stFitness.getMax() == lastMax &&
//         stFitness.getMean() == lastMean &&
//         stFitness.getMin() == lastMin)
//         convergeCount++;
//     else
//         convergeCount = 0;

//     lastMax = stFitness.getMax();
//     lastMean = stFitness.getMean();
//     lastMin = stFitness.getMin();
//     return (convergeCount > 300) ? true : false;
// }

int DSMGA2::doIt (bool output) {
    generation = 0;
    while (!shouldTerminate ()) {
        double fitness_before = Chromosome::elitistFitness;
        oneRun (output);
        double fitness_after = Chromosome::elitistFitness;
        //cout << fitness_before << " " << fitness_after << endl;
        // if (fitness_after <= fitness_before)
        // {
        //   Chromosome::generationsWithoutImprovement++;
        //   //cout << "generations without improvement: " << Chromosome::generationsWithoutImprovement << endl;
        //   if (Chromosome::generationsWithoutImprovement >= 100)
        //     break;
        // }
        // else 
        //     Chromosome::generationsWithoutImprovement = 0;

    }
    return generation;
}


void DSMGA2::oneRun (bool output) {

    if (CACHE)
        Chromosome::cache.clear();

    mixing();


    double max = -INF;
    stFitness.reset ();

    for (int i = 0; i < nCurrent; ++i) {
        double fitness = population[i].getFitness();
        //cout << i << " " << fitness << endl;
        if (fitness > max) {
            max = fitness;
            bestIndex = i;
        }
        stFitness.record (fitness);

    }

    //if (output)
    //    showStatistics ();

    ++generation;
}


bool DSMGA2::shouldTerminate () {
    bool  termination = false;

    //cout << Chromosome::nfe << " " << maxFe << endl;
    //cout << generation << " " << maxGen << endl;
    termination = true;
    for (int i = 1; i < nCurrent; ++i)
    {
        for (int j = 0; j < ell; ++j)
        {
            if (population[i].getVal(j) != population[0].getVal(j))
            {
                termination = false;
                break;
            }
        }
        //cout << population[i].getFitness() << endl;
        if (termination == false)
            break;
    }
    if (termination == true)
        cout <<"CONVERGED!\n";

    // if (maxFe != -1) {
        // if (Chromosome::nfe >= 100000000)
        // {
        //     termination = true;
        //     cout << "max FE reached!\n";
        // }
    //}

    // if (maxGen != -1) {
    //     if (generation > maxGen)
    //     {
    //         termination = true;
    //         cout << "max gen num reached!\n";
    //     }
    // }

    //if (population[0].getMaxFitness() <= stFitness.getMax() )
    //    termination = true;

    // if (stFitness.getMax() - EPSILON <= stFitness.getMean() )
    //     termination = true;
    if (Chromosome::elitistFitness >= population[0].getMaxFitness())
    {
        termination = true;
        cout << "VTR HIT!\n";
        throw customException("vtr");
    }

    return termination;

}


bool DSMGA2::foundOptima () {
    return (stFitness.getMax() > population[0].getMaxFitness());
}


void DSMGA2::showStatistics () {

    // printf ("Gen:%d  Fitness:(Max/Mean/Min):%f/%f/%f #evals:%d \n ",
            // generation, stFitness.getMax (), stFitness.getMean (),
            // stFitness.getMin (), Chromosome::nfe);
    //fflush(NULL);
}



void DSMGA2::buildFastCounting() {

    if (SELECTION) {
        for (int i = 0; i < nCurrent; i++)
            for (int j = 0; j < ell; j++) {
                fastCounting[j].setVal(i, population[selectionIndex[i]].getVal(j));
            }

    } else {
        for (int i = 0; i < nCurrent; i++)
            for (int j = 0; j < ell; j++) {
                fastCounting[j].setVal(i, population[i].getVal(j));
            }
    }

}

int DSMGA2::countOne(int x) const {

    int n = 0;

    for (int i=0; i<fastCounting[0].lengthLong; ++i) {
        unsigned long val = 0;

        val = fastCounting[x].gene[i];

        n += myBD.countOne(val);
    }

    return n;
}


int DSMGA2::countXOR(int x, int y) const {

    int n = 0;

    for (int i=0; i<fastCounting[0].lengthLong; ++i) {
        unsigned long val = 0;


        val = fastCounting[x].gene[i];

        val ^= fastCounting[y].gene[i];

        n += myBD.countOne(val);
    }

    return n;
}

//2016-03-09
// Almost identical to DSMGA2::findClique
// except check 00 or 01 before adding connection
void DSMGA2::findMask(Chromosome& ch, list<int>& result,int startNode){
    result.clear();

    
	DLLA rest(ell);
	genOrderELL();
	for( int i = 0; i < ell; i++){
		if(orderELL[i] == startNode)
			result.push_back(orderELL[i]);
		else
			rest.insert(orderELL[i]);
	}

	double *connection = new double[ell];

	for(DLLA::iterator iter = rest.begin(); iter != rest.end(); iter++){
	    pair<double, double> p = graph(startNode, *iter);
		int i = ch.getVal(startNode);
		int j = ch.getVal(*iter);
		if(i == j)//p00 or p11
			connection[*iter] = p.first;
		else      //p01 or p10
			connection[*iter] = p.second;
	}
   
    while(!rest.isEmpty()){

	    double max = -INF;
		int index = -1;
		for(DLLA::iterator iter = rest.begin(); iter != rest.end(); iter++){
		    if(max < connection[*iter]){
			    max = connection[*iter];
				index = *iter;
			}
		}

		rest.erase(index);
		result.push_back(index);

		for(DLLA::iterator iter = rest.begin(); iter != rest.end(); iter++){
			pair<double, double> p = graph(index, *iter);
			int i = ch.getVal(index);
			int j = ch.getVal(*iter);
			if(i == j)//p00 or p11
				connection[*iter] += p.first;
			else      //p01 or p10
				connection[*iter] += p.second;
		}
	}

	delete []connection;
    // for (auto it=result.begin(); it!= result.end(); ++it)
    //     cout << *it << " ";
    // cout << endl;
    
}

void DSMGA2::restrictedMixing(Chromosome& ch) {
    
    int startNode = myRand.uniformInt(0, ell - 1);    
    
    // for (auto it = mask.begin(); it != mask.end(); ++it)
    //     cout << *it << " ";
    // cout << endl;

    list<int> mask;
	findMask(ch, mask,startNode);
    size_t size = findSize(ch, mask);
   
    list<int> mask_size; 
    findMask_size(ch,mask_size,startNode,size);
    size_t size_original = findSize(ch,mask_size);

    if (size > size_original)
        size = size_original;
    while (mask.size() > size)
        mask.pop_back();
    
    // for (auto it = mask.begin(); it != mask.end(); ++it)
    //     cout << *it << " ";
    // cout << endl;

    bool taken = restrictedMixing(ch, mask);


    EQ = true;
    if (taken) {
    
        genOrderN();
        // cout << " good mask:\n";
        // for (auto it = mask.begin(); it != mask.end(); ++it)
        //     cout << *it << " ";
        // cout << endl;

        for (int i=0; i<nCurrent; ++i) {

            if (EQ)
                backMixingE(ch, mask, population[orderN[i]]);
            else
                backMixing(ch, mask, population[orderN[i]]);
        }
    }

}
void DSMGA2::findMask_size(Chromosome& ch, list<int>& result,int startNode,int bound){
    result.clear();

    
	DLLA rest(ell);

	for( int i = 0; i < ell; i++){
		if(orderELL[i] == startNode)
			result.push_back(orderELL[i]);
		else
			rest.insert(orderELL[i]);
	}

	double *connection = new double[ell];

	for(DLLA::iterator iter = rest.begin(); iter != rest.end(); iter++){
	    pair<double, double> p = graph_size(startNode, *iter);
		int i = ch.getVal(startNode);
		int j = ch.getVal(*iter);
		if(i == j)//p00 or p11
			connection[*iter] = p.first;
		else      //p01 or p10
			connection[*iter] = p.second;
	}
    bound--;
    while(!rest.isEmpty()&&bound>0){
        bound--;
	    double max = -INF;
		int index = -1;
		for(DLLA::iterator iter = rest.begin(); iter != rest.end(); iter++){
		    if(max < connection[*iter]){
			    max = connection[*iter];
				index = *iter;
			}
		}

		rest.erase(index);
		result.push_back(index);

		for(DLLA::iterator iter = rest.begin(); iter != rest.end(); iter++){
			pair<double, double> p = graph_size(index, *iter);
			int i = ch.getVal(index);
			int j = ch.getVal(*iter);
			if(i == j)//p00 or p11
				connection[*iter] += p.first;
			else      //p01 or p10
				connection[*iter] += p.second;
		}
	}

    delete []connection;
  
}

void DSMGA2::backMixing(Chromosome& source, list<int>& mask, Chromosome& des) {

    Chromosome trial(ell, source.populationSize);
    trial = des;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it)
        trial.setVal(*it, source.getVal(*it));

    if (trial.getFitness() > des.getFitness()) {
        pHash.erase(des.getKey());
        pHash[trial.getKey()] = trial.getFitness();
        des = trial;
          
        return;
    }

}

void DSMGA2::backMixingE(Chromosome& source, list<int>& mask, Chromosome& des) {

    Chromosome trial(ell, source.populationSize);
    trial = des;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it)
        trial.setVal(*it, source.getVal(*it));

    // for (auto it = mask.begin(); it != mask.end(); ++it)
    //     cout << *it << " ";
    // cout << endl;

    if (trial.getFitness() > des.getFitness()) {
        //cout << "back f change: " << des.getFitness() << " " << trial.getFitness() << endl;
        // for (auto it = mask.begin(); it != mask.end(); ++it)
        //     cout << *it << " ";
        // cout << endl;

        pHash.erase(des.getKey());
        pHash[trial.getKey()] = trial.getFitness();

        EQ = false;
        des = trial;

return;
    }

    //2016-10-21
    if (trial.getFitness() >= des.getFitness() - EPSILON) {
        pHash.erase(des.getKey());
        pHash[trial.getKey()] = trial.getFitness();

        des = trial;
        return;
    }

}

bool DSMGA2::restrictedMixing(Chromosome& ch, list<int>& mask) {

    bool taken = false;
    size_t lastUB = 0;

    for (size_t ub = 1; ub <= mask.size(); ++ub) {
        

        size_t size = 1;
        Chromosome trial(ell, ch.populationSize);
        trial = ch;
	    
		//2016-03-03
	    vector<int> takenMask;

        for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
            
            //2016-03-03
			takenMask.push_back(*it);
            //cout << *it << ":" << trial.getVal(*it) << "--";
            trial.flip(*it);
            //cout << trial.getVal(*it) <<endl;

            ++size;
            if (size > ub) break;
        }

        //if (isInP(trial)) continue;
        //2016-10-21
        if (isInP(trial)) break;

        if (trial.getFitness() >= ch.getFitness() - EPSILON) {
            //cout << "f change: " << ch.getFitness() << " " << trial.getFitness() << endl;
            pHash.erase(ch.getKey());
            pHash[trial.getKey()] = trial.getFitness();

            taken = true;
            ch = trial;
            //for (int i = 0; i < takenMask.size(); ++i)
            //    cout << takenMask[i] << " ";
            //cout << endl;
        }

        if (taken) {
            lastUB = ub;
            break;
        }
    }

    if (lastUB != 0) {
        while (mask.size() > lastUB)
            mask.pop_back();
    }

    return taken;

}

size_t DSMGA2::findSize(Chromosome& ch, list<int>& mask) const {

    DLLA candidate(nCurrent);
    for (int i=0; i<nCurrent; ++i)
        candidate.insert(i);

    size_t size = 0;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {

        int allele = ch.getVal(*it);

        for (DLLA::iterator it2 = candidate.begin(); it2 != candidate.end(); ++it2) {
            if (population[*it2].getVal(*it) == allele)
                candidate.erase(*it2);

            if (candidate.isEmpty())
                break;
        }

        if (candidate.isEmpty())
            break;

        ++size;
    }

    return size;


}

size_t DSMGA2::findSize(Chromosome& ch, list<int>& mask, Chromosome& ch2) const {

    size_t size = 0;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
        if (ch.getVal(*it) == ch2.getVal(*it)) break;
        ++size;
    }
    return size;
}

void DSMGA2::mixing() {

    if (SELECTION)
        selection();

    //* really learn model
    buildFastCounting();
    buildGraph();
    buildGraph_sizecheck();
    //for (int i=0; i<ell; ++i)
    //    findClique(i, masks[i]); // replaced by findMask in restrictedMixing

    int repeat = (ell>50)? ell/50: 1;

    for (int k=0; k<repeat; ++k) {

        genOrderN();
        for (int i=0; i<nCurrent; ++i) {
            restrictedMixing(population[orderN[i]]);
            if (Chromosome::hit) break;
        }
        if (Chromosome::hit) break;
    }


}

inline bool DSMGA2::isInP(const Chromosome& ch) const {

    unordered_map<unsigned long, double>::const_iterator it = pHash.find(ch.getKey());
    return (it != pHash.end());
}

inline void DSMGA2::genOrderN() {
    myRand.uniformArray(orderN, nCurrent, 0, nCurrent-1);
}

inline void DSMGA2::genOrderELL() {
    myRand.uniformArray(orderELL, ell, 0, ell-1);
}

void DSMGA2::buildGraph() {

    int *one = new int [ell];
    for (int i=0; i<ell; ++i) {
        one[i] = countOne(i);
    }

    for (int i=0; i<ell; ++i) {

        for (int j=i+1; j<ell; ++j) {

            int n00, n01, n10, n11;
            int nX =  countXOR(i, j);

            n11 = (one[i]+one[j]-nX)/2;
            n10 = one[i] - n11;
            n01 = one[j] - n11;
            n00 = nCurrent - n01 - n10 - n11;

            double p00 = (double)n00/(double)nCurrent;
            double p01 = (double)n01/(double)nCurrent;
            double p10 = (double)n10/(double)nCurrent;
            double p11 = (double)n11/(double)nCurrent;
            double p1_ = p10 + p11;
            double p0_ = p00 + p01;
            double p_0 = p00 + p10;
            double p_1 = p01 + p11;

            double linkage = computeMI(p00,p01,p10,p11);
            
            //2016-04-08_computeMI_entropy
            double linkage00 = 0.0, linkage01 = 0.0;
            if (p00 > EPSILON)
                linkage00 += p00*log(p00/p_0/p0_);
            if (p11 > EPSILON)
                linkage00 += p11*log(p11/p_1/p1_);
            if (p01 > EPSILON)
                linkage01 += p01*log(p01/p0_/p_1);
            if (p10 > EPSILON)
                linkage01 += p10*log(p10/p1_/p_0);
            
            //cout << i << " " << j << " " << linkage << endl;
            if(Chromosome::nfe < 0){
                pair<double, double> p(linkage, linkage);
                graph.write(i, j, p);
            }
            else{
                pair<double, double> p(linkage00, linkage01);
                graph.write(i, j, p);
            }
				
        }
    }


    delete []one;

}
void DSMGA2::buildGraph_sizecheck() {

    int *one = new int [ell];
    for (int i=0; i<ell; ++i) {
        one[i] = countOne(i);
    }

    for (int i=0; i<ell; ++i) {

        for (int j=i+1; j<ell; ++j) {

            int n00, n01, n10, n11;
            int nX =  countXOR(i, j);

            n11 = (one[i]+one[j]-nX)/2;
            n10 = one[i] - n11;
            n01 = one[j] - n11;
            n00 = nCurrent - n01 - n10 - n11;

            double p00 = (double)n00/(double)nCurrent;
            double p01 = (double)n01/(double)nCurrent;
            double p10 = (double)n10/(double)nCurrent;
            double p11 = (double)n11/(double)nCurrent;
            double p1_ = p10 + p11;
            double p0_ = p00 + p01;
            double p_0 = p00 + p10;
            double p_1 = p01 + p11;

            double linkage = computeMI(p00,p01,p10,p11);
            
            //2016-04-08_computeMI_entropy
            double linkage00 = 0.0, linkage01 = 0.0;
            if (p00 > EPSILON)
                linkage00 += p00*log(p00/p_0/p0_);
            if (p11 > EPSILON)
                linkage00 += p11*log(p11/p_1/p1_);
            if (p01 > EPSILON)
                linkage01 += p01*log(p01/p0_/p_1);
            if (p10 > EPSILON)
                linkage01 += p10*log(p10/p1_/p_0);
        
	
            pair<double, double> p(linkage, linkage);
            graph_size.write(i, j, p);
			
        }
    }


    delete []one;

}


// from 1 to ell, pick by max edge
void DSMGA2::findClique(int startNode, list<int>& result) {
    
   }

    
    double DSMGA2::computeMI(double a00, double a01, double a10, double a11) const {

    double p0 = a00 + a01;
    double q0 = a00 + a10;
    double p1 = 1-p0;
    double q1 = 1-q0;

    double join = 0.0;
    if (a00 > EPSILON)
        join += a00*log(a00);
    if (a01 > EPSILON)
        join += a01*log(a01);
    if (a10 > EPSILON)
        join += a10*log(a10);
    if (a11 > EPSILON)
        join += a11*log(a11);

    double p = 0.0;
    if (p0 > EPSILON)
        p += p0*log(p0);
    if (p1 > EPSILON)
        p += p1*log(p1);


    double q = 0.0;
    if (q0 > EPSILON)
        q += q0*log(q0);
    if (q1 > EPSILON)
        q += q1*log(q1);

    return -p-q+join;

}


void DSMGA2::selection () {
    tournamentSelection ();
}


// tournamentSelection without replacement
void DSMGA2::tournamentSelection () {
    int i, j;

    int randArray[selectionPressure * nCurrent];

    for (i = 0; i < selectionPressure; i++)
        myRand.uniformArray (randArray + (i * nCurrent), nCurrent, 0, nCurrent - 1);

    for (i = 0; i < nCurrent; i++) {

        int winner = 0;
        double winnerFitness = -INF;

        for (j = 0; j < selectionPressure; j++) {
            int challenger = randArray[selectionPressure * i + j];
            double challengerFitness = population[challenger].getFitness();

            if (challengerFitness > winnerFitness) {
                winner = challenger;
                winnerFitness = challengerFitness;
            }

        }
        selectionIndex[i] = winner;
    }
}
