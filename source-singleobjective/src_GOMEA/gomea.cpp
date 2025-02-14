//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include <gomea.hpp>

double GOMEA::readVTR()
{
  string filename = config->folder + "/vtr.txt";
  ifstream inFile;
  inFile.open(filename);

  string vtr_str;
  inFile >> vtr_str;
  double vtr = stod(vtr_str);

  inFile.close();

  return vtr;
}

