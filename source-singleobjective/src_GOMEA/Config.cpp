#include "Config.hpp"

const double INF_FITNESS=-1e+308;

void Config::splitString(const string &str, vector<string> &splitted, char delim)
{
    size_t current, previous = 0;
    current = str.find(delim);
    while (current != string::npos)
    {
        splitted.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delim, previous);
    }
    splitted.push_back(str.substr(previous, current - previous));
}

bool Config::isNumber(const string &str)
{
	return !str.empty() && all_of(str.begin(), str.end(), ::isdigit);
}

bool Config::parseCommandLine(int argc, char **argv)
{

  command = "";
  for (size_t i = 0; i < argc; ++i) {
    if (i != 0) {
      command += " ";
    }
    command += string(argv[i]);
  }

  const struct option longopts[] =
  {
	{"help",        no_argument,         0, 'h'},    
  {"FI",          no_argument,         0, 'f'},
  {"partial",     no_argument,         0, 'p'},
  {"analyzeFOS",  no_argument,         0, 'w'},
  {"saveEvals",   no_argument,         0, 's'},  
  {"donorSearch", no_argument,         0, 'd'},    
  {"tournamentSelection", no_argument, 0, 't'}, 
    
  {"approach",    required_argument,   0, 'a'},  
  {"GOM",         required_argument,   0, 'C'},  
  {"maxEvals",    required_argument,   0, 'E'},  
  {"hillClimber", required_argument,   0, 'H'},    
	{"problem",     required_argument,   0, 'P'},  
	{"L",           required_argument,   0, 'L'},  
	{"FOS",         required_argument,   0, 'F'},
	{"seed",        required_argument,   0, 'S'},
	{"alphabet",    required_argument,   0, 'A'},
	{"scheme",      required_argument,   0, 'X'},
	{"instance",    required_argument,   0, 'I'},
	{"vtr",         required_argument,   0, 'V'},
	{"timeLimit",   required_argument,   0, 'T'},
	{"folder",      required_argument,   0, 'O'},
	{"threshold",   required_argument,   0, 'M'},        
	{"elitist_mode", required_argument, 0, 'W'},
  {"orderFOS",    required_argument,   0, 'B'}, 
  {"similarityMeasure", required_argument,   0, 'Z'}, 
  {"populationSize", required_argument,   0, 'Q'}, 
  {"functionName", required_argument,   0, 'N'},
	{"noveltyArchive", required_argument, 0, 'n'},
	{"noveltyK", required_argument, 0, 'k'},
	{"noveltyFullPopulation", required_argument, 0, 'g'},
	{"noveltyUpdate", required_argument, 0, 'u'},
	// {"noveltyFeasQ", required_argument, 0, 'q'},
	{"nfcombine", required_argument, 0, 'q'},
	{"mixingOperator", required_argument, 0, 'm'},
	{"ls", required_argument, 0, 'l'},
	{"fitness_sharing", required_argument, 0, 'z'},
	{"crowding", required_argument, 0, 'c'},
	{"topology", required_argument, 0, 'y'},
	{"relinker", required_argument, 0, 'x'},
	{"multifos", required_argument, 0, 'j'},
	{"distance", required_argument, 0, 'b'},
	{"nisl", required_argument, 0, 'D'},
	{"v", required_argument, 0, 'v'},
	{"reference", required_argument, 0, 'r'},
	{"sampleref", required_argument, 0, 'R'},

    {0,             0,                   0,  0 }
  };


  int c, index;
  while ((c = getopt_long(argc, argv, "c::f::p::w::s::d::t::C::P::F::L::O::T::S::V::A::M::I::X::B::Z::Q::H::E::N::", longopts, &index)) != -1)
  {
  	switch (c)
	{
		case 'v':
			verbosity = atoi(optarg);
			break;
		case 'a':
			approach = atoi(optarg);
			break;
		case 'z':
			{
				// Fitness sharing.
				const string optarg_str = string(optarg);
				vector<string> tmp;
				splitString(optarg_str, tmp, '_');
				fitness_sharing = atoi(tmp[0].c_str());
				if (tmp.size() > 1)
					share_sigma = atof(tmp[1].c_str());
				if (tmp.size() > 2)
					share_alpha = atof(tmp[2].c_str());
				if (tmp.size() > 3)
					share_fitness_filter = atoi(tmp[3].c_str()) > 0;
				if (tmp.size() > 4)
					share_steadystate = atoi(tmp[4].c_str()) > 0;
			}
			break;
		case 'y':
			{
				// Population Topology.
				const string optarg_str = string(optarg);
				vector<string> tmp;
				splitString(optarg_str, tmp, '_');
				populationTopology = atoi(tmp[0].c_str());
				if (tmp.size() > 1)
					topologyMode = atoi(tmp[1].c_str());
				if (tmp.size() > 2)
					topologyParam2 = atoi(tmp[2].c_str());
				if (tmp.size() > 3)
					topologyThreshold = atof(tmp[3].c_str());
			}
			break;
		case 'x':
			{
				// Population Topology.
				const string optarg_str = string(optarg);
				vector<string> tmp;
				splitString(optarg_str, tmp, '_');
				solution_relinker = atoi(tmp[0].c_str());
			}
			break;
		case 'b':
			distance_kind = atoi(optarg);
			break;
		case 'c':
			crowding = atoi(optarg);
			break;
		case 'f':
			// Toggle the use of FI
			// As of writing this message, this would turn it off.
			useForcedImprovements = 1 - useForcedImprovements;
			break;
		case 'l':
			
			break;
		case 'm':
			mixingOperator = atoi(optarg);
			break;
		case 'p':
			usePartialEvaluations = 1;
			break;
		case 'j':
			multiFOS = atoi(optarg);
			break;
		case 'w':
			AnalyzeFOS = 1;
			break;
		case 's':
			saveEvaluations = 1;
			break;
		case 'h':
			printHelp = 1;
			break;
		case 'd':
			donorSearch = 1;
			break;
		case 't':
			tournamentSelection = 1;
			break;
		case 'C':
			conditionalGOM = atoi(optarg);
			break;
		case 'P':
			{
				const string optarg_str = string(optarg);
				if (isNumber(optarg_str)) 	
					problemIndex = atoi(optarg);
				else
				{
					vector<string> tmp;
					splitString(optarg_str, tmp, '_');
					cout << tmp[0] << " " << tmp[1] << " " << tmp[2] << endl;	
				
					problemIndex = atoi(tmp[0].c_str());
					k = atoi(tmp[1].c_str());
					s = atoi(tmp[2].c_str());

					if (tmp.size() > 3)
						fn = atoi(tmp[3].c_str());
					
					cout << problemIndex << endl;
				}
				
			}
			break;
		case 'F':
			FOSIndex = atoi(optarg);
			break;
		case 'H':
			hillClimber = atoi(optarg);
			break;
		case 'L':
			numberOfVariables = atoi(optarg);
			break;
		case 'M':
			MI_threshold = atof(optarg);
			break;
		case 'O':
			folder= string(optarg);
			break;
		case 'T':
			timelimitSeconds = atoi(optarg);
			break;
		case 'V':
			{
				const string optarg_str = string(optarg);
				vector<string> tmp;
				splitString(optarg_str, tmp, '_');
				vtr = atof(tmp[0].c_str());
				if (tmp.size() > 1)
				{
					vtr_n_unique = atoi(tmp[1].c_str());
					// cout << "Looking for " << vtr_n_unique << "unique solutions with fitness " << vtr << "!" << endl;
				}
			}
			break;
		case 'S':
			randomSeed = atoll(optarg);
			break;
		case 'A':
			alphabet = string(optarg);
			break;
		case 'I':
			problemInstancePath = string(optarg);
			break;
		case 'N':
			functionName = string(optarg);
			break;
		case 'X':
			populationScheme = atoi(optarg);
			break;
		case 'Q':
			populationSize = atoi(optarg);
			break;
		case 'B':
			orderFOS = atoi(optarg);
			break;
		case 'W':
			use_elitist_per_population = atoi(optarg);
			// std::cout << "Set elitist mode to " << use_elitist_per_population << "\n";
			break;
		case 'Z':
			similarityMeasure = atoi(optarg);
			break;
		case 'E':
			maxEvaluations = atoll(optarg);
			break;
		case 'D':
			noImprovementStretchLimit = atoi(optarg);
			break;
		case 'r':
		{
			vector<char> reference_in;
			for (size_t i = 0; *(optarg + i) != '\0'; ++i)
			{
				if (*(optarg + i) == '0')
				{
					reference_in.push_back(0);
				} else {
					reference_in.push_back(1);
				}
			}
			reference_solution = reference_in;
		}
			break;
		case 'R':
			sscanf(optarg, "%lf", &p_sample_reference);
			break;

		// case '?':
		// 	if (optopt == 'P' || optopt == 'L' || optopt =='F' || optopt == 'T' || optopt == 'O')
		// 		cerr << "Option -" << optopt << "requires an argument.\n";
		// 	else if (isprint (optopt))
		// 	  cerr << "Unknown option -" << optopt << endl;
		// 	else
		// 		cerr << "Unknown option character \"" << optopt << "\"" << endl;
		// 	exit(0);
		default:
			abort();
	}
  }

  if (populationScheme == 3) //fixed pop size
  {
  	maximumNumberOfGOMEAs = 1;
  	basePopulationSize = populationSize;
  }

  alphabetSize.resize(numberOfVariables);

  if (atoi(alphabet.c_str()))
  	fill(alphabetSize.begin(), alphabetSize.end(), atoi(alphabet.c_str()));
  else
  {
  	ifstream in;
  	in.open(alphabet, ifstream::in);
  	for (int i = 0; i < numberOfVariables; ++i)
  	{
  		in >> alphabetSize[i];
  	}
  	in.close();
  }

  if (printHelp)
  {
  	printUsage();
  	exit(0);
  }
  return 1;
}

void Config::printUsage()
{
  cout << "Usage: GOMEA [-h] [-p] [-v] [-w] [-s] [-h] [-P] [-L] [-F] [-T] [-S]\n";
  cout << "   -h: Prints out this usage information.\n";
  cout << "   -f: Whether to use Forced Improvements.\n";
  cout << "   -p: Whether to use partial evaluations\n";
  cout << "   -w: Enables writing FOS contents to file\n";
  cout << "   -s: Enables saving all evaluations in archive\n";
  cout << endl;
  cout << "    P: Index of optimization problem to be solved (maximization). Default: 0\n";
  cout << "    F: FOS type. Default: 0 (univariate FOS)\n";  
  cout << "    L: Number of variables. Default: 1\n";
  cout << "    A: Alphabet size. Default: 2 (binary optimization)\n";  
  cout << "    C: GOM type. Default: 0 (LT). 1 - conditionalGOM with MI, 2 - conditionalGOM with predefined VIG\n";
  cout << "    O: Folder where GOMEA runs. Default: \"test\"\n";
  cout << "    T: timeLimit in seconds. Default: 1\n";
  cout << "    S: random seed. Default: 42\n";
}

void Config::writeOverviewToFile()
{
	ofstream outFile(folder + "/description.txt", ofstream::out);
	if (outFile.fail())
	{
		cerr << "Problems with opening file " << folder + "/description.txt!\n";
		cerr << "Could not write experiment description to file.\n";
		exit(0);
	}
	printOverview(outFile);
	outFile.close();
}

void Config::printOverview()
{
	printOverview(cout);
}

void Config::printOverview(ostream &sout)
{
  sout << "### Settings ######################################\n";
  sout << "#\n";
  sout << "" << endl;
  sout << "# Variant Mode: ";
  if (approach == 0) {
    sout << "GOMEA (No Modifications)" << endl;
  } else if (approach == 1) {
    sout << "GOMEA (Novelty/Sharing Options)" << endl;
  } else if (approach == 2) {
    sout << "Bit Flip searcher" << endl;
  }

  sout << "# Use Forced Improvements : " << (useForcedImprovements ? "enabled" : "disabled")  << endl;
  sout << "# Use partial evaluations : " << (usePartialEvaluations ? "enabled" : "disabled")  << endl;
  sout << "# Write FOS to file : " << (AnalyzeFOS ? "enabled" : "disabled") << endl;
  sout << "# Save all evaluations : " << (saveEvaluations ? "enabled" : "disabled") << endl;

  string conditionalGOMDesc = "Unconditional";
  if (conditionalGOM == 1)
  	conditionalGOMDesc = "Conditional by MI, MI threshold="+to_string(MI_threshold);
  else if (conditionalGOM == 2)
  	conditionalGOMDesc = "Conditional by VIG";  
  else if (conditionalGOM == 3)
  	conditionalGOMDesc = "Conditional by MI Global, MI percentile="+to_string(MI_threshold);
  else if (conditionalGOM == 4)
  	conditionalGOMDesc = "Conditional by chi^2 test, p_value="+to_string(MI_threshold);
  else if (conditionalGOM == 5)
  	conditionalGOMDesc = "Conditional by mined dependencies";
  else if (conditionalGOM == 10)
  	conditionalGOMDesc = "Restricted+back GOM";
  
  sout << "# conditionalGOM : " << (conditionalGOMDesc) << endl;
  string populationSchemeName = "IMS";
  if (populationScheme == 1)
  	populationSchemeName = "P3-Multiple Insertion Quadratic";
  else if (populationScheme == 2)
  	populationSchemeName = "P3";  
  else if (populationScheme == 3)
  	populationSchemeName = "Fixed Population";  
  else if (populationScheme == 4)
  	populationSchemeName = "P3-Multiple Insertion Linear";

  sout << "# population scheme : " << populationSchemeName << endl;
  if (populationScheme == 3)
  	sout << "# population size : " << populationSize << endl;
  if (hillClimber == 0)
	  sout << "# use hill climber : " << "disabled" << endl;
  else if (hillClimber == 1)
	  sout << "# use hill climber : " << "single" << endl;
  else if (hillClimber == 2)
	  sout << "# use hill climber : " << "multiple" << endl;
  else if (hillClimber == 3)
	  sout << "# use hill climber : " << "Local Search" << endl;

  sout << "# use exhaustive donor search : " << (donorSearch ? "enabled" : "disabled") << endl;
  sout << "# use tournament selection : " << (tournamentSelection ? "enabled" : "disabled") << endl;
  sout << "# re-use offsprings in mixing : " << (reuseOffsprings ? "enabled" : "disabled") << endl;
  sout << "# similarity measure : " << (similarityMeasure ? "normalized MI" : "MI") << endl;
  sout << "# FOS ordering : " << (orderFOS ? "ascending" : "random") << endl;

  sout << "#\n";
  sout << "###################################################\n";
  sout << "#\n";
  sout << "# Problem                      = " << problemName << " " << functionName << endl;
  sout << "# Problem Instance Filename    = " << problemInstancePath << endl;
  sout << "# FOS                          = " << FOSName << endl;
  sout << "# Number of variables          = " << numberOfVariables << endl;
  sout << "# Alphabet size                = ";
  for (int i = 0; i < numberOfVariables; ++i)
  	sout << alphabetSize[i] << " ";
  sout << endl;
  sout << "# Time Limit (seconds)         = " << timelimitSeconds << endl;
  sout << "# #Evals Limit                 = " << maxEvaluations << endl;
  sout << "# VTR                          = " << ((vtr < 1e+308) ? to_string(vtr) : "not set") << endl;
  sout << "# VTR - Number of Unique Sol.  = " << vtr_n_unique << endl;
  sout << "# Random seed                  = " << randomSeed << endl;
  sout << "# Folder                       = " << folder << endl;
  sout << "# Command                      = " << command << endl;
  sout << "#\n";
  sout << "### Settings ######################################\n";
}

void Config::checkOptions()
{
	if (!problemNameByIndex(this, problemName))
	{
		cerr << "No problem with index " << problemIndex << " installed!";
		exit(0);
	}

	// if (!FOSNameByIndex(FOSIndex, FOSName))
	// {
	// 	cerr << "No FOS with index " << FOSIndex << " installed!";
	// 	exit(0);
	// }
}
