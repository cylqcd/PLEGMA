#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <set>
#include <optional>
#include <PLEGMA_Hprobing.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq"};

int main(int argc, char **argv)
{
	initializeOptions(argc, argv, true, listOpt);
	//================ Add your options in this between initializeOptions and initializePLEGMA ================//
	bool accumFlag = true;
	int NdumpStep = 1;
	std::string loopsPrefix="./";
	int k_probing = 0;
  int hadamLow=0;
  int Nhadam = (k_probing>0) ? 2*std::pow(2,N_DIMS*(k_probing-1)) : 1;
  int hadamHgh = Nhadam;
  bool spinColorDil = false;
  auto add_options = [&](Options& options) {
		options.set("accum-loops", "Accumulate loops over the stochastic source vectors", verbosity, accumFlag);
		options.set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);
		options.set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
		options.set("k-probing", "Hierarchical probing, with distance D=2**k (Options:0,1,2,3,...) (0 means No probing)",verbosity,k_probing);
		options.set("hadamard-low", "From which Hadamard vector to start (Options:[0,max))", verbosity, hadamLow);
		options.set("hadamard-high", "Up to which Hadamard vector to stop (Options: 0>= , <=max) (default max)", verbosity, hadamHgh);
		options.set("spin-color-dil", "Whether we want spin color dilution",verbosity,spinColorDil);
	};
	add_options(*HGC_options);
	int Nsc = spinColorDil ? N_SPINS*N_COLS : 1;
	if((k_probing>0) && (hadamLow<0 || hadamHgh<0)) PLEGMA_error("Negative values for number of Hadamard vector not allowed");
	if((k_probing>0) && (hadamLow>hadamHgh))  PLEGMA_error("hadamard-high should be > hadamard-low");
	if((k_probing>0) && (hadamHgh>Nhadam)) PLEGMA_error("hadamard-high should be <= from max number of Hadamard vectors");
   	//=========================================================================================================//
	initializePLEGMA();

	// Reading from Lime file and loading to device
	PLEGMA_Gauge<double> gauge;
	gauge.readFile(latfile, LIME_FORMAT);
	gauge.calculatePlaq();

	// Loading to QUDA and computing plaquette also there
	initGaugeQuda(gauge, true);
	plaqQuda();

	applyBoundaryConditions(gauge,true);

	if(!accumFlag) NdumpStep = 1;
	if(accumFlag && (NdumpStep<1)) PLEGMA_error("dump-step should be >= 1");

	// ensuring mu negative
	if(mu>0) mu*=-1.;
  QUDA_solver *solver = new QUDA_solver(mu);
	QudaInvertParam inv_params = solver->getInvParams();

	PLEGMA_Vector<double> *sourceDil = nullptr;
  sourceDil = new PLEGMA_Vector<double>(BOTH);

	PLEGMA_FT<double> *ft[2]={nullptr,nullptr};
	ft[0] = new PLEGMA_FT<double>(maxQsq, 3);

	bool oneDLoops = false;
  bool twoDLoops = false;
	PLEGMA_QLoops<double> qloops_std(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);
	PLEGMA_Vector<double> phi;
	PLEGMA_Vector<double> source(DEVICE);
	source.randInit(rng_seed);

	PLEGMA_Hprobing *hprob = nullptr;

  std::size_t foundPos = latfile.find("conf.");
  if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
  std::string confID = latfile.substr(foundPos+5,latfile.length());

  hprob = new PLEGMA_Hprobing(k_probing);

  std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
  for(int isrc = 0; isrc < numSourcePositions; isrc++){ // numSourcePosition is actually stochastic source position but anyway
    source.stochastic_Z(2);
    for(int ih = hadamLow; ih < hadamHgh; ih++){
      	for(int isc = 0; isc < Nsc; isc++){
          	if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}
			if(spinColorDil && k_probing>0){ sourceDil->applyHpropColoring4D(*sourceDil,*hprob,ih,indDof);}
			else if(!spinColorDil && k_probing>0){ sourceDil->applyHpropColoring4D(source,*hprob,ih,indDof);}
			if(spinColorDil || k_probing>0) solver->solve(phi,*sourceDil); else solver->solve(phi,source);
			// for convention reasons for quark loops we put the normalization factors of the fields later in the analysis
			phi.scale(1./(2.*inv_params.kappa));
			qloops_std.oneEnd_trick(phi,*sourceDil,-1.,true);
      	}
    }
      
    if((isrc+1)%NdumpStep == 0){
      qloops_std.dumpLoops(ft, loopsPrefix + "/trace", confID, corr_file_format, isrc);
    }
    if(!accumFlag){ // In case we do not accumulate we clear the buffers
      qloops_std.clearAccumBuffs();
    }
  }

	delete hprob;
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	delete solver;
	finalize();
	return 0;
}