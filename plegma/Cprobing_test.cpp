#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <set>
#include <optional>
#include <PLEGMA_Cprobing.h>

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
					    "nsrc", "maxQsq"};

int main(int argc, char **argv)
{
	PLEGMA_printf(">>> About to call initializeOptions()\n");
	initializeOptions(argc, argv, true, listOpt);
	PLEGMA_printf(">>> Returned from initializeOptions()\n");
	//================ Add your options in this between initializeOptions and initializePLEGMA ================//
	bool accumFlag = true;
	int NdumpStep = 1;
	std::string loopsPrefix="./";
	int coloring_distance = 1;
	int probing_dimension = 4;
	bool c_probing = false;
	bool spinColorDil = false;
  	auto add_options = [&](Options& options) {
		options.set("accum-loops", "Accumulate loops over the stochastic source vectors", verbosity, accumFlag);
		options.set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);
  		options.set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
		options.set("coloring-distance", "Coloring distance for classical probing", verbosity, coloring_distance);
		options.set("probing-dimension", "Probing dimension for classical probing", verbosity, probing_dimension);
		options.set("c-probing", "Whether to use classical probing", verbosity, c_probing);
		options.set("spin-color-dil", "Whether we want spin color dilution",verbosity,spinColorDil);
	};
	add_options(*HGC_options);	 
	int Nsc = spinColorDil ? N_SPINS*N_COLS : 1;
	PLEGMA_printf("\n Number of spin-colors used: %d\n",Nsc);
	std::string tag = "/trace";
	if (c_probing) {
		tag += "_cD" + std::to_string(coloring_distance);
	}
	if (spinColorDil) {
		tag += "_dil";
	} else {
		tag += "_nodil";
	}
   	//=========================================================================================================//
	PLEGMA_printf(">>> About to call initializePLEGMA()\n");
	initializePLEGMA();
	PLEGMA_printf(">>> Returned from initializePLEGMA()\n");

	PLEGMA_printf("Loading gauge!\n");
	// Reading from Lime file and loading to device
	PLEGMA_Gauge<double> gauge;
	gauge.readFile(latfile, LIME_FORMAT);
	gauge.calculatePlaq();

	PLEGMA_printf("Init gauge!\n");
	// Loading to QUDA and computing plaquette also there
	initGaugeQuda(gauge, true);
	plaqQuda();

	applyBoundaryConditions(gauge,true);

	if(!accumFlag) NdumpStep = 1;
	if(accumFlag && (NdumpStep<1)) PLEGMA_error("dump-step should be >= 1");

	PLEGMA_printf("Starting solver init!\n");
	// ensuring mu negative
	if(mu>0) mu*=-1.;
  	QUDA_solver *solver = new QUDA_solver(mu);
	QudaInvertParam inv_params = solver->getInvParams();
	PLEGMA_printf("Finished solver init!\n");

	PLEGMA_printf("Source init!\n");
	PLEGMA_Vector<double> *sourceDil = nullptr;
  	if(c_probing) sourceDil = new PLEGMA_Vector<double>(BOTH);

	PLEGMA_printf("FT init!\n");
	PLEGMA_FT<double> *ft[2]={nullptr,nullptr};
	ft[0] = new PLEGMA_FT<double>(maxQsq, 3);

	bool oneDLoops = false;
  	bool twoDLoops = false;
	PLEGMA_QLoops<double> qloops_std(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);
	PLEGMA_Vector<double> phi;
	PLEGMA_Vector<double> source(DEVICE);
	source.randInit(rng_seed);

	PLEGMA_printf("Start probing!\n");
	PLEGMA_Cprobing *cprob = nullptr;
  	if(c_probing){
		std::size_t foundPos = latfile.find("conf.");
		if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
		std::string confID = latfile.substr(foundPos+5,latfile.length());

		cprob = new PLEGMA_Cprobing(coloring_distance, probing_dimension);

		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank==0) {
			PLEGMA_printf("Local colors array is:\n");
			for(int i=0; i<HGC_localVolume; i++)
				PLEGMA_printf("%d ", cprob->H_localColors()[i]);
		}

		int Nc = cprob->get_Ncol();
		std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
		PLEGMA_printf("\n Number of spin-colors used: %d\n",Nsc);
		for(int isrc = 0; isrc < numSourcePositions; isrc++){
			source.stochastic_Z(2);
			for(int ic = 1; ic < Nc+1; ic++){
				for(int isc = 0; isc < Nsc; isc++){
					if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}
					if(spinColorDil && c_probing){ sourceDil->applyCprobColoring(*sourceDil,*cprob,ic,indDof);}
	  				else if(!spinColorDil && c_probing){ sourceDil->applyCprobColoring(source,*cprob,ic,indDof);}
					// sourceDil->unload();

					// Print out the probing vector for verification
					// ----------------------------------------------------------------------------------------------------
					// if (rank == 0) {
					// 	const double *H = sourceDil->H_elem();
					// 	if (!H) { PLEGMA_printf("ERROR: sourceDil host ptr is null\n"); MPI_Abort(MPI_COMM_WORLD,1); }

					// 	const int NdofPerSite = 12;         // adjust if different
					// 	const double eps = 1e-12;

					// 	const int maxSites = 33; // or HGC_localVolume to dump all
					// 	const int nSitesToPrint = std::min(static_cast<int>(HGC_localVolume), maxSites);

					// 	PLEGMA_printf("Probing vector (real parts only), grouped by site (layout: [N_dof][Lx][Ly][Lz][Lt])\n");
					// 	for (int s = 0; s < nSitesToPrint; ++s) {
					// 		bool allZero = true;
					// 		char line[2048]; line[0] = '\0';

					// 		for (int d : indDof) {
					// 			size_t off = (static_cast<size_t>(d) * HGC_localVolume + s) * 2; // Re at [off], Im at [off+1]
					// 			double re = H[off];
					// 			allZero &= (std::fabs(re) < eps);

					// 			char buf[16];
					// 			// print +1 / -1 / 0 depending on sign (since we expect ±1 probing)
					// 			snprintf(buf, sizeof(buf), "%+d ", (std::fabs(re) < eps) ? 0 : (re > 0 ? 1 : -1));
					// 			strncat(line, buf, sizeof(line) - strlen(line) - 1);
					// 		}

					// 		int color = cprob->H_localColors()[s];
					// 		PLEGMA_printf("site %6d  color %4d  [%s]  %s\n",
					// 					s, color, line, allZero ? "ALL_ZERO" : "NONZERO");
					// 	}

					// 	if (nSitesToPrint < (int)HGC_localVolume)
					// 		PLEGMA_printf("... (printed %d of %zu sites)\n", nSitesToPrint, HGC_localVolume);
					// }
					// ----------------------------------------------------------------------------------------------------

					// Verify that the probing vector has nonzeros only at the expected color sites
					// ----------------------------------------------------------------------------------------------------
					// double sum_nonzero = 0.0, sum_total = 0.0;

					// for (int i = 0; i < HGC_localVolume; i++) {
					// 	// Assume each site value has 2 components (Re, Im)
					// 	double re = sourceDil->H_elem()[2*i];
					// 	double im = sourceDil->H_elem()[2*i+1];
					// 	double norm = sqrt(re*re + im*im);

					// 	int color = cprob->H_localColors()[i];
					// 	bool shouldBeZero = (color != ic);  // +1 since colors are 1..Nc

					// 	if (shouldBeZero && norm > 1e-12) {
					// 		PLEGMA_printf("❌ Nonzero at site %d, color=%d, but masking for %d\n", i, color, ic);
					// 	}
					// 	if (!shouldBeZero && norm < 1e-12) {
					// 		PLEGMA_printf("❌ Zero at site %d, color=%d, expected nonzero (color %d)\n", i, color, ic);
					// 	}

					// 	if (!shouldBeZero) sum_nonzero += norm;
					// 	sum_total += norm;
					// }

					// PLEGMA_printf("Color %d test done: nonzero sum = %.6e (of total %.6e)\n",
					// ic, sum_nonzero, sum_total);
					// ----------------------------------------------------------------------------------------------------

					if(spinColorDil || c_probing){
						TIME(solver->solve(phi,*sourceDil)); 
					} else {
						TIME(solver->solve(phi,source));
					}
					phi.scale(1./(2.*inv_params.kappa));
					if(spinColorDil || c_probing){
						TIME(qloops_std.oneEnd_trick(phi,*sourceDil,-1.,true));
					} else {
						TIME(qloops_std.oneEnd_trick(phi,source,-1.,true));
					}
				}
			}
		
			if((isrc+1)%NdumpStep == 0){
				qloops_std.dumpLoops(ft, loopsPrefix + tag, confID, corr_file_format, isrc);
			}
			if(!accumFlag){ // In case we do not accumulate we clear the buffers
				qloops_std.clearAccumBuffs();
			}
		}
	}

	if(c_probing) delete cprob;
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	delete solver;
	finalize();
	return 0;
}