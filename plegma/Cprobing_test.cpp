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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};

int main(int argc, char **argv)
{
	initializeOptions(argc, argv, true, listOpt);
	//================ Add your options in this between initializeOptions and initializePLEGMA ================//
	bool c_probing = false;
	int coloring_distance = 0;
	int probing_dimension = 4;
  	auto add_options = [&](Options& options) {
		options.set("coloring_distance", "Coloring distance for classical probing", verbosity, coloring_distance);
		options.set("probing_dimension", "Probing dimension for classical probing", verbosity, probing_dimension);
		options.set("c-probing", "Whether to use classical probing", verbosity, c_probing);
	};
	add_options(*HGC_options);
   	//=========================================================================================================//
	initializePLEGMA();

	PLEGMA_Cprobing *cprob = nullptr;
  	if(c_probing==true){
		cprob = new PLEGMA_Cprobing(coloring_distance, probing_dimension);

		PLEGMA_printf("Number of colors for classical probing is %d\n",cprob->get_Ncol());
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank==0) {
			PLEGMA_printf("Local colors array is:\n");
			for(int i=0; i<HGC_localVolume; i++)
				PLEGMA_printf("%d ", cprob->H_localColors()[i]);
		}
	}

	if(c_probing==true) delete cprob;
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	//finalize();
	return 0;
}