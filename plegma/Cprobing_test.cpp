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
	int coloring_distance = 1;
	int probing_dimension = 4;
  	auto add_options = [&](Options& options) {
		options.set("coloring_distance", "Coloring distance for classical probing", verbosity, coloring_distance);
		options.set("probing_dimension", "Probing dimension for classical probing", verbosity, probing_dimension);
		options.set("c-probing", "Whether to use classical probing", verbosity, c_probing);
	};
	add_options(*HGC_options);
   	//=========================================================================================================//
	initializePLEGMA();

	PLEGMA_Vector<double> *sourceDil = nullptr;
  	if(c_probing) sourceDil = new PLEGMA_Vector<double>(BOTH);

	PLEGMA_Vector<double> source(DEVICE);
	source.randInit(rng_seed);
	source.stochastic_Z(2);

	PLEGMA_Cprobing *cprob = nullptr;
  	if(c_probing){
		cprob = new PLEGMA_Cprobing(coloring_distance, probing_dimension);

		PLEGMA_printf("Number of colors for classical probing is %d\n",cprob->get_Ncol());
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank==0) {
			PLEGMA_printf("Local colors array is:\n");
			for(int i=0; i<HGC_localVolume; i++)
				PLEGMA_printf("%d ", cprob->H_localColors()[i]);
		}


		int Nc = cprob->get_Ncol();
		std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
		for(int ic = 1; ic < Nc+1; ic++){
			sourceDil->applyCprobColoring(source,*cprob,ic,indDof);
			sourceDil->unload();

			if (rank == 0) {
				const double *H = sourceDil->H_elem();
				if (!H) { PLEGMA_printf("ERROR: sourceDil host ptr is null\n"); MPI_Abort(MPI_COMM_WORLD,1); }

				const int NdofPerSite = 12;         // adjust if different
				const double eps = 1e-12;

				const int maxSites = 33; // or HGC_localVolume to dump all
				const int nSitesToPrint = std::min(static_cast<int>(HGC_localVolume), maxSites);

				PLEGMA_printf("Probing vector (real parts only), grouped by site (layout: [N_dof][Lx][Ly][Lz][Lt])\n");
				for (int s = 0; s < nSitesToPrint; ++s) {
					bool allZero = true;
					char line[2048]; line[0] = '\0';

					for (int d : indDof) {
						size_t off = (static_cast<size_t>(d) * HGC_localVolume + s) * 2; // Re at [off], Im at [off+1]
						double re = H[off];
						allZero &= (std::fabs(re) < eps);

						char buf[16];
						// print +1 / -1 / 0 depending on sign (since we expect ±1 probing)
						snprintf(buf, sizeof(buf), "%+d ", (std::fabs(re) < eps) ? 0 : (re > 0 ? 1 : -1));
						strncat(line, buf, sizeof(line) - strlen(line) - 1);
					}

					int color = cprob->H_localColors()[s];
					PLEGMA_printf("site %6d  color %4d  [%s]  %s\n",
								s, color, line, allZero ? "ALL_ZERO" : "NONZERO");
				}

				if (nSitesToPrint < (int)HGC_localVolume)
					PLEGMA_printf("... (printed %d of %zu sites)\n", nSitesToPrint, HGC_localVolume);
			}



			
			double sum_nonzero = 0.0, sum_total = 0.0;

			for (int i = 0; i < HGC_localVolume; i++) {
				// Assume each site value has 2 components (Re, Im)
				double re = sourceDil->H_elem()[2*i];
				double im = sourceDil->H_elem()[2*i+1];
				double norm = sqrt(re*re + im*im);

				int color = cprob->H_localColors()[i];
				bool shouldBeZero = (color != ic);  // +1 since colors are 1..Nc

				if (shouldBeZero && norm > 1e-12) {
					PLEGMA_printf("❌ Nonzero at site %d, color=%d, but masking for %d\n", i, color, ic);
				}
				if (!shouldBeZero && norm < 1e-12) {
					PLEGMA_printf("❌ Zero at site %d, color=%d, expected nonzero (color %d)\n", i, color, ic);
				}

				if (!shouldBeZero) sum_nonzero += norm;
				sum_total += norm;
			}

			PLEGMA_printf("Color %d test done: nonzero sum = %.6e (of total %.6e)\n",
			ic, sum_nonzero, sum_total);
		}
	}

	if(c_probing) delete cprob;
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	//finalize();
	return 0;
}