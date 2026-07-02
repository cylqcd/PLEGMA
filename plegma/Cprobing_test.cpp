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
	std::string sigma_conf_path = "";
	int start_src = 0;
  	auto add_options = [&](Options& options) {
		options.set("accum-loops", "Accumulate loops over the stochastic source vectors", verbosity, accumFlag);
		options.set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);
  		options.set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
		options.set("coloring-distance", "Coloring distance for classical probing", verbosity, coloring_distance);
		options.set("probing-dimension", "Probing dimension for classical probing", verbosity, probing_dimension);
		options.set("c-probing", "Whether to use classical probing", verbosity, c_probing);
		options.set("spin-color-dil", "Whether we want spin color dilution",verbosity, spinColorDil);
		options.set("sigma-config-path", "Path to the configuration file for the sigma factors", verbosity, sigma_conf_path);
		options.set("start-src", "Starting index for the stochastic sources (inclusive)", verbosity, start_src);
	};
	add_options(*HGC_options);	 
	int Nsc = spinColorDil ? N_SPINS*N_COLS : 1;
	PLEGMA_printf("\n Number of spin-colors used: %d\n",Nsc);
	std::string tag = "/traceC";
	std::string options_tag = "";
	if (c_probing) {
		options_tag += "_cD" + std::to_string(coloring_distance);
	}
	if (spinColorDil) {
		options_tag += "_dil";
	} else {
		options_tag += "_nodil";
	}
	options_tag += "_nsrc" + std::to_string(numSourcePositions);
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

	std::size_t foundPos = latfile.find("conf.");
	if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
	std::string confID = latfile.substr(foundPos+5,latfile.length());

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
  	if(c_probing || spinColorDil) sourceDil = new PLEGMA_Vector<double>(BOTH);

	PLEGMA_printf("FT init!\n");
	PLEGMA_FT<double> *ft[2]={nullptr,nullptr};
	ft[0] = new PLEGMA_FT<double>(maxQsq, 3);

	bool oneDLoops = false;
  	bool twoDLoops = false;
	PLEGMA_QLoops<double> *qloops_std = nullptr;
	qloops_std = new PLEGMA_QLoops<double>(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);

	PLEGMA_Vector<double> phi;
	PLEGMA_Vector<double> source(DEVICE);
	int rng_seed = 42;
	source.randInit(rng_seed);

	PLEGMA_printf("Start probing!\n");
	PLEGMA_Cprobing *cprob = nullptr;
  	if(c_probing) cprob = new PLEGMA_Cprobing(coloring_distance, probing_dimension, sigma_conf_path);

	// int rank;
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// if(rank==0) {
	// 	PLEGMA_printf("Local colors array is:\n");
	// 	for(int i=0; i<HGC_localVolume; i++)
	// 		PLEGMA_printf("%d ", cprob->H_localColors()[i]);
	// }

	// --- PROBE COLLECTION SETUP ---
	// int M_collect = 8;                       // number of probes to collect for orthogonality test
	// int Collected = 0;                        // how many probes we've collected so far (global counter per rank)
	// int NdofPerSite = 12;                     // change if different (spin*color)
	// int nSitesLocal = HGC_localVolume;        // local lattice sites
	// int nComplexLocal = nSitesLocal * NdofPerSite;
	// int nReImLocal = 2 * nComplexLocal;       // interleaved real/imag entries per rank

	// store local copies of the first M_collect probe vectors (host doubles)
	// std::vector<std::vector<double>> local_collected; 
	// local_collected.reserve(M_collect);

	int Nc = c_probing ? cprob->get_Ncol() : 1;
	// int N_total = Nc * Nsc;
	// PLEGMA_printf("DEBUG: hierarchical total_per_source = %d  (N_probes_h=%d Nsc_h=%d)\n", Nc * Nsc, Nc, Nsc);
	std::complex<double> scale_val = std::complex<double>(-1.0, 0.0);
	std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
	PLEGMA_printf("\n Number of spin-colors used: %d\n",Nsc);
	for(int isrc = start_src; isrc < numSourcePositions; isrc++){
		PLEGMA_printf("\n ### Calculations for source-position %d begin now ###\n\n", isrc);
		source.stochastic_Z(2);
		for(int ic = 1; ic < Nc+1; ic++){
			for(int isc = 0; isc < Nsc; isc++){
				if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}
				if(spinColorDil && c_probing){ sourceDil->applyCprobColoring(*sourceDil,*cprob,ic,indDof);}
				else if(!spinColorDil && c_probing){ sourceDil->applyCprobColoring(source,*cprob,ic,indDof);}
				// sourceDil->unload();

				// --- COPY THIS PROBE INTO LOCAL_STORAGE (only for first M_collect probes) ---
				// if(Collected < M_collect){
				// 	// ensure host pointer is available; if using device pointer you may need to call unload()
				// 	const double *host_ptr = nullptr;
				// 	if(spinColorDil || c_probing){
				// 		// sourceDil should have host copy accessor; if not, call sourceDil->unload() / sourceDil->loadHost() as needed
				// 		host_ptr = sourceDil->H_elem();
				// 	} else {
				// 		host_ptr = source.H_elem();
				// 	}
				// 	if(!host_ptr){
				// 		PLEGMA_printf("DEBUG: host_ptr is null while collecting probe %d on rank\n", Collected);
				// 	} else {
				// 		std::vector<double> tmp;
				// 		tmp.resize(nReImLocal);
				// 		for(int i=0;i<nReImLocal;i++) tmp[i] = host_ptr[i];
				// 		local_collected.push_back(std::move(tmp));
				// 		Collected++;
				// 	}
				// }
				// --- end copy ---


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
				PLEGMA_printf("DEBUG: scale_val = (% .12e, % .12e)\n", scale_val.real(), scale_val.imag());

				if(spinColorDil || c_probing){
					TIME(qloops_std->oneEnd_trick(phi,*sourceDil,scale_val,true));
				} else {
					TIME(qloops_std->oneEnd_trick(phi,source,scale_val,true));
				}
			}
		}	

		if((isrc+1)%NdumpStep == 0){
				TIME(qloops_std->dumpLoops(ft, loopsPrefix + tag + options_tag, confID, corr_file_format, isrc));
		}
		if(!accumFlag){ // In case we do not accumulate we clear the buffers
			qloops_std->clearAccumBuffs();
		}
	}

	// --- RUN ORTHO CHECK ON COLLECTED VECTORS ---
	// int Pcollected = (int)local_collected.size();

	// if(Pcollected > 0){
	// 	// local inner-product contributions
	// 	std::vector<std::vector<std::complex<double>>> local_IP(Pcollected, std::vector<std::complex<double>>(Pcollected, {0.0,0.0}));

	// 	for(int i=0;i<Pcollected;i++){
	// 		for(int j=0;j<Pcollected;j++){
	// 			double re_sum = 0.0;
	// 			double im_sum = 0.0;
	// 			for(int c=0;c<nComplexLocal;c++){
	// 				double re_i = local_collected[i][2*c + 0];
	// 				double im_i = local_collected[i][2*c + 1];
	// 				double re_j = local_collected[j][2*c + 0];
	// 				double im_j = local_collected[j][2*c + 1];
	// 				re_sum += re_i * re_j + im_i * im_j;
	// 				im_sum += re_i * im_j - im_i * re_j;
	// 			}
	// 			local_IP[i][j] = std::complex<double>(re_sum, im_sum);
	// 		}
	// 	}

	// 	// flatten and MPI_Allreduce
	// 	int Nmat = Pcollected * Pcollected;
	// 	std::vector<double> local_flat(2*Nmat, 0.0), global_flat(2*Nmat, 0.0);
	// 	for(int i=0;i<Pcollected;i++){
	// 		for(int j=0;j<Pcollected;j++){
	// 			int idx = i*Pcollected + j;
	// 			local_flat[2*idx + 0] = local_IP[i][j].real();
	// 			local_flat[2*idx + 1] = local_IP[i][j].imag();
	// 		}
	// 	}
	// 	MPI_Allreduce(local_flat.data(), global_flat.data(), 2*Nmat, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// 	// reconstruct and print
	// 	std::vector<std::vector<std::complex<double>>> global_IP(Pcollected, std::vector<std::complex<double>>(Pcollected));
	// 	std::vector<double> norms(Pcollected, 0.0);
	// 	for(int i=0;i<Pcollected;i++){
	// 		for(int j=0;j<Pcollected;j++){
	// 			int idx = i*Pcollected + j;
	// 			double gre = global_flat[2*idx + 0];
	// 			double gim = global_flat[2*idx + 1];
	// 			global_IP[i][j] = std::complex<double>(gre, gim);
	// 			if(i==j) norms[i] = std::abs(global_IP[i][i]); // norm squared
	// 		}
	// 	}

	// 	if(rank==0){
	// 		PLEGMA_printf("===== PROBE ORTHOGONALITY CHECK: P = %d =====\n", Pcollected);
	// 		PLEGMA_printf("Norms (|<v_i,v_i>|) per probe (should be >0):\n");
	// 		for(int i=0;i<Pcollected;i++) PLEGMA_printf("  i=%2d   norm2=%.12e\n", i, norms[i]);

	// 		PLEGMA_printf("Normalized overlap matrix O_ij (real,imag):\n");
	// 		const double eps_small = 1e-16;
	// 		for(int i=0;i<Pcollected;i++){
	// 			for(int j=0;j<Pcollected;j++){
	// 				double denom = sqrt(std::max(norms[i]*norms[j], eps_small));
	// 				std::complex<double> Oij = global_IP[i][j] / denom;
	// 				PLEGMA_printf(" (% .6e,% .6e)", Oij.real(), Oij.imag());
	// 			}
	// 			PLEGMA_printf("\n");
	// 		}
	// 		PLEGMA_printf("=============================================\n");
	// 	}
	// }

	if(qloops_std) delete qloops_std;
	if(c_probing) delete cprob;
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	delete solver;
	finalize();
	return 0;
}