#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <set>
#include <optional>
#include <PLEGMA_Hprobing.h>
#include <fstream>

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
	std::string loopsPrefix="./";
	int k_probing = 0;
	HGC_options->set("k-probing", "Hierarchical probing, with distance D=2**k (Options:0,1,2,3,...) (0 means No probing)", verbosity, k_probing);
	int probing_dimension = 4;
	HGC_options->set("probing-dimension", "Dimension for the hierarchical probing", verbosity, probing_dimension);
	if (probing_dimension != 3 && probing_dimension != 4)
    	PLEGMA_error("Hierarchical probing supports only 3D and 4D coloring");
	int hadamLow=0;
	int Nhadam = (k_probing>0) ? 2*std::pow(2, probing_dimension*(k_probing-1)) : 1;
	int hadamHgh = Nhadam;
	bool spinColorDil = false;
	int start_src = 0;
	std::string dbg_out_path=".";
	std::string reference_noise_path;
	auto add_options = [&](Options& options) {
		options.set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
		options.set("hadamard-low", "From which Hadamard vector to start (Options:[0,max))", verbosity, hadamLow);
		options.set("hadamard-high", "Up to which Hadamard vector to stop (Options: 0>= , <=max) (default max)", verbosity, hadamHgh);
		options.set("spin-color-dil", "Whether we want spin color dilution",verbosity,spinColorDil);
		options.set("start-src", "Starting index for the stochastic sources (inclusive)", verbosity, start_src);
		options.set("reference-noise-path", "Directory of converted reference Z2 binaries (one source only)", verbosity, reference_noise_path);
		options.set("debug-output-path", "Where to save the debug outputs", verbosity, dbg_out_path);
	};
	add_options(*HGC_options);
	if(!dbg_out_path.empty() && dbg_out_path.back() != '/')
    dbg_out_path += "/";
	int Nsc = spinColorDil ? N_SPINS*N_COLS : 1;
	if((k_probing>0) && (hadamLow<0 || hadamHgh<0)) PLEGMA_error("Negative values for number of Hadamard vector not allowed");
	if((k_probing>0) && (hadamLow>=hadamHgh))  PLEGMA_error("hadamard-high should be > hadamard-low");
	if((k_probing>0) && (hadamHgh>Nhadam)) PLEGMA_error("hadamard-high should be <= from max number of Hadamard vectors");
	std::string tag = "/traceH";
	std::string options_tag = "";
	if (k_probing>0) {
		int coloring_distance = std::pow(2,k_probing-1);
		options_tag += "_cD" + std::to_string(coloring_distance);
	}
	if (spinColorDil) {
		options_tag += "_dil";
	} else {
		options_tag += "_nodil";
	}
	options_tag += "_nsrc" + std::to_string(numSourcePositions);
	options_tag += "_hadamHgh" + std::to_string(hadamHgh);
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

	// ensuring mu negative
	if(mu>0) mu*=-1.;
  	QUDA_solver *solver = new QUDA_solver(mu);
	QudaInvertParam inv_params = solver->getInvParams();

	PLEGMA_Vector<double> *sourceDil = nullptr;
  	if(spinColorDil || k_probing>0) sourceDil = new PLEGMA_Vector<double>(BOTH);

	PLEGMA_FT<double> *ft[2]={nullptr,nullptr};
	ft[0] = new PLEGMA_FT<double>(maxQsq, 3);

	bool oneDLoops = false;
  	bool twoDLoops = false;
	PLEGMA_QLoops<double> *qloops_std = nullptr;
	qloops_std = new PLEGMA_QLoops<double>(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);

	PLEGMA_Vector<double> phi;
	PLEGMA_Vector<double> source(BOTH);
	int rng_seed = 42;
	if(reference_noise_path.empty()) source.randInit(rng_seed);

	PLEGMA_Hprobing *hprob = nullptr;

	// std::size_t foundPos = latfile.find("conf.");
	// if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
	std::string confID = "crosscheck"; //latfile.substr(foundPos+5,latfile.length());

	if(k_probing>0) hprob = new PLEGMA_Hprobing(k_probing, probing_dimension);

	// ------------ DEBUG: PRINT FIRST 3 TIMESLICES OF COLORING ------------
	if(k_probing > 0){

		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank == 0){

			const int *colors = hprob->H_arrVc();

			const int V3 =
				HGC_localL[0] *
				HGC_localL[1] *
				HGC_localL[2];

			const int Nt_print = std::min(3, HGC_localL[3]);

			for(int t = 0; t < Nt_print; t++){

				PLEGMA_printf(
					"\n===== COLORS FOR LOCAL t = %d =====\n",
					t
				);

				for(int i = 0; i < V3; i++){

					PLEGMA_printf(
						"%d ",
						colors[t*V3 + i]
					);

					// Just for readability
					if((i+1) % HGC_localL[0] == 0)
						PLEGMA_printf("\n");
				}

				PLEGMA_printf("\n");
			}
		}
	}
	// --------------------------------------------------------------------

	int N_probes_h = (k_probing>0) ? (hadamHgh - hadamLow) : 1;
	std::complex<double> scale_val = std::complex<double>(-1.0, 0.0);
	// PLEGMA_printf("DEBUG: hierarchical N_probes_h=%d scale_val=% .12e\n", N_probes_h, scale_val.real());

	// ------------ DEBUG ----------------------------------------------------
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// --- PROBE COLLECTION SETUP ---
	int M_collect = std::min(Nhadam, 8);  // first few Hadamard probes
	int Collected = 0;

	int NdofPerSite = 12;
	int nSitesLocal = HGC_localVolume;
	int nComplexLocal = nSitesLocal * NdofPerSite;
	int nReImLocal = 2 * nComplexLocal;

	std::vector<std::vector<double>> local_collected;
	local_collected.reserve(M_collect);

	std::vector<int> collected_ih;
	collected_ih.reserve(M_collect);

	auto dump_vector_binary =
    [](PLEGMA_Vector<double>& vec,
       const std::string& filename,
       int nReImLocal)
	{
		vec.unload();

		const double *ptr = vec.H_elem();

		if(!ptr){
			PLEGMA_printf(
				"DEBUG ERROR: H_elem() is null for %s\n",
				filename.c_str()
			);
			return;
		}

		std::ofstream out(
			filename,
			std::ios::binary
		);

		out.write(
			reinterpret_cast<const char*>(ptr),
			nReImLocal * sizeof(double)
		);

		out.close();
	};
	// -----------------------------------------------------------------------


    if(!reference_noise_path.empty()) {
        if(numSourcePositions - start_src != 1)
            PLEGMA_error("Reference noise requires exactly one source: nsrc=start-src+1");
        const int local[4] = {32,32,16,16};
        const int grid[4] = {1,1,2,4};
        for(int d=0; d<4; ++d) {
            if(HGC_localL[d] != local[d] || HGC_nProc[d] != grid[d])
                PLEGMA_error("Reference binaries require local xyzt=32 32 16 16 and procs=1 1 2 4");
        }
    }
    auto load_reference_noise = [&]() {
        // Use actual communicator coordinates rather than assuming rank numbering.
        const int file_rank = 4*commCoords(2) + commCoords(3);
        const std::string filename = reference_noise_path + "/reference_Z2_rank"
                                   + std::to_string(file_rank) + ".bin";
        const std::streamsize bytes = static_cast<std::streamsize>(nReImLocal)*sizeof(double);
        std::ifstream input(filename, std::ios::binary | std::ios::ate);
        if(!input || input.tellg() != static_cast<std::streamoff>(bytes))
            PLEGMA_error("Missing or wrong-size reference noise: %s", filename.c_str());
        input.seekg(0);
        input.read(reinterpret_cast<char*>(source.H_elem()), bytes);
        if(!input) PLEGMA_error("Cannot read reference noise: %s", filename.c_str());
        const double *p = source.H_elem();
        for(int j=0; j<nComplexLocal; ++j) {
            if((p[2*j] != 1.0 && p[2*j] != -1.0) || p[2*j+1] != 0.0)
                PLEGMA_error("Reference contains invalid Z2 entry: %s", filename.c_str());
        }
        source.load();
    };

	std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
	for(int isrc = start_src; isrc < numSourcePositions; isrc++){
		PLEGMA_printf("\n ### Calculations for source-position %d begin now ###\n\n", isrc);
    	if(reference_noise_path.empty()) {
            source.stochastic_Z(2);
        } else {
            load_reference_noise();
        }
        if(isrc == start_src) {
            dump_vector_binary(source, dbg_out_path + "debug_raw_Z2_rank"
                               + std::to_string(rank) + ".bin", nReImLocal);
        }
		for(int ih = hadamLow; ih < hadamHgh; ih++){
			qloops_std->clearAccumBuffs();
			for(int isc = 0; isc < Nsc; isc++){
				if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}

				// ---------------------------------------------------------
				// DEBUG: keep copy of source immediately BEFORE HP coloring
				// ---------------------------------------------------------

				std::vector<double> before;

				bool debug_this_probe =
					(k_probing > 0 &&
					isrc == start_src &&
					isc == 0 &&
					ih < std::min(hadamLow + 8, hadamHgh));

				if(debug_this_probe){

					PLEGMA_Vector<double> *before_vec =
						spinColorDil ? sourceDil : &source;

					before_vec->unload();

					const double *before_ptr = before_vec->H_elem();

					if(!before_ptr){
						PLEGMA_printf(
							"DEBUG ERROR: before_ptr is null for ih=%d\n",
							ih
						);
					}
					else{
						before.resize(nReImLocal);

						for(int i = 0; i < nReImLocal; i++)
							before[i] = before_ptr[i];
					}
				}

				// ---------------------------------------------------------
				// DEBUG: source immediately BEFORE hierarchical coloring
				// ---------------------------------------------------------

				if(k_probing > 0 &&
				isrc == start_src &&
				isc == 0 &&
				ih == hadamLow)
				{
					if(spinColorDil){
						dump_vector_binary(
							*sourceDil,
							dbg_out_path
							+ "debug_before_H_"
							+ std::to_string(ih)
							+ "_rank"
							+ std::to_string(rank)
							+ ".bin",
							nReImLocal
						);
					}
					else{
						dump_vector_binary(
							source,
							dbg_out_path
							+ "debug_before_H_"
							+ std::to_string(ih)
							+ "_rank"
							+ std::to_string(rank)
							+ ".bin",
							nReImLocal
						);
					}
				}

				if(spinColorDil && k_probing>0){ sourceDil->applyHprobColoring(*sourceDil,*hprob,ih,indDof);}
				else if(!spinColorDil && k_probing>0){ sourceDil->applyHprobColoring(source,*hprob,ih,indDof);}

				// ---------------------------------------------------------
				// DEBUG: source AFTER hierarchical coloring
				// ---------------------------------------------------------

				if(k_probing > 0 &&
				isrc == start_src &&
				isc == 0 &&
				ih < std::min(hadamLow + 8, hadamHgh))
				{
					dump_vector_binary(
						*sourceDil,
						dbg_out_path
						+ "debug_k"
						+ std::to_string(k_probing)
						+ "_after_H_"
						+ std::to_string(ih)
						+ "_rank"
						+ std::to_string(rank)
						+ ".bin",
						nReImLocal
					);
				}

				// ------------ DEBUG ----------------------------------------------------
				if(k_probing > 0){
					sourceDil->unload();

					// --- COPY THIS PROBE INTO LOCAL STORAGE ---
					if(isrc == start_src && isc == 0 && Collected < M_collect){

						const double *host_ptr = sourceDil->H_elem();

						if(!host_ptr){
							PLEGMA_printf(
								"DEBUG: host_ptr is null while collecting probe %d\n",
								Collected
							);
						}
						else{
							std::vector<double> tmp(nReImLocal);

							for(int i = 0; i < nReImLocal; i++)
								tmp[i] = host_ptr[i];

							local_collected.push_back(std::move(tmp));
							collected_ih.push_back(ih);
							Collected++;
						}
					}
					// -----------------------------------------------------------------------
				}

				if(debug_this_probe){

					const double *after = sourceDil->H_elem();

					if(!after){
						PLEGMA_printf(
							"DEBUG ERROR: after ptr is null for ih=%d\n",
							ih
						);
					}
					else if(before.size() != nReImLocal){
					    PLEGMA_printf(
							"DEBUG ERROR: before vector unavailable for ih=%d\n",
							ih
						);
					}
					else if(rank == 0){
						PLEGMA_printf(
							"\n===== COLOR APPLICATION CHECK ih=%d =====\n",
							ih
						);

						const int Nprint = 20;

						for(int c = 0;
							c < std::min(Nprint, nComplexLocal);
							c++)
						{
							std::complex<double> z_before(
								before[2*c],
								before[2*c+1]
							);

							std::complex<double> z_after(
								after[2*c],
								after[2*c+1]
							);

							std::complex<double> ratio(0.0,0.0);

							if(std::abs(z_before) > 1e-14)
								ratio = z_after / z_before;

							PLEGMA_printf(
								"c=%4d  before=(% .3f,% .3f)"
								"  after=(% .3f,% .3f)"
								"  ratio=(% .3f,% .3f)\n",
								c,
								z_before.real(),
								z_before.imag(),
								z_after.real(),
								z_after.imag(),
								ratio.real(),
								ratio.imag()
							);
						}

						PLEGMA_printf(
							"==========================================\n"
						);
					}
				}

				if(spinColorDil || k_probing>0){
					TIME(solver->solve(phi,*sourceDil)); 
				} else {
					TIME(solver->solve(phi,source));
				}
				phi.scale(1./(2.*inv_params.kappa));
				// PLEGMA_printf("DEBUG: scale_val = (% .12e, % .12e)\n", scale_val.real(), scale_val.imag());

				if(spinColorDil || k_probing>0){ 
					TIME(qloops_std->oneEnd_trick(phi, *sourceDil, scale_val, true)); 
				} else {
					TIME(qloops_std->oneEnd_trick(phi, source, scale_val, true));
				}
			}

			TIME(qloops_std->dumpLoops(ft, loopsPrefix + tag + options_tag, confID, corr_file_format, isrc, ih));
		}
  	}

	// --- RUN ORTHO CHECK ON COLLECTED VECTORS ---
	int Pcollected = (int)local_collected.size();

	if(Pcollected > 0){

		std::vector<std::vector<std::complex<double>>> local_IP(
			Pcollected,
			std::vector<std::complex<double>>(
				Pcollected,
				{0.0,0.0}
			)
		);

		for(int i = 0; i < Pcollected; i++){
			for(int j = 0; j < Pcollected; j++){

				double re_sum = 0.0;
				double im_sum = 0.0;

				for(int c = 0; c < nComplexLocal; c++){

					double re_i = local_collected[i][2*c + 0];
					double im_i = local_collected[i][2*c + 1];

					double re_j = local_collected[j][2*c + 0];
					double im_j = local_collected[j][2*c + 1];

					// conj(v_i) * v_j
					re_sum += re_i * re_j + im_i * im_j;
					im_sum += re_i * im_j - im_i * re_j;
				}

				local_IP[i][j] =
					std::complex<double>(re_sum, im_sum);
			}
		}

		// Flatten and MPI reduce
		int Nmat = Pcollected * Pcollected;

		std::vector<double> local_flat(2*Nmat, 0.0);
		std::vector<double> global_flat(2*Nmat, 0.0);

		for(int i = 0; i < Pcollected; i++){
			for(int j = 0; j < Pcollected; j++){

				int idx = i*Pcollected + j;

				local_flat[2*idx + 0] =
					local_IP[i][j].real();

				local_flat[2*idx + 1] =
					local_IP[i][j].imag();
			}
		}

		MPI_Allreduce(
			local_flat.data(),
			global_flat.data(),
			2*Nmat,
			MPI_DOUBLE,
			MPI_SUM,
			MPI_COMM_WORLD
		);

		// Reconstruct global inner products
		std::vector<std::vector<std::complex<double>>> global_IP(
			Pcollected,
			std::vector<std::complex<double>>(Pcollected)
		);

		std::vector<double> norms(Pcollected, 0.0);

		for(int i = 0; i < Pcollected; i++){
			for(int j = 0; j < Pcollected; j++){

				int idx = i*Pcollected + j;

				double gre = global_flat[2*idx + 0];
				double gim = global_flat[2*idx + 1];

				global_IP[i][j] =
					std::complex<double>(gre,gim);

				if(i == j)
					norms[i] =
						std::abs(global_IP[i][i]);
			}
		}

		if(rank == 0){

			PLEGMA_printf(
				"\n===== COLORED STOCHASTIC PROBE ORTHOGONALITY CHECK: P = %d =====\n",
				Pcollected
			);

			PLEGMA_printf(
				"Norms (|<v_i,v_i>|):\n"
			);

			for(int i = 0; i < Pcollected; i++)
				PLEGMA_printf(
					"  ih=%2d   norm2=%.12e\n",
					collected_ih[i],
					norms[i]
				);

			PLEGMA_printf(
				"\nNormalized overlap matrix O_ij:\n"
			);

			const double eps_small = 1e-16;

			for(int i = 0; i < Pcollected; i++){
				for(int j = 0; j < Pcollected; j++){

					double denom =
						sqrt(std::max(
							norms[i]*norms[j],
							eps_small
						));

					std::complex<double> Oij =
						global_IP[i][j] / denom;

					PLEGMA_printf(
						" (% .6e,% .6e)",
						Oij.real(),
						Oij.imag()
					);
				}

				PLEGMA_printf("\n");
			}

			PLEGMA_printf(
				"=========================================================\n\n"
			);
		}
	}

	if(qloops_std) delete qloops_std;
	if(sourceDil) delete sourceDil;
	delete hprob;

	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	delete solver;
	finalize();
	return 0;
}