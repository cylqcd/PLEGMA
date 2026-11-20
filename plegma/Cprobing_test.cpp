#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <set>
#include <charconv>
#include <optional>
#include <PLEGMA_Cprobing.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <memory>
#include <vector>

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
	std::string loopsPrefix="./";
	int coloring_distance = 1;
	int probing_dimension = 4;
	bool c_probing = false;
	bool spinColorDil = false;
	std::string sigma_conf_path = "";
	int start_src = 0;
	std::string dbg_out_path = ".";
	std::string reference_noise_path;
	bool rotate_reference_chiral = false;
	int dump_solution_color = 0;
	std::string dump_solution_colors;
	int dirac_check_steps = 0;
  	auto add_options = [&](Options& options) {
  		options.set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
		options.set("coloring-distance", "Coloring distance for classical probing", verbosity, coloring_distance);
		options.set("probing-dimension", "Probing dimension for classical probing", verbosity, probing_dimension);
		options.set("c-probing", "Whether to use classical probing", verbosity, c_probing);
		options.set("spin-color-dil", "Whether we want spin color dilution",verbosity, spinColorDil);
		options.set("sigma-config-path", "Path to the configuration file for the sigma factors", verbosity, sigma_conf_path);
		options.set("debug-output-path", "Existing directory for debug vector dumps", verbosity, dbg_out_path);
		options.set("reference-noise-path", "Directory of converted reference Z2 binaries (one source only)", verbosity, reference_noise_path);
		options.set("rotate-reference-chiral", "Rotate the colleague's chiral reference source into UKQCD", verbosity, rotate_reference_chiral);
		options.set("dump-solution-color", "Save the pre-contraction solution for this probing color (0 disables)", verbosity, dump_solution_color);
		options.set("dump-solution-colors", "Comma-separated probing colors to save before contraction", verbosity, dump_solution_colors);
		options.set("dirac-check-steps", "Dump D^k source for k=1..steps and check D*solution for selected solution colors (0 disables)", verbosity, dirac_check_steps);
		options.set("start-src", "Starting index for the stochastic sources (inclusive)", verbosity, start_src);
	};
	add_options(*HGC_options);
	if (rotate_reference_chiral && reference_noise_path.empty())
		PLEGMA_error("rotate-reference-chiral requires reference-noise-path");
	if (rotate_reference_chiral && spinColorDil)
		PLEGMA_error("rotate-reference-chiral requires spin-color-dil false");
	if (probing_dimension != 3 && probing_dimension != 4)
		PLEGMA_error("Classical probing supports only 3D and 4D coloring");
	if (start_src < 0 || start_src >= numSourcePositions)
		PLEGMA_error("start-src must be >= 0 and < nsrc");
	if (!dbg_out_path.empty() && dbg_out_path.back() != '/') dbg_out_path += "/";
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
	if (rotate_reference_chiral) options_tag += "_chiralToUKQCD";
	options_tag += "_nsrc" + std::to_string(numSourcePositions);
   	//=========================================================================================================//
	PLEGMA_printf(">>> About to call initializePLEGMA()\n");
	initializePLEGMA();
	PLEGMA_printf(">>> Returned from initializePLEGMA()\n");

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("GPU rank %d: process coordinates (x,y,z,t)=(%d,%d,%d,%d)\n",
		rank, commCoords(0), commCoords(1), commCoords(2), commCoords(3));
	fflush(stdout);

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

	// Match the hierarchical crosscheck, including gauge paths without "conf.".
	std::string confID = "crosscheck";

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

	PLEGMA_Vector<double> source(BOTH); // BOTH only for debugging
	int rng_seed = 42;
	if(reference_noise_path.empty()) source.randInit(rng_seed);

	PLEGMA_printf("Start probing!\n");
	PLEGMA_Cprobing *cprob = nullptr;
  	if(c_probing) cprob = new PLEGMA_Cprobing(coloring_distance, probing_dimension, sigma_conf_path);

	int Nc = c_probing ? cprob->get_Ncol() : 1;
	if (Nc < 1) PLEGMA_error("Classical probing returned no colors");
	if (dump_solution_color < 0 || dump_solution_color > Nc)
		PLEGMA_error("dump-solution-color must be between 0 and %d", Nc);

	std::set<int> solution_colors;
	if (dump_solution_color > 0)
		solution_colors.insert(dump_solution_color);
	if (!dump_solution_colors.empty()) {
		const char* first = dump_solution_colors.data();
		const char* end = first + dump_solution_colors.size();
		while (true) {
			const char* last = std::find(first, end, ',');
			int color = 0;
			const auto parsed = std::from_chars(first, last, color);
			if (parsed.ec != std::errc{} || parsed.ptr != last || color < 1 || color > Nc)
				PLEGMA_error("dump-solution-colors requires comma-separated integers between 1 and %d", Nc);
			solution_colors.insert(color);
			if (last == end) break;
			first = last + 1;
		}
	}
	PLEGMA_Vector<double> phi(solution_colors.empty() ? DEVICE : BOTH);

	if (dirac_check_steps < 0 || (dirac_check_steps > 0 && solution_colors.empty()))
		PLEGMA_error("dirac-check-steps must be nonnegative and requires selected solution colors when enabled");
	std::unique_ptr<QUDA_dirac> full_dirac;
	std::unique_ptr<PLEGMA_Vector<double>> dirac_out, dirac_next;
	// The test's rescaled phi solves M*phi=rhs in mass-normalized runs.
	// Otherwise phi includes an extra 1/(2*kappa), so use M/(2*kappa).
	const bool mass_normalized_solve =
		inv_params.mass_normalization == QUDA_MASS_NORMALIZATION ||
		inv_params.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION;
	const auto check_normalization =
		mass_normalized_solve ? QUDA_KAPPA_NORMALIZATION : QUDA_MASS_NORMALIZATION;
	if (dirac_check_steps > 0) {
		full_dirac = std::make_unique<QUDA_dirac>(inv_params.dslash_type);
		dirac_out = std::make_unique<PLEGMA_Vector<double>>(BOTH);
		dirac_next = std::make_unique<PLEGMA_Vector<double>>(BOTH);
		PLEGMA_printf("Dirac check uses %s for the dumped solution convention\n",
			mass_normalized_solve ? "M" : "M/(2*kappa)");
	}

	if (c_probing && rank == 0) {
		const int *colors = cprob->H_localColors();
		if (!colors) PLEGMA_error("Classical coloring host array is null");
		const int V3 = HGC_localL[0] * HGC_localL[1] * HGC_localL[2];
		const int Nt_print = std::min(3, HGC_localL[3]);
		for (int t = 0; t < Nt_print; ++t) {
			PLEGMA_printf("\n===== CLASSICAL COLORS FOR LOCAL t = %d =====\n", t);
			for (int i = 0; i < V3; ++i) {
				PLEGMA_printf("%d ", colors[t*V3+i]);
				if ((i+1) % HGC_localL[0] == 0) PLEGMA_printf("\n");
			}
		}
	}
	std::complex<double> scale_val(-1.0, 0.0);
	// --- PROBE COLLECTION SETUP ---
	int M_collect = c_probing ? std::min(Nc, 8) : 0;  // first few color masks
	int Collected = 0;

	int NdofPerSite = N_SPINS * N_COLS;
	int nSitesLocal = HGC_localVolume;
	int nComplexLocal = nSitesLocal * NdofPerSite;
	int nReImLocal = 2 * nComplexLocal;

	std::vector<std::vector<double>> local_collected;
	local_collected.reserve(M_collect);

	std::vector<int> collected_ic;
	collected_ic.reserve(M_collect);

	auto dump_vector_binary =
    [](PLEGMA_Vector<double>& vec,
       const std::string& filename,
       int nReImLocal)
	{
		vec.unload();

		const double *ptr = vec.H_elem();

		if(!ptr){
			PLEGMA_error("Debug vector host pointer is null");
			MPI_Abort(MPI_COMM_WORLD, 1);
			return;
		}

		std::ofstream out(
			filename,
			std::ios::binary
		);

		if (!out) {
			PLEGMA_error("Cannot open debug output file; check debug-output-path");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		out.write(
			reinterpret_cast<const char*>(ptr),
			nReImLocal * sizeof(double)
		);

		out.close();
		if (!out) {
			PLEGMA_error("Failed writing debug output file");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
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
        double *p = source.H_elem();
        for(int j=0; j<nComplexLocal; ++j) {
            if((p[2*j] != 1.0 && p[2*j] != -1.0) || p[2*j+1] != 0.0)
                PLEGMA_error("Reference contains invalid Z2 entry: %s", filename.c_str());
        }
        if (rotate_reference_chiral) {
            // Reference and PLEGMA layout: [spin][QCD color][site][real/imag].
            // Maps the supplied chiral gamma matrices (x,y,z,t = 0,1,2,3)
            // to the UKQCD matrices (x,y,z,t = 1,2,3,4).
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            for (int color = 0; color < N_COLS; ++color) {
                for (int site = 0; site < nSitesLocal; ++site) {
                    for (int ri = 0; ri < 2; ++ri) {
                        const int i0 = 2 * (color * nSitesLocal + site) + ri;
                        const int spin_stride = 2 * N_COLS * nSitesLocal;
                        const int i1 = i0 + spin_stride;
                        const int i2 = i1 + spin_stride;
                        const int i3 = i2 + spin_stride;
                        const double s0 = p[i0], s1 = p[i1];
                        const double s2 = p[i2], s3 = p[i3];
                        p[i0] =  (s1 + s3) * inv_sqrt2;
                        p[i1] = -(s0 + s2) * inv_sqrt2;
                        p[i2] =  (s1 - s3) * inv_sqrt2;
                        p[i3] =  (s2 - s0) * inv_sqrt2;
                    }
                }
            }
            PLEGMA_printf("Rotated reference source from chiral to UKQCD basis\n");
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

		if (isrc == start_src) {
			dump_vector_binary(
				source,
				dbg_out_path + (rotate_reference_chiral ? "debug_raw_UKQCD_rank" : "debug_raw_Z2_rank")
							+ std::to_string(rank) + ".bin",
				nReImLocal
			);
		}

		for(int ic = 1; ic <= Nc; ic++){
			// Sum color contributions downstream; do not divide by Nc.
			qloops_std->clearAccumBuffs();
			for(int isc = 0; isc < Nsc; isc++){
				if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}

				// ---------------------------------------------------------
				// DEBUG: keep copy of source immediately BEFORE classical coloring
				// ---------------------------------------------------------

				std::vector<double> before;

				bool debug_this_probe =
					(c_probing &&
					isrc == start_src &&
					isc == 0 &&
					ic <= M_collect);

				if(debug_this_probe){

					PLEGMA_Vector<double> *before_vec =
						spinColorDil ? sourceDil : &source;

					before_vec->unload();

					const double *before_ptr = before_vec->H_elem();

					if(!before_ptr){
						PLEGMA_printf(
							"DEBUG ERROR: before_ptr is null for ic=%d\n",
							ic
						);
					}
					else{
						before.resize(nReImLocal);

						for(int i = 0; i < nReImLocal; i++)
							before[i] = before_ptr[i];
					}
				}

				// ---------------------------------------------------------
				// DEBUG: source immediately BEFORE classical coloring
				// ---------------------------------------------------------

				if(c_probing &&
				isrc == start_src &&
				isc == 0 &&
				ic == 1)
				{
					if(spinColorDil){
						dump_vector_binary(
							*sourceDil,
							dbg_out_path
							+ "debug_before_C_"
							+ std::to_string(ic)
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
							+ "debug_before_C_"
							+ std::to_string(ic)
							+ "_rank"
							+ std::to_string(rank)
							+ ".bin",
							nReImLocal
						);
					}
				}

				if(spinColorDil && c_probing){ sourceDil->applyCprobColoring(*sourceDil,*cprob,ic,indDof);}
				else if(!spinColorDil && c_probing){ sourceDil->applyCprobColoring(source,*cprob,ic,indDof);}

				// ---------------------------------------------------------
				// DEBUG: source AFTER classical coloring
				// ---------------------------------------------------------

				if(c_probing &&
				isrc == start_src &&
				isc == 0 &&
				ic <= M_collect)
				{
					dump_vector_binary(
						*sourceDil,
						dbg_out_path
						+ "debug_cD"
						+ std::to_string(coloring_distance)
						+ "_after_C_"
						+ std::to_string(ic)
						+ "_rank"
						+ std::to_string(rank)
						+ ".bin",
						nReImLocal
					);
				}

				// ------------ DEBUG ----------------------------------------------------
				if(debug_this_probe){
					sourceDil->unload();

					// --- COPY THIS PROBE INTO LOCAL STORAGE ---
					if(isrc == start_src && isc == 0 && Collected < M_collect){

						const double *host_ptr = sourceDil->H_elem();

						if(!host_ptr){
							MPI_Abort(MPI_COMM_WORLD, 1);
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
							collected_ic.push_back(ic);
							Collected++;
						}
					}
					// -----------------------------------------------------------------------
				}

				if(debug_this_probe){

					const double *after = sourceDil->H_elem();

					if(!after){
						PLEGMA_printf(
							"DEBUG ERROR: after ptr is null for ic=%d\n",
							ic
						);
					}
					else if(before.size() != nReImLocal){
					    PLEGMA_printf(
							"DEBUG ERROR: before vector unavailable for ic=%d\n",
							ic
						);
					}
					else if(rank == 0){
						PLEGMA_printf(
							"\n===== CLASSICAL MASK CHECK ic=%d (ratios should be 0 or 1) =====\n",
							ic
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

				auto& rhs = (spinColorDil || c_probing) ? *sourceDil : source;
				const bool dump_solution = solution_colors.count(ic) != 0;
				std::string file_suffix;
				if (dump_solution) {
					file_suffix = options_tag
						+ "_Ns" + std::to_string(isrc)
						+ "_Nh" + std::to_string(ic)
						+ "_dil" + std::to_string(isc)
						+ "_x" + std::to_string(commCoords(0))
						+ "_y" + std::to_string(commCoords(1))
						+ "_z" + std::to_string(commCoords(2))
						+ "_t" + std::to_string(commCoords(3)) + ".bin";
				}

				if (dump_solution && full_dirac) {
					dump_vector_binary(rhs, dbg_out_path + "debug_dirac_rhs" + file_suffix, nReImLocal);
					full_dirac->apply<M>(*dirac_out, rhs, check_normalization);
					for (int step = 1; ; ++step) {
						dump_vector_binary(*dirac_out, dbg_out_path + "debug_dirac_D"
							+ std::to_string(step) + "_rhs" + file_suffix, nReImLocal);
						if (step == dirac_check_steps) break;
						full_dirac->apply<M>(*dirac_next, *dirac_out, check_normalization);
						dirac_out.swap(dirac_next);
					}
				}

				TIME(solver->solve(phi, rhs));
				phi.scale(1./(2.*inv_params.kappa));

				if (dump_solution)
					dump_vector_binary(phi, dbg_out_path + "debug_solution" + file_suffix, nReImLocal);

				if (dump_solution && full_dirac) {
					full_dirac->apply<M>(*dirac_out, phi, check_normalization);
					dump_vector_binary(*dirac_out, dbg_out_path + "debug_dirac_D_solution"
						+ file_suffix, nReImLocal);
					const double rhs_norm = rhs.norm();
					dirac_out->add(rhs, {-1.0, 0.0});
					const double residual_norm = dirac_out->norm();
					if (rhs_norm > 0.0) {
						PLEGMA_printf("Dirac check Ns%d Nh%d dil%d: relative full residual %.12e\n",
							isrc, ic, isc, residual_norm / rhs_norm);
					} else {
						PLEGMA_printf("Dirac check Ns%d Nh%d dil%d: zero RHS, absolute full residual %.12e\n",
							isrc, ic, isc, residual_norm);
					}
				}
				// PLEGMA_printf("DEBUG: scale_val = (% .12e, % .12e)\n", scale_val.real(), scale_val.imag());

				if(spinColorDil || c_probing){ 
					TIME(qloops_std->oneEnd_trick(phi, *sourceDil, scale_val, true)); 
				} else {
					TIME(qloops_std->oneEnd_trick(phi, source, scale_val, true));
				}
			}

			TIME(qloops_std->dumpLoops(ft, loopsPrefix + tag + options_tag, confID, corr_file_format, isrc, ic));
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
					"  ic=%2d   norm2=%.12e\n",
					collected_ic[i],
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

	delete qloops_std;
	delete sourceDil;
	delete cprob;
	while (not threads.empty()) { threads.back().join(); threads.pop_back(); }
	delete ft[0];
	dirac_next.reset();
	dirac_out.reset();
	full_dirac.reset();
	delete solver;
	finalize();
	return 0;
}
