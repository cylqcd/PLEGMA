#include <PLEGMA.h>
#include <PLEGMA_utils.h>

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

std::string make_group_name_LIBE(const std::string& type,
                            int val_u, int val_d,
                            int val_s, int val_c) {
    char* group;
    asprintf(&group, "baryon_%s_u[%+d]_d[%+d]_s[%+d]_c[%+d]",
             type.c_str(), val_u, val_d, val_s, val_c);
    std::string result(group);
    free(group);
    return result;
}

std::string make_group_name_SIB(const std::string& type,
                            double val_u, double val_d,
                            double val_s, double val_c) {
    char* group;
    asprintf(&group, "baryon_%s_u[%+.4e]_d[%+.4e]_s[%+.4e]_c[%+.4e]",
             type.c_str(), val_u, val_d, val_s, val_c);
    std::string result(group);
    free(group);
    return result;
}

int main(int argc, char **argv)
{
	initializeOptions(argc, argv, true, listOpt);
	//================ Add your options in this between initializeOptions and initializePLEGMA ================//
	std::vector<double> mu_s;
	std::vector<double> mu_c;
	double mu_ud = mu;
	double mu_ud_factor[QUDA_MAX_MG_LEVEL];
	for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
	double delta_mu = 0.005 * mu_ud;
	int nsmearGauss_s = nsmearGauss/2;
	int nsmearGauss_c = 0;
	bool run_ud = true;
	int start_src = 0;
	double des;
	std::string qedfile;
	std::string srcInputFile = "./input.src";
  	auto add_options = [&](Options& options) {
		options.set("run-ud", "Whether to run or not light quark flavors", verbosity, run_ud);
		options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
		options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
		// options.set("delta-mu", "The mass difference for SIB", verbosity, delta_mu);
		options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
		//options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
		options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
		options.set("start-src", "The source position from which to start the calculation", verbosity, start_src);
		options.set("dqed", "dqed used for LIBE", verbosity, des);
		options.set("qed-filename", "The path to the QED field", verbosity, qedfile);
	};
	add_options(*HGC_options);
   	//=========================================================================================================//
	initializePLEGMA();

	if(link_recon!=QUDA_RECONSTRUCT_NO or link_recon_sloppy!=QUDA_RECONSTRUCT_NO or link_recon_precondition!=QUDA_RECONSTRUCT_NO) {
		PLEGMA_error("QED requires QUDA_RECONSTRUCT_NO link_recon =%d link_recon_sloppy=%d link_recon_predcondition=%d\n",link_recon,link_recon_sloppy,link_recon_precondition);
	}

  	std::string given_twop_filename = twop_filename;
  	{
		PLEGMA_Gauge<double> smearedGauge(BOTH);
		PLEGMA_GaugeU1<double> gaugeU1;

		// Reading from Lime file and loading to device
		PLEGMA_Gauge<double> gauge;
		gauge.readFile(latfile, LIME_FORMAT);
		gauge.calculatePlaq();
		
		// Loading to QUDA and computing plaquette also there
		initGaugeQuda(gauge, true);
		plaqQuda();

		TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
		PLEGMA_printf("Plaquette after smearing:\n");
		smearedGauge.calculatePlaq();

		if(des!=0) {
			std::string U1_conf = qedfile;
			PLEGMA_printf("\n ### Going to read %s ###\n\n", U1_conf.c_str());
			gaugeU1.readFile(U1_conf, LIME_FORMAT);
		}			
		
		if(run_ud) {
			updateOptions(LIGHT);
			mu = mu_ud;
		}
		// else if(mu_s.size()>0) {
		// 	updateOptions(STRANGE);
		// 	mu = mu_s[0];
		// } else {
		// 	updateOptions(CHARM);
		// 	mu = mu_c[0];
		// }
		PLEGMA_printf("\n ### Running setup for mu=%.4e ###\n\n",mu);
		TIME(QUDA_solver solver(mu));

		std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props_u(3);
		std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props_d(3);
		std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props_s(3);
		std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props_c(3);
		if (des != 0) {
			for (int isgn = 0; isgn < 3; isgn++) {
				props_u[isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
				props_d[isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
				props_s[isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
				props_c[isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
			}
		}
    
		for(int isource = start_src ; isource < numSourcePositions; isource++){
			PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
				isource, sourcePositions[isource][0], sourcePositions[isource][1],
				sourcePositions[isource][2], sourcePositions[isource][3]);

			//updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);     
			site& source = sourcePositions[isource];
			PLEGMA_Propagator<float> propUP(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<float> propDN(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<float> propSIBUP(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<float> propSIBDN(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<float> propST;
			PLEGMA_Propagator<float> propCH;
			PLEGMA_Propagator<float> propSIBST;
			PLEGMA_Propagator<float> propSIBCH;

			PLEGMA_Gauge3D<double> smearedGauge3D;
			smearedGauge3D.absorb(smearedGauge, source[DIM_T]);	

			char * src_string;
			asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
			twop_filename = given_twop_filename + std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
			"_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
			"_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE) + src_string + ".h5";
			free(src_string);

			if(access( twop_filename.c_str(), F_OK ) != -1)
			continue;

			auto computePropagator = [&](PLEGMA_Propagator<float>& prop, const double run_mu, WHICHFLAVOR fl, int nSmear) {
				// ensuring mu value
				if(mu != run_mu) {
					updateOptions(fl);
					mu = run_mu;
					solver.UpdateSolver();
				}
				for(int isc = 0 ; isc < 12 ; isc++){
					PLEGMA_Vector<double> vectorInOut;
					{ // Smearing the source
						PLEGMA_Vector3D<double> vector1, vector2;
						vector1.pointSource(source, isc/3, isc%3, DEVICE);
						TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
						vectorInOut.absorb(vector2,source[DIM_T]);
					}
					// Inverting
					PLEGMA_printf("Going to invert %s for component %d\n",
							fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
					TIME(solver.solve(vectorInOut, vectorInOut));
					{ // Smearing the solution
						PLEGMA_Vector<double> vectorAuxD;
						PLEGMA_Vector<float> vectorAuxF;
						TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
						vectorAuxF.copy(vectorAuxD);
						prop.absorb(vectorAuxF, isc/3, isc%3);
					}
				}  
				prop.rotateToPhysicalBase_device(run_mu/abs(run_mu));
				prop.applyBoundaries_device(source[DIM_T]);
			};

			//--------------------- LIBE ---------------------

			for(int isgn=0; isgn<3; isgn++) {
				if(des==0 and isgn!=1) continue;

				double phase = (isgn-1)*des;
				PLEGMA_printf("\nPhase: %.4f\n", phase);
				if(isgn!=1){
					PLEGMA_Gauge<double> gauge2;
					gauge2.copy(gauge);
					gaugeU1.calculatePlaq(phase);
					gauge2.qedPhase(gaugeU1, phase);
					gauge2.calculatePlaq();
					updateGaugeQuda(gauge2, true);	//initGaugeQuda or updateGaugeQuda?
					plaqQuda();
					//applyBoundaryConditions(gauge2,true);
				} else {
					updateGaugeQuda(gauge, true);	
				}
				solver.UpdateSolver();

				if (run_ud) {
					TIME(computePropagator(propUP, mu_ud, LIGHT, nsmearGauss));
					TIME(computePropagator(propDN, -mu_ud, LIGHT, nsmearGauss));
					
					propUP.unload();
					propDN.unload();
					props_u[isgn]->copy(propUP, HOST);
					props_d[isgn]->copy(propDN, HOST);
				}

				if (mu_s.size()>0) {
					TIME(computePropagator(propST, mu_s[0], STRANGE, nsmearGauss_s));
					propST.unload();
					props_s[isgn]->copy(propST, HOST);
				}
				if (mu_c.size()>0) {
					TIME(computePropagator(propCH, mu_c[0], CHARM, nsmearGauss_c));
					propCH.unload();
					props_c[isgn]->copy(propCH, HOST);
				}
			}
			
			PLEGMA_Propagator<float> prop_u;
			PLEGMA_Propagator<float> prop_d;
			PLEGMA_Propagator<float> prop_s;
			PLEGMA_Propagator<float> prop_c;

			if(des!=0 && run_ud) {
				// Base case: all signs are 0 (index 1), contract all baryons
				{
					PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
					prop_u.copy(*props_u[1], HOST); prop_u.load();
					prop_d.copy(*props_d[1], HOST); prop_d.load();
					prop_s.copy(*props_s[1], HOST); prop_s.load();
					prop_c.copy(*props_c[1], HOST); prop_c.load();

					bool only_up = false, only_dn = false, only_st = false, only_ch = false;

					TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch));
					
					std::string group = make_group_name_LIBE("all", 0, 0, 0, 0);
					std::string full_group = "LIBE/" + group;
					corr.setGroups(full_group.c_str());
					THREAD(corr.writeFile(twop_filename, corr_file_format));
				}

				// Now vary one sign at a time: indices 0 (-1) and 2 (+1)
				for (int sign_idx : {0, 2}) {

					// Vary up
					{
						PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
						prop_u.copy(*props_u[sign_idx], HOST); prop_u.load();
						prop_d.copy(*props_d[1], HOST); prop_d.load();
						prop_s.copy(*props_s[1], HOST); prop_s.load();
						prop_c.copy(*props_c[1], HOST); prop_c.load();

						bool only_up = true, only_dn = false, only_st = false, only_ch = false;

						TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch));

						std::string group = make_group_name_LIBE("only_up", sign_idx-1, 0, 0, 0);
						std::string full_group = "LIBE/" + group;
						corr.setGroups(full_group.c_str());
						THREAD(corr.writeFile(twop_filename, corr_file_format));
					}

					// Vary down
					{
						PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
						prop_u.copy(*props_u[1], HOST); prop_u.load();
						prop_d.copy(*props_d[sign_idx], HOST); prop_d.load();
						prop_s.copy(*props_s[1], HOST); prop_s.load();
						prop_c.copy(*props_c[1], HOST); prop_c.load();

						bool only_up = false, only_dn = true, only_st = false, only_ch = false;

						TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch));

						std::string group = make_group_name_LIBE("only_dn", 0, sign_idx-1, 0, 0);
						std::string full_group = "LIBE/" + group;
						corr.setGroups(full_group.c_str());
						THREAD(corr.writeFile(twop_filename, corr_file_format));
					}

					// Vary strange
					{
						PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
						prop_u.copy(*props_u[1], HOST); prop_u.load();
						prop_d.copy(*props_d[1], HOST); prop_d.load();
						prop_s.copy(*props_s[sign_idx], HOST); prop_s.load();
						prop_c.copy(*props_c[1], HOST); prop_c.load();

						bool only_up = false, only_dn = false, only_st = true, only_ch = false;

						TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch));

						std::string group = make_group_name_LIBE("only_st", 0, 0, sign_idx-1, 0);
						std::string full_group = "LIBE/" + group;
						corr.setGroups(full_group.c_str());
						THREAD(corr.writeFile(twop_filename, corr_file_format));
					}

					// Vary charm
					{
						PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
						prop_u.copy(*props_u[1], HOST); prop_u.load();
						prop_d.copy(*props_d[1], HOST); prop_d.load();
						prop_s.copy(*props_s[1], HOST); prop_s.load();
						prop_c.copy(*props_c[sign_idx], HOST); prop_c.load();

						bool only_up = false, only_dn = false, only_st = false, only_ch = true;

						TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch));

						std::string group = make_group_name_LIBE("only_ch", 0, 0, 0, sign_idx-1);
						std::string full_group = "LIBE/" + group;
						corr.setGroups(full_group.c_str());
						THREAD(corr.writeFile(twop_filename, corr_file_format));
					}
				}

				// Loop over all pairs of flavors to assign +1 and -1
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						if (i >= j) continue; // unordered pairs

						// Two permutations: i=+, j=- and i=-, j=+
						for (int sign_case = 0; sign_case < 2; sign_case++) {
							std::array<int, 4> sign_idx = {0, 0, 0, 0};

							sign_idx[i] = (sign_case == 0) ? +1 : -1;
							sign_idx[j] = (sign_case == 0) ? +1 : -1;

							PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);

							prop_u.copy(*props_u[sign_idx[0]], HOST); prop_u.load();
							prop_d.copy(*props_d[sign_idx[1]], HOST); prop_d.load();
							prop_s.copy(*props_s[sign_idx[2]], HOST); prop_s.load();
							prop_c.copy(*props_c[sign_idx[3]], HOST); prop_c.load();

							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_mixed=true));

							std::string group = make_group_name_LIBE("mixed", sign_idx[0]-1, sign_idx[1]-1, sign_idx[2]-1, sign_idx[3]-1);
							std::string full_group = "LIBE/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
						}
					}
				}
			}

			//--------------------- SIB ---------------------
			updateGaugeQuda(gauge, true);
			solver.UpdateSolver();

			if (run_ud) {
				double run_mu_up = mu_ud + delta_mu;
				double run_mu_dn = -mu_ud - delta_mu;
				TIME(computePropagator(propSIBUP, run_mu_up, LIGHT, nsmearGauss));
				TIME(computePropagator(propSIBDN, run_mu_dn, LIGHT, nsmearGauss));
			}

			double run_mu_st = 1.005 * mu_s[0];
			double run_mu_ch = 1.005 * mu_c[0];
			TIME(computePropagator(propSIBST, run_mu_st, STRANGE, nsmearGauss_s));
			TIME(computePropagator(propSIBCH, run_mu_ch, CHARM, nsmearGauss_c));

			{
				prop_u.copy(*props_u[1], HOST);
				prop_u.load();
				prop_d.copy(*props_d[1], HOST);
				prop_d.load();
				prop_s.copy(*props_s[1], HOST);
				prop_s.load();
				prop_c.copy(*props_c[1], HOST);
				prop_c.load();

				PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);

				for (int flavor = 0; flavor < 4; flavor++) {
					bool only_up = false, only_dn = false, only_st = false, only_ch = false, only_mixed = false;

					// Modify the respective propagator
					switch (flavor) {
						case 0:
							only_up = true;
							TIME(corr.contractBaryonsUDSC(propSIBUP, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_mixed));

							make_group_name_SIB("only_up", run_mu_up, -mu_ud, mu_s[0], mu_c[0]);
							std::string full_group = "SIB/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
							break;
						case 1:
							only_dn = true;
							TIME(corr.contractBaryonsUDSC(prop_u, propSIBDN, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_mixed));

							make_group_name_SIB("only_dn", mu_ud, run_mu_dn, mu_s[0], mu_c[0]);
							std::string full_group = "SIB/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
							break;
						case 2:
							only_st = true;
							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, propSIBST, prop_c, only_up, only_dn, only_st, only_ch, only_mixed));

							make_group_name_SIB("only_st", mu_ud, -mu_ud, run_mu_st, mu_c[0]);
							std::string full_group = "SIB/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
							break;
						case 3:
							only_ch = true;
							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, propSIBCH, only_up, only_dn, only_st, only_ch, only_mixed));

							make_group_name_SIB("only_ch", mu_ud, -mu_ud, mu_s[0], run_mu_ch);
							std::string full_group = "SIB/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
							break;
					}
				}
			}
      	}
	}
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	//finalize();
	return 0;
}