#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <set>
#ifdef PLEGMA_UDSC_BARYONS
#include <PLEGMA_baryons_udsc.cuh>
#endif

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

std::string make_group_name_QED(const std::string& type,
								 double mu_val,
                                 int val_u, int val_d,
                                 int val_s, int val_c) {
    char* group;
    asprintf(&group, "baryon_%s_mul[%.4f]_u[%+d]_d[%+d]_s[%+d]_c[%+d]",
             type.c_str(), mu_val, val_u, val_d, val_s, val_c);
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

std::string make_group_name_CM(const std::string& type,
                            double val_u, double val_d,
                            double val_s, double val_c,
							double val_dk) {
    char* group;
    asprintf(&group, "baryon_%s_u[%+.4e]_d[%+.4e]_s[%+.4e]_c[%+.4e]_dk[%+.4e]",
             type.c_str(), val_u, val_d, val_s, val_c, val_dk);
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
	std::vector<double> mu_l;
	double mu_ud = mu;
	double mu_ud_factor[QUDA_MAX_MG_LEVEL];
	for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
	int nsmearGauss_s = nsmearGauss/2;
	int nsmearGauss_c = 0;
	bool run_ud = true;
	bool run_SIB = true;
	bool run_CM = true;
	bool run_heavy_SIB = false;
	bool run_heavy_CM = false;
	int start_src = 0;
	double des;
	int SIB_shift_sign = 1;
	std::vector<double> dks;
	std::string qedfile;
	std::string srcInputFile = "./input.src";
  	auto add_options = [&](Options& options) {
		options.set("run-ud", "Whether to run or not light quark flavors", verbosity, run_ud);
		options.set("run-SIB", "Whether to run or not strong isospin breaking", verbosity, run_SIB);
		options.set("run-CM", "Whether to run or not critical mass", verbosity, run_CM);
		options.set("mu-l", "List of additional mu-light to run for the light quarks in baryons", verbosity, mu_l); // mul-factors=1,4,7,10
		options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
		options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
		// options.set("delta-mu", "The mass difference for SIB", verbosity, delta_mu);
		options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
		// options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
		options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
		options.set("start-src", "The source position from which to start the calculation", verbosity, start_src);
		options.set("dqed", "dqed used for LIBE", verbosity, des);
  		HGC_options->set("dkappa", "dkappa used for LIBE", verbosity, dks);
		options.set("qed-filename", "The path to the QED field", verbosity, qedfile);
		options.set("run-heavy-SIB", "Whether to run SIB for strange and charm quarks", verbosity, run_heavy_SIB);
		options.set("run-heavy-CM", "Whether to run CM for strange and charm quarks", verbosity, run_heavy_CM);
		options.set("SIB-shift-sgn", "The sign of the shift for SIB", verbosity, SIB_shift_sign);
	};
	add_options(*HGC_options);
   	//=========================================================================================================//
	initializePLEGMA();

	// mu_l.insert(mu_l.begin(), mu);
	double kappa0 = kappa;

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
		
		if(run_ud) {
			// updateOptions(LIGHT);
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

		std::vector<std::vector<std::shared_ptr<PLEGMA_Propagator<double>>>> props_u;
		std::vector<std::vector<std::shared_ptr<PLEGMA_Propagator<double>>>> props_d;

		int Nmu = mu_l.size();
		props_u.resize(Nmu, std::vector<std::shared_ptr<PLEGMA_Propagator<double>>>(3));
		props_d.resize(Nmu, std::vector<std::shared_ptr<PLEGMA_Propagator<double>>>(3));

		for (int imu = 0; imu < Nmu; imu++) {
			for (int isgn = 0; isgn < 3; ++isgn) {
				props_u[imu][isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
				props_d[imu][isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
			}
		}

		std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props_s(3);
		std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props_c(3);
		for (int isgn = 0; isgn < 3; isgn++) {
			props_s[isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
			props_c[isgn] = std::make_shared<PLEGMA_Propagator<double>>(HOST);
		}
    
		for(int isource = start_src ; isource < numSourcePositions; isource++){
			PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
				isource, sourcePositions[isource][0], sourcePositions[isource][1],
				sourcePositions[isource][2], sourcePositions[isource][3]);
			
			if(kappa != kappa0) {
				kappa = kappa0;
				solver.UpdateSolver();
			}
			

			//updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);     
			site& source = sourcePositions[isource];
			PLEGMA_Propagator<double> none(NONE);
			PLEGMA_Propagator<double> propUP(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<double> propDN(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<double> propST;
			PLEGMA_Propagator<double> propCH;
			PLEGMA_Propagator<double> propSIBUP(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<double> propSIBDN(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<double> propSIBST(run_SIB ? BOTH : NONE);
			PLEGMA_Propagator<double> propSIBCH(run_SIB ? BOTH : NONE);
			PLEGMA_Propagator<double> propCMUP(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<double> propCMDN(run_ud ? BOTH : NONE);
			PLEGMA_Propagator<double> propCMST(run_CM ? BOTH : NONE);
			PLEGMA_Propagator<double> propCMCH(run_CM ? BOTH : NONE);

			PLEGMA_Gauge3D<double> smearedGauge3D;
			smearedGauge3D.absorb(smearedGauge, source[DIM_T]);	

			if(des!=0) {
				char * src_string;
				asprintf(&src_string, "%04d", isource);
				std::string U1_conf = qedfile + src_string;
				PLEGMA_printf("\n ### Going to read %s ###\n\n", U1_conf.c_str());
				gaugeU1.readFile(U1_conf, LIME_FORMAT);
				free(src_string);
			}			

			char * src_string;
			asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
			twop_filename = given_twop_filename + std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
			"_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
			"_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE) + src_string + ".h5";
			free(src_string);

			if(access( twop_filename.c_str(), F_OK ) != -1)
			continue;

			auto computePropagator = [&](PLEGMA_Propagator<double>& prop, const double run_mu, WHICHFLAVOR fl, int nSmear) {
				// ensuring mu value
				if(mu != run_mu) {
					// updateOptions(fl);
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
						// PLEGMA_Vector<float> vectorAuxF;
						TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
						// vectorAuxF.copy(vectorAuxD);
						prop.absorb(vectorAuxD, isc/3, isc%3);
					}
				}  
				prop.rotateToPhysicalBase_device(run_mu/abs(run_mu));
				prop.applyBoundaries_device(source[DIM_T]);
			};

			//--------------------- QED ---------------------
			
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
					updateGaugeQuda(gauge2, true);
					plaqQuda();
					//applyBoundaryConditions(gauge2,true);
				} else {
					updateGaugeQuda(gauge, true);	
				}
				solver.UpdateSolver();

				if (run_ud) {
					for (int imu = 0; imu < mu_l.size(); imu++) {
							PLEGMA_printf("[QED] Running inversion for mu = %.6e\n", mu_l[imu]);

							// Compute and store light quark propagators
							TIME(computePropagator(propUP,  mu_l[imu], LIGHT, nsmearGauss));
							TIME(computePropagator(propDN, -mu_l[imu], LIGHT, nsmearGauss));

							propUP.unload();
							propDN.unload();

							props_u[imu][isgn]->copy(propUP, HOST);
							props_d[imu][isgn]->copy(propDN, HOST);
						}
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
			
			PLEGMA_Propagator<double> prop_u;
			PLEGMA_Propagator<double> prop_d;
			PLEGMA_Propagator<double> prop_s;
			PLEGMA_Propagator<double> prop_c;

			for (int imu = 0; imu < mu_l.size(); imu++) {
				double mu_val = mu_l[imu];

				if (des != 0 && run_ud) {

					// --------- All signs zero -------------
					{
						PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
						TIME(prop_u.copy(*props_u[imu][1], HOST); prop_u.load());
						TIME(prop_d.copy(*props_d[imu][1], HOST); prop_d.load());
						TIME(prop_s.copy(*props_s[1], HOST); prop_s.load());
						TIME(prop_c.copy(*props_c[1], HOST); prop_c.load());
						
						if (imu == 0) {		
							bool only_up = false, only_dn = false, only_st = false, only_ch = false, only_light = false;
							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

							std::string group = make_group_name_QED("all", mu_val, 0, 0, 0, 0);
							std::string full_group = "QED/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
						} else {
							bool only_up = false, only_dn = false, only_st = false, only_ch = false, only_light = true;
							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

							std::string group = make_group_name_QED("only_light", mu_val, 0, 0, 0, 0);
							std::string full_group = "QED/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
						}
					}

					// --------- Varying each sign separately -------------
					for (int sign_idx : {0, 2}) {

						// UP
						{
							PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
							TIME(prop_u.copy(*props_u[imu][sign_idx], HOST); prop_u.load());
							TIME(prop_d.copy(*props_d[imu][1], HOST); prop_d.load());
							TIME(prop_s.copy(*props_s[1], HOST); prop_s.load());
							TIME(prop_c.copy(*props_c[1], HOST); prop_c.load());

							bool only_up = true, only_dn = false, only_st = false, only_ch = false, only_light = false;
							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

							std::string group = make_group_name_QED("only_up", mu_val, sign_idx - 1, 0, 0, 0);
							std::string full_group = "QED/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
						}

						// DOWN
						{
							PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
							TIME(prop_u.copy(*props_u[imu][1], HOST); prop_u.load());
							TIME(prop_d.copy(*props_d[imu][sign_idx], HOST); prop_d.load());
							TIME(prop_s.copy(*props_s[1], HOST); prop_s.load());
							TIME(prop_c.copy(*props_c[1], HOST); prop_c.load());

							bool only_up = false, only_dn = true, only_st = false, only_ch = false, only_light = false;
							TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

							std::string group = make_group_name_QED("only_dn", mu_val, 0, sign_idx - 1, 0, 0);
							std::string full_group = "QED/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
						}

						// STRANGE
						{
							PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
							TIME(prop_u.copy(*props_u[imu][1], HOST); prop_u.load());
							TIME(prop_d.copy(*props_d[imu][1], HOST); prop_d.load());
							TIME(prop_s.copy(*props_s[sign_idx], HOST); prop_s.load());
							TIME(prop_c.copy(*props_c[1], HOST); prop_c.load());

							if (imu == 0) {	
								bool only_up = false, only_dn = false, only_st = true, only_ch = false, only_light = false;
								TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

								std::string group = make_group_name_QED("only_st", mu_val, 0, 0, sign_idx - 1, 0);
								std::string full_group = "QED/" + group;
								corr.setGroups(full_group.c_str());
								THREAD(corr.writeFile(twop_filename, corr_file_format));
							} else {
								bool only_up = false, only_dn = false, only_st = true, only_ch = false, only_light = true;
								TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

								std::string group = make_group_name_QED("light_st", mu_val, 0, 0, sign_idx - 1, 0);
								std::string full_group = "QED/" + group;
								corr.setGroups(full_group.c_str());
								THREAD(corr.writeFile(twop_filename, corr_file_format));
							}
						}

						// CHARM
						{
							PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
							TIME(prop_u.copy(*props_u[imu][1], HOST); prop_u.load());
							TIME(prop_d.copy(*props_d[imu][1], HOST); prop_d.load());
							TIME(prop_s.copy(*props_s[1], HOST); prop_s.load());
							TIME(prop_c.copy(*props_c[sign_idx], HOST); prop_c.load());

							if (imu == 0) {	
								bool only_up = false, only_dn = false, only_st = false, only_ch = true, only_light = false;
								TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

								std::string group = make_group_name_QED("only_ch", mu_val, 0, 0, 0, sign_idx - 1);
								std::string full_group = "QED/" + group;
								corr.setGroups(full_group.c_str());
								THREAD(corr.writeFile(twop_filename, corr_file_format));
							} else {
								bool only_up = false, only_dn = false, only_st = false, only_ch = true, only_light = true;
								TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

								std::string group = make_group_name_QED("light_ch", mu_val, 0, 0, 0, sign_idx - 1);
								std::string full_group = "QED/" + group;
								corr.setGroups(full_group.c_str());
								THREAD(corr.writeFile(twop_filename, corr_file_format));
							}
						}
					}

					// --------- All combinations of +1 and -1 on 2 flavors -------------
					const int minus = 0, zero = 1, plus = 2; // Map indices to phases
					for (int i = 0; i < 4; i++) {
						for (int j = 0; j < 4; j++) {
							if (i >= j) continue;

							// Loop over all 4 sign combinations for flavors i and j
							const std::array<std::pair<int, int>, 4> sign_combinations = {
								std::make_pair(plus, plus),
								std::make_pair(plus, minus),
								std::make_pair(minus, plus),
								std::make_pair(minus, minus)
							};

							for (const auto& pair : sign_combinations) {
								int sign_i = pair.first;
								int sign_j = pair.second;
								std::array<int, 4> sign_idx = {zero, zero, zero, zero};
								sign_idx[i] = sign_i;
								sign_idx[j] = sign_j;

								TIME(prop_u.copy(*props_u[imu][sign_idx[0]], HOST); prop_u.load());
								TIME(prop_d.copy(*props_d[imu][sign_idx[1]], HOST); prop_d.load());
								TIME(prop_s.copy(*props_s[sign_idx[2]], HOST); prop_s.load());
								TIME(prop_c.copy(*props_c[sign_idx[3]], HOST); prop_c.load());

								PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);

								bool changed[4] = {
									sign_idx[0] != zero,
									sign_idx[1] != zero,
									sign_idx[2] != zero,
									sign_idx[3] != zero
								};

								std::vector<std::string> filtered_baryons;
								for (const std::string& b : BP_prop_prods) {
									std::set<char> flavors(b.begin(), b.end());
									if (imu > 0 && flavors.find('u') == flavors.end() && flavors.find('d') == flavors.end())
										continue;
									int distinct_changed = 0;	// Evaluate how many distinct flavors have changed from phase 0
									if (flavors.count('u') && changed[0]) distinct_changed++;
									if (flavors.count('d') && changed[1]) distinct_changed++;
									if (flavors.count('s') && changed[2]) distinct_changed++;
									if (flavors.count('c') && changed[3]) distinct_changed++;

									if (distinct_changed == 2)
										filtered_baryons.push_back(b); // save all baryons with exactly 2 distinct changed flavors
								}

								// std::string combined;
								// for (const auto& baryon : filtered_baryons) {
								// 	combined += baryon + " ";
								// }
								// PLEGMA_printf("imu=%lu, i=%d, j=%d, sign_i=%d, sign_j=%d | Filtered baryons: %s\n", imu, i, j, sign_i, sign_j, combined.c_str());

								TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, prop_c,
															false, false, false, false, false,
															&filtered_baryons));

								std::string group = make_group_name_QED("mixed", mu_val,
																		sign_idx[0] - 1,
																		sign_idx[1] - 1,
																		sign_idx[2] - 1,
																		sign_idx[3] - 1);
								std::string full_group = "QED/" + group;
								corr.setGroups(full_group.c_str());
								THREAD(corr.writeFile(twop_filename, corr_file_format));
							}
						}
					}
				}
			}

			//--------------------- SIB ---------------------
			if (run_SIB) {
				updateGaugeQuda(gauge, true);
				solver.UpdateSolver();

				assert(SIB_shift_sign == 1 || SIB_shift_sign == -1);

				double run_mu_st = (1 + SIB_shift_sign * 0.005) * mu_s[0];
				double run_mu_ch = (1 + SIB_shift_sign * 0.005) * mu_c[0];
				TIME(computePropagator(propSIBST, run_mu_st, STRANGE, nsmearGauss_s));
				TIME(computePropagator(propSIBCH, run_mu_ch, CHARM, nsmearGauss_c));

				TIME(prop_s.copy(*props_s[1], HOST));
				prop_s.load();
				TIME(prop_c.copy(*props_c[1], HOST));
				prop_c.load();

				for (size_t imu = 0; imu < mu_l.size(); imu++) {
					double run_mu = mu_l[imu];
					PLEGMA_printf("[SIB] Running inversion for mu = %.6e\n", run_mu);
					double run_mu_up = (1 + SIB_shift_sign * 0.005) * run_mu;
					double run_mu_dn = -(1 + SIB_shift_sign * 0.005) * run_mu;
					if (run_ud) {
						TIME(computePropagator(propSIBUP, run_mu_up, LIGHT, nsmearGauss));
						TIME(computePropagator(propSIBDN, run_mu_dn, LIGHT, nsmearGauss));
					}

					{
						TIME(prop_u.copy(*props_u[imu][1], HOST));
						prop_u.load();
						TIME(prop_d.copy(*props_d[imu][1], HOST));
						prop_d.load();

						
						std::string group;
						std::string full_group;

						PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);

						for (int flavor = 0; flavor < 4; flavor++) {
							if (!run_heavy_SIB && flavor > 1) { // Skip strange and charm SIB if not running heavy SIB
								continue;
							}

							bool only_up = false, only_dn = false, only_st = false, only_ch = false, only_light = false;

							// Modify the respective propagator
							switch (flavor) {
								case 0:
									only_up = true;
									TIME(corr.contractBaryonsUDSC(propSIBUP, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

									group = make_group_name_SIB("only_up", run_mu_up, -run_mu, mu_s[0], mu_c[0]);
									full_group = "SIB/" + group;
									corr.setGroups(full_group.c_str());
									THREAD(corr.writeFile(twop_filename, corr_file_format));
									break;
								case 1:
									only_dn = true;
									TIME(corr.contractBaryonsUDSC(prop_u, propSIBDN, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));

									group = make_group_name_SIB("only_dn", run_mu, run_mu_dn, mu_s[0], mu_c[0]);
									full_group = "SIB/" + group;
									corr.setGroups(full_group.c_str());
									THREAD(corr.writeFile(twop_filename, corr_file_format));
									break;
								case 2:
									if (imu == 0) {
										only_st = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, propSIBST, prop_c, only_up, only_dn, only_st, only_ch, only_light));

										group = make_group_name_SIB("only_st", run_mu, -run_mu, run_mu_st, mu_c[0]);
										full_group = "SIB/" + group;
										corr.setGroups(full_group.c_str());
										THREAD(corr.writeFile(twop_filename, corr_file_format));
										break;
									} else {
										only_st = true, only_light = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, propSIBST, prop_c, only_up, only_dn, only_st, only_ch, only_light));

										group = make_group_name_SIB("light_st", run_mu, -run_mu, run_mu_st, mu_c[0]);
										full_group = "SIB/" + group;
										corr.setGroups(full_group.c_str());
										THREAD(corr.writeFile(twop_filename, corr_file_format));
										break;
									}
								case 3:
									if (imu == 0) {
										only_ch = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, propSIBCH, only_up, only_dn, only_st, only_ch, only_light));

										group = make_group_name_SIB("only_ch", run_mu, -run_mu, mu_s[0], run_mu_ch);
										full_group = "SIB/" + group;
										corr.setGroups(full_group.c_str());
										THREAD(corr.writeFile(twop_filename, corr_file_format));
										break;
									} else {
										only_ch = true, only_light = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, propSIBCH, only_up, only_dn, only_st, only_ch, only_light));

										group = make_group_name_SIB("light_ch", run_mu, -run_mu, mu_s[0], run_mu_ch);
										full_group = "SIB/" + group;
										corr.setGroups(full_group.c_str());
										THREAD(corr.writeFile(twop_filename, corr_file_format));
										break;
									}
							}
						}
					}
				}
			}

			// === Critical Mass Correction ===
			if (run_CM) {
				for (size_t idk = 0; idk < dks.size(); idk++) {
					double dk = dks[idk];
					kappa = (1 + dk) * kappa0;
					solver.UpdateSolver();

					PLEGMA_printf("[CM] Running inversion with kappa = %.6e\n", kappa);

					double run_mu_st = mu_s[0];
					double run_mu_ch = mu_c[0];
					TIME(computePropagator(propCMST, run_mu_st, STRANGE, nsmearGauss_s));
					TIME(computePropagator(propCMCH, run_mu_ch, CHARM, nsmearGauss_c));

					for (size_t imu = 0; imu < mu_l.size(); ++imu) {
						double run_mu = mu_l[imu];
						TIME(computePropagator(propCMUP, run_mu, LIGHT, nsmearGauss));
						TIME(computePropagator(propCMDN, -run_mu, LIGHT, nsmearGauss));

						TIME(prop_u.copy(*props_u[imu][1], HOST));
						prop_u.load();
						TIME(prop_d.copy(*props_d[imu][1], HOST));
						prop_d.load();
						TIME(prop_s.copy(*props_s[1], HOST));
						prop_s.load();
						TIME(prop_c.copy(*props_c[1], HOST));
						prop_c.load();

						for (int flavor = 0; flavor < 4; flavor++) {
							if (!run_heavy_CM && flavor > 1) { // Skip strange and charm CM if not running heavy CM
								continue;
							}

							bool only_up = false, only_dn = false, only_st = false, only_ch = false, only_light = false;
							std::string group, full_group;
							PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);

							switch (flavor) {
								case 0:
									only_up = true;
									TIME(corr.contractBaryonsUDSC(propCMUP, prop_d, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));
									group = make_group_name_CM("only_up", run_mu, -run_mu, run_mu_st, run_mu_ch, dk);
									break;
								case 1:
									only_dn = true;
									TIME(corr.contractBaryonsUDSC(prop_u, propCMDN, prop_s, prop_c, only_up, only_dn, only_st, only_ch, only_light));
									group = make_group_name_CM("only_dn", run_mu, -run_mu, run_mu_st, run_mu_ch, dk);
									break;
								case 2:
									if (imu == 0) {
										only_st = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, propCMST, prop_c, only_up, only_dn, only_st, only_ch, only_light));
										group = make_group_name_CM("only_st", run_mu, -run_mu, run_mu_st, run_mu_ch, dk);
										break;
									} else {
										only_st = true, only_light = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, propCMST, prop_c, only_up, only_dn, only_st, only_ch, only_light));
										group = make_group_name_CM("light_st", run_mu, -run_mu, run_mu_st, run_mu_ch, dk);
										break;
									}
								case 3:
									if (imu == 0) {
										only_ch = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, propCMCH, only_up, only_dn, only_st, only_ch, only_light));
										group = make_group_name_CM("only_ch", run_mu, -run_mu, run_mu_st, run_mu_ch, dk);
										break;
									} else {
										only_ch = true, only_light = true;
										TIME(corr.contractBaryonsUDSC(prop_u, prop_d, prop_s, propCMCH, only_up, only_dn, only_st, only_ch, only_light));
										group = make_group_name_CM("light_ch", run_mu, -run_mu, run_mu_st, run_mu_ch, dk);
										break;
									}
							}
							full_group = "CM/" + group;
							corr.setGroups(full_group.c_str());
							THREAD(corr.writeFile(twop_filename, corr_file_format));
						}
					}
				}
			}
		}
	}
	while(not threads.empty()) {threads.back().join(); threads.pop_back();}

	//finalize();
	return 0;
}