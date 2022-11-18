#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss","momlist-filename","momlisttwopt-filename","momlistthreept-filename",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks","Projs", "threep-filename","confnumber"};


void produceOutput( PLEGMA_ScattCorrelator<float> source,
                      std::string outputFilename,
                      std::string diagram_name,
                      int n_stochastic_samples,
                      int n_coherent_source,
                      int *coherent_source_table_timeslice ){
    source.apply_phase();
    source.apply_sign(diagram_name);
    source.applyBoundaryConditions( true ,  n_coherent_source, coherent_source_table_timeslice);
    source.normalize_nstoch(n_stochastic_samples);
    source.writeHDF5( outputFilename );

  }



void produceOutput( PLEGMA_ScattCorrelator<float> source,
                      std::string outputFilename,
                      std::string diagram_name,
                      int n_coherent_source,
                      int *coherent_source_table_timeslice
                    ){
    source.apply_phase();
    source.apply_sign(diagram_name);
    source.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice );
    source.writeHDF5( outputFilename );

  }

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                      std::string outputFilename,
                      std::string diagram_name
                    ){
    source.apply_phase();
    source.apply_sign(diagram_name);
    source.applyBoundaryConditions( true );
    source.writeHDF5( outputFilename );
  }

  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::vector<double> mu_s;
  std::vector<double> mu_c;
  int rand_seed1=1234;
  int confnumber_int;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int nsmearGauss_c = 0;
  int startSource = 0;
  int nroots=4;
  std::string prOrNt = "neutron";
  std::string srcInputFile = "./input.src";
  std::string outdiagramPrefix="";
  std::string outfilename;

  std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5};

  std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
  std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

  std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> gammas_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};//,S12,S13,S23,S41,S42,S43};

  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4};




  auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
    options.set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    options.set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
		     };
  add_options(*HGC_options);
  if(prOrNt != "proton" && prOrNt != "neutron") PLEGMA_error("This exec is only for nucleon, %s is not allowed",prOrNt.c_str());
  //=========================================================================================================//
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();
      
      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();

      // Gauge for contractions
      contractGauge.copy(gauge);
      // apply boundary conditions since is needed for the covariant derivative
      applyBoundaryConditions(contractGauge,true);
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));

    std::string given_twop_filename = twop_filename;
    std::string given_threep_filename = threep_filename;

    //Get the confnumber for latfile
    char *ssource;
    asprintf(&ssource,"%04d", confnumber_int);
    std::string confnumber= ssource;
    free(ssource);


    //Reading the momentum lists
    PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
    momList sourcemomentumList_threept(4,pathListMomenta_threept,{1,2,3,});
    PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList_threept.size());
    //We have four types of momenta in the list here
    //The first three entries are the pi2 pion source momentum
    //The second three entries are the pf1 nucleon sink momentum
    //The third three entries are the pf2 pion sink momentum (it is assumed that pf2 is the same as -pf1)
    //The fourth three entries are the pc momentum at the insertion
    //pf1 + pf2 = 0, momentum at the sink is zero , the the pion momentum
    //at sink follows from the nucleon momentum
    //In addition the following momentum conversations are imposed
    //pi1 + pi2 = pc so the momentum phase factor is calculated as pc-pi2
    //We define the list of momenta such that the total momentum should be the [2] \
    //column, the pc, and of coarse it is understand that this refers to the source only.
    //At the sink we have always zero momentum

    

    momList sourcemomentumList_twopt(3,pathListMomenta_twopt,{1,2,});
    //For the twopoint functions we have also three momentum
    //first is pi2
    //second is pf1
    //third is pf2
    //and in this case the total momentum is defined as the sum of pf1 and pf2
    //to get pi1 we have to subtract pi2 from the total momentum


    if(sourcemomentumList_twopt.empty())
     PLEGMA_error("twopt momentumList empty");
    if(sourcemomentumList_threept.empty())
     PLEGMA_error("threept momentumList empty");

    PLEGMA_Vector<double> vectorSource_oet;
    vectorSource_oet.randInit(rand_seed1);

    
    for(int isource = startSource; isource < numSourcePositions; isource++){

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      std::vector<std::vector<int>> mpi2_twopt = sourcemomentumList_twopt.uniq_p(0);
      momList list_mpi2_twopt(1,{mpi2_twopt,},{0,});

      std::vector<std::vector<int>> mpi2_threept = sourcemomentumList_threept.uniq_p(0);
      momList list_mpi2_threept(1,{mpi2_threept,},{0,});

      std::vector<std::vector<int>> mpf1_threept = sourcemomentumList_threept.uniq_p(1);
      momList list_mpf1_threept(1,{mpf1_threept,},{0,});

      std::vector<std::vector<int>> mpf1_twopt = sourcemomentumList_twopt.uniq_p(1);
      momList list_mpf1_twopt(1,{mpf1_twopt,},{0,});


      PLEGMA_ScattCorrelator<float> corrP0UP(sourcePositions[isource], list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrP0DN(sourcePositions[isource], list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrPPUP(sourcePositions[isource], list_mpi2_twopt);//this is pi^{+}pi^{+}
      PLEGMA_ScattCorrelator<float> corrPPDN(sourcePositions[isource], list_mpi2_twopt);//this is pi^{-}pi^{-}

      corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
      corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
      corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
      corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");

      vectorSource_oet.stochastic_Z(nroots);

      site& source = sourcePositions[isource];

      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto computeOetPropagator = [&](PLEGMA_Vector<float>& vec_SS, PLEGMA_Vector<float>& vec_SL,
                                   double run_mu, WHICHFLAVOR fl, int nSmear,  std::vector<int> sourceMom, bool finalize) {
	                         // ensuring mu value
                                 if(mu != run_mu) {
                                   updateOptions(fl);
                                   mu = run_mu;
                                   solver.UpdateSolver();
                                 }

	              		 PLEGMA_Vector<double> vectorInOut;

                                 {  // absorbing the source and put momentum to the sink
                                   PLEGMA_Vector3D<double> vector1;
				                                      
				   PLEGMA_Vector3D<double> vector2;


                                   vector1.absorb(vectorSource_oet,sourcePositions[isource][DIM_T]);

				   vector1.mulMomentumPhases(sourceMom,+1);//Here we use the convention momentum +pi at thye source
				                                           //Note that this is automatically satisfied for the threept 
									   //when the momenta is applied at the right hand side, so
									   //not int the sequential case this refers to the meson 3pt function

				   TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));

                                   vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
                                 }
				 {
                                    PLEGMA_Vector<double> vectorAuxD;
                                    TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)));
                                    TIME(vectorInOut.copy(vectorAuxD));
                                 }


                                 TIME(solver.solve(vectorInOut, vectorInOut));

				 {
                                    PLEGMA_Vector<double> vectorAuxD;
                                    TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)));
                                    TIME(vectorInOut.copy(vectorAuxD));
                                 }


				 if(vec_SL.getAllocation() != NONE) {
				   PLEGMA_Vector<float> vectorAuxF;
				   vectorAuxF.copy(vectorInOut);
				   vec_SL.copy(vectorAuxF);
				 }

				 { // Smearing the solution
				   PLEGMA_Vector<double> vectorAuxD;
				   PLEGMA_Vector<float> vectorAuxF;
				   TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
				   vectorAuxF.copy(vectorAuxD);
				   vec_SS.copy(vectorAuxF);
				 }

      };

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear,  bool finalize) {
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
//				   if (iproj>7)
		                   {
                                    PLEGMA_Vector<double> vectorAuxD;
                                    TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)));
                                    TIME(vectorInOut.copy(vectorAuxD));
                                   }

				   TIME(solver.solve(vectorInOut, vectorInOut));
//				   if (iproj>7)
//				   
				  
	                           {
                                    PLEGMA_Vector<double> vectorAuxD;
                                    TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)));
                                    TIME(vectorInOut.copy(vectorAuxD));
                                   }
				  

				   if(prop_SL.getAllocation() != NONE) {
				     PLEGMA_Vector<float> vectorAuxF;
				     vectorAuxF.copy(vectorInOut);
				     prop_SL.absorb(vectorAuxF, isc/3, isc%3);
				   }
				   { // Smearing the solution
				     PLEGMA_Vector<double> vectorAuxD;
				     PLEGMA_Vector<float> vectorAuxF;
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
				     vectorAuxF.copy(vectorAuxD);
				     prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				   }
				 }
				 if(finalize) {
				   prop_SS.rotateToPhysicalBase_device(run_mu/abs(run_mu));
				   prop_SS.applyBoundaries_device(source[DIM_T]);
				 }
      };
      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string + ".h5";
      threep_filename = given_threep_filename + src_string;
      free(src_string);
      
      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propDN;
      
      PLEGMA_Vector<float> oet_mom_zero_up_SS;
      PLEGMA_Vector<float> oet_mom_zero_dn_SS;

      std::vector<PLEGMA_Vector<float>*> oet_mom_fini_up_SS;
      std::vector<PLEGMA_Vector<float>*> oet_mom_fini_dn_SS;

      int length_fini_mom=mpi2_twopt.size();
      for(int i=0; i< length_fini_mom; ++i) {
        oet_mom_fini_up_SS.push_back(new PLEGMA_Vector<float>(HOST));
        oet_mom_fini_dn_SS.push_back(new PLEGMA_Vector<float>(HOST));
      }


      PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1_twopt);
      TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"));

      PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1_twopt);
      TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"));




      { // Whithin this scope we keep track also of the propagator non smeared on the sink
	PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE);
        PLEGMA_Propagator<float> propDN_SL(tSinks.size()>0 ? BOTH:NONE);

        PLEGMA_Vector<float> oet_mom_zero_up_SL(tSinks.size()>0 ? BOTH:NONE);
        PLEGMA_Vector<float> oet_mom_zero_dn_SL(tSinks.size()>0 ? BOTH:NONE);

        std::vector<PLEGMA_Vector<float>*> oet_mom_fini_up_SL;
	std::vector<PLEGMA_Vector<float>*> oet_mom_fini_dn_SL;

	for(int i=0; i< length_fini_mom; ++i) {
          oet_mom_fini_up_SL.push_back(new PLEGMA_Vector<float>(HOST));
          oet_mom_fini_dn_SL.push_back(new PLEGMA_Vector<float>(HOST));
        }


	bool computed_light = false;
	// If twop_filename exists we hold the computation of the light props
	if(access( twop_filename.c_str(), F_OK ) == -1) {
	  TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
	  TIME(computePropagator(propDN, propDN_SL, -mu_ud, LIGHT, nsmearGauss, false));
	  computed_light = true;
	}

        bool computed_light_oet = true;
	std::vector<int> zero_mom({0,0,0});
        TIME(computeOetPropagator(oet_mom_zero_up_SS, oet_mom_zero_up_SL,  mu_ud, LIGHT, nsmearGauss, zero_mom, false));
        TIME(computeOetPropagator(oet_mom_zero_dn_SS, oet_mom_zero_dn_SL, -mu_ud, LIGHT, nsmearGauss, zero_mom, false));



	//Computing T reductions+recombination
        {
          PLEGMA_ScattCorrelator<float> reductionsT1N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
          PLEGMA_ScattCorrelator<float> reductionsT2N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
          //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
          //PLEGMA_printf("Nucleon T2 reduction\n");
          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
          //PLEGMA_printf("Nucleon T2 reduction ready\n");
          TIME(corrNP.N_diagrams( reductionsT1N, reductionsT2N ));
          //PLEGMA_printf("Nucleon diagram ready\n");

	}

	//Computing T reductions+recombination
        {
          PLEGMA_ScattCorrelator<float> reductionsT1N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
          PLEGMA_ScattCorrelator<float> reductionsT2N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
          //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN));
          //PLEGMA_printf("Nucleon T2 reduction\n");
          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN));
          //PLEGMA_printf("Nucleon T2 reduction ready\n");
          TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ));
          //PLEGMA_printf("Nucleon diagram ready\n");

	}

        //Section M diagrams 2pt functions 
	{
          PLEGMA_Vector<float> vectorAuxF_SS;
	  PLEGMA_Vector<float> vectorAuxF_SL;
          for(int i_pi2=0; i_pi2<mpi2_twopt.size(); ++i_pi2){

            auto &momentum_i2 =  mpi2_twopt[i_pi2];

            momList filtered_sourcemomentumList_twopoint = sourcemomentumList_twopt.extract(momentum_i2, 0);

            PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList_twopoint);//Proton piplus

            PLEGMA_ScattCorrelator<float> corrD1if12(sourcePositions[isource], filtered_sourcemomentumList_twopoint);//Proton pizero up
            PLEGMA_ScattCorrelator<float> corrD1if34(sourcePositions[isource], filtered_sourcemomentumList_twopoint);//Proton pizero dn
            PLEGMA_ScattCorrelator<float> corrD1if56(sourcePositions[isource], filtered_sourcemomentumList_twopoint);//Neutron piplus

	    corrM.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "MNPPP");

            corrD1if34.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP01");

            corrD1if12.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP02");

            corrD1if56.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MN0PP");


            TIME(computeOetPropagator(vectorAuxF_SS, vectorAuxF_SL, mu_ud, LIGHT, nsmearGauss, momentum_i2, false));

	    TIME(corrM.M_diagrams( corrNP, vectorAuxF_SS, oet_mom_zero_up_SS ));

	    TIME(corrD1if34.M_diagrams( corrNP, vectorAuxF_SS, oet_mom_zero_dn_SS ));

            TIME(corrD1if56.M_diagrams( corrN0, vectorAuxF_SS, oet_mom_zero_up_SS ));

	    TIME(corrPPUP.P_diagrams( vectorAuxF_SS, oet_mom_zero_up_SS, i_pi2, true));
	    TIME(corrP0UP.P_diagrams( vectorAuxF_SS, oet_mom_zero_dn_SS, i_pi2, true));

            vectorAuxF_SS.unload();
            oet_mom_fini_up_SS[i_pi2]->copy(vectorAuxF_SS,HOST);
            vectorAuxF_SS.load();

            vectorAuxF_SL.unload();
            oet_mom_fini_up_SL[i_pi2]->copy(vectorAuxF_SL,HOST);
            vectorAuxF_SL.load();


	    TIME(computeOetPropagator(vectorAuxF_SS, vectorAuxF_SL, -mu_ud, LIGHT, nsmearGauss, momentum_i2, false));
	    TIME(corrP0DN.P_diagrams( vectorAuxF_SS, oet_mom_zero_up_SS, i_pi2, true));
            TIME(corrPPDN.P_diagrams( vectorAuxF_SS, oet_mom_zero_dn_SS, i_pi2, true));


            TIME(corrD1if12.M_diagrams( corrNP, vectorAuxF_SS, oet_mom_zero_up_SS));

            vectorAuxF_SS.unload();
            oet_mom_fini_dn_SS[i_pi2]->copy(vectorAuxF_SS,HOST);
            vectorAuxF_SS.load();

            vectorAuxF_SL.unload();
            oet_mom_fini_dn_SL[i_pi2]->copy(vectorAuxF_SL,HOST);
            vectorAuxF_SL.load();

            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";

	    TIME(produceOutput(corrM, outfilename,"4pt",1,NULL)); //Proton pi plus
            TIME(produceOutput(corrD1if12, outfilename,"4pt",1,NULL));//Proton pi zero up
            TIME(produceOutput(corrD1if34, outfilename,"4pt",1,NULL));//Proton pi zero dn
            TIME(produceOutput(corrD1if56, outfilename,"4pt",1,NULL));//Neutron pi plus

	  } //end of loop mpi2
	}//section M diagram 2pt functions

	            

#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK

        std::vector<int> filter={0,0,0};
        momList filtered_sourcemomentumList_pi20 = sourcemomentumList_threept.extract(filter, 0);

	for(size_t its = 0; its < tSinks.size(); its++){
	  int tsinkMtsource = tSinks[its];
	  if(tsinkMtsource >= HGC_totalL[3])
	    PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	  int signPer = (tsinkMtsource+source[3]) >= HGC_totalL[3] ? -1 : +1;
	  int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3]; 


	  //Correlators for storing up and down insertion between the nucleon

  	  PLEGMA_ScattCorrelator<float> corrUp(source,  filtered_sourcemomentumList_pi20, tsinkMtsource+1 );
          PLEGMA_ScattCorrelator<float> corrDn(source,  filtered_sourcemomentumList_pi20, tsinkMtsource+1 );

  	  corrUp.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon, gammas_insertion, "M"+prOrNt+"Up");

  	  corrDn.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon, gammas_insertion, "M"+prOrNt+"Dn");

	  //3D propagators at t_sink
	  PLEGMA_Propagator3D<float> propUP3D;
	  PLEGMA_Propagator3D<float> propDN3D;
	  PLEGMA_Gauge3D<double> smearedGauge3D_sink;
	  propUP3D.absorb(propUP, global_fixSinkTime);
	  propDN3D.absorb(propDN, global_fixSinkTime);
	  smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

	  WHICHPARTICLE nucleon = get_particle(prOrNt); 
	  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4};//,S12,S13,S23,S41,S42,S43};
          
	  for (int alpha=0;alpha<N_SPINS; ++alpha){
            for (int beta=0; beta<N_SPINS; ++beta){

	      auto computeThreep = [&](double run_mu, PLEGMA_Propagator3D<float>& prop1, PLEGMA_Propagator3D<float>& prop2, int signProps, PLEGMA_Propagator<float> &propF, std::string fl) {
	        if(not computed_light) {
		  TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
		  TIME(computePropagator(propDN, propDN_SL, -mu_ud, LIGHT, nsmearGauss, false));
		  propUP3D.absorb(propUP, global_fixSinkTime);
		  propDN3D.absorb(propDN, global_fixSinkTime);
		  computed_light = true;
	        }
	        PLEGMA_Propagator<float> seqProp;
	        // ensuring mu positive
	        if(mu != run_mu) {
		  updateOptions(LIGHT);
		  mu = run_mu;
		  solver.UpdateSolver();
	        }


	        for(int i_pf1=0; i_pf1<mpf1_threept.size(); ++i_pf1){

                  auto &momentum_f1 =  mpf1_threept[i_pf1];


	          std::string filename = threep_filename + "_P" +std::to_string(alpha)+std::to_string(beta) + "_dt" + std::to_string(tsinkMtsource) + "_" + fl + "pf_x"+std::to_string(momentum_f1[0])+"_y"+std::to_string(momentum_f1[1])+"_z"+std::to_string(momentum_f1[2])+".h5";
	          if(access( filename.c_str(), F_OK ) != -1) {
		    PLEGMA_printf("File %s already exists. Skipping...", filename.c_str());
		    return;
	          }

                  momList filtered_sinkList = sourcemomentumList_threept.extract(momentum_f1, 1);

				     
	          for(int nu = 0 ; nu < 4 ; nu++){
		    for(int c2 = 0 ; c2 < 3 ; c2++){
		      PLEGMA_Vector<double> vectorInOut;
		      {
		        PLEGMA_Vector3D<double> vectorAuxD1,vectorAuxD2;
		        PLEGMA_Vector3D<float> vectorAuxF;
		        if(&prop1 != &prop2)
		          vectorAuxF.seqSourceNucleon(prop1, prop2, get_projector(alpha,beta), nucleon, nu, c2);
		        else
		          vectorAuxF.seqSourceNucleon(prop1, get_projector(alpha,beta), nucleon, nu, c2);
					 
		        // put a momentum in the sink
                        vectorAuxF.mulMomentumPhases(momentum_f1,-1);//At the sink the momenta should be -
			                                             //We have initially -
								     //The we have conjugation twice
								     //so eventually we have -
		        vectorAuxF.conjugate();
		        vectorAuxF.apply_gamma(G5);
		        vectorAuxD1.copy(vectorAuxF);
		        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nsmearGauss, alphaGauss));
		        vectorInOut.absorb(vectorAuxD2, global_fixSinkTime);
		      }
		      double norm = vectorInOut.norm();
		      vectorInOut.scale(1/norm);
		      if (get_projector(alpha,beta)>7)
		      {
	                PLEGMA_Vector<double> vectorAuxD;
	                int sgn=run_mu/fabs(run_mu);
                        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,sgn));
                        TIME(vectorInOut.copy(vectorAuxD));
		      }
		    
		      TIME(solver.solve(vectorInOut, vectorInOut));
		      if (get_projector(alpha,beta)>7){
                        PLEGMA_Vector<double> vectorAuxD;
	                int sgn=run_mu/(fabs(run_mu));
                        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,sgn));
                        TIME(vectorInOut.copy(vectorAuxD));
		      }
		    
		      vectorInOut.scale(norm);
		      PLEGMA_Vector<float> vectorAuxF;
		      vectorAuxF.copy(vectorInOut);
		      seqProp.absorb(vectorAuxF, nu, c2);
		    }//c2 color seq

		  }//nu alpha seq

                  seqProp.apply_gamma(G5);
	          
		  seqProp.conjugate();

		  std::vector<std::vector<int>> mpc = filtered_sinkList.uniq_p(3);
	    	  momList list_mpc(1,{mpc,},{0,});
				     
	          PLEGMA_ScattCorrelator<float> corr( source, list_mpc, tsinkMtsource+1);

    		  // LOCAL contractions
		  if (get_projector(alpha,beta)<8){
                    TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
		  }
		  else{
                    TIME(corr.contractNucleonThrp_local(seqProp, propF, 0, gammas));
                  }
	          if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
//                THREAD(corr.writeFile(filename, corr_file_format));

		  if (fl=="up"){
                    corrUp.absorbSourceSinkSpinMom(corr, alpha, beta, i_pf1 );
		  }
		  else{
      	            corrDn.absorbSourceSinkSpinMom(corr, alpha, beta, i_pf1 );
		  }
	          
				     
                  // ONED contractions
	          // TIME(corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas));
	          // if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	          // THREAD(corr.writeFile( filename, corr_file_format));
				     
	          // noe contractions
 	          // TIME(corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps));
	          // if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	          // THREAD(corr.writeFile( filename, corr_file_format));
		  //
	        }//momentum list
	      };
	      if(nucleon == PROTON) {
	        TIME(computeThreep(-mu_ud, propUP3D, propDN3D, +1, propUP_SL, "up"));
	        TIME(computeThreep( mu_ud, propUP3D, propUP3D, -1, propDN_SL, "dn"));
	      } else {
	        TIME(computeThreep( mu_ud, propDN3D, propUP3D, -1, propDN_SL, "dn"));
	        TIME(computeThreep(-mu_ud, propDN3D, propDN3D, +1, propUP_SL, "up"));
	      }
	    } //loop over beta
	  }//loop over alpha

           
          auto computeOetInvThroughSink = [&](PLEGMA_Vector<float>& vec_SC, double run_mu, PLEGMA_Vector3D<float>& prop, int nSmear, WHICHFLAVOR fl, std::vector<int> momentum_f1, int i_gamma_i2, int i_gamma_f2 ) {
               // ensuring mu positive
               if(mu != run_mu) {
                  updateOptions(fl);
                  mu = run_mu;
                  solver.UpdateSolver();
               }

               PLEGMA_Vector<double> vectorInOut;
               {
                  PLEGMA_Vector3D<double> vectorAuxD1, vectorAuxD2;
                  PLEGMA_Vector3D<float> vectorAuxF;
                  vectorAuxF.copy(prop);
                  vectorAuxF.mulMomentumPhases(momentum_f1,-1); // put momentum at the sink
			                                        //Here the momentum phase should be - in the end
			                                        //We have initially -pf1 so +pf2
								//We have a conjugation in the sequential source
								//So in the end we have a minus result
                  vectorAuxD1.copy(vectorAuxF);
                  TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nSmear, alphaGauss));
                  vectorInOut.absorb(vectorAuxD2, global_fixSinkTime);
               }

               GAMMAS_SCATT G_i2= apply_g5( glist_source_meson[i_gamma_i2], LEFT );
               GAMMAS_SCATT G_f2= apply_g5( glist_sink_meson[i_gamma_f2], RIGHT );

               vectorInOut.apply_gamma_scatt(G_i2,RIGHT);
               vectorInOut.apply_gamma_scatt(G_f2,LEFT);

               vectorInOut.apply_gamma(G5);

               double norm = vectorInOut.norm();
               vectorInOut.scale(1/norm);
               {
                  PLEGMA_Vector<double> vectorAuxD;
                  TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)));
                  vectorInOut.copy(vectorAuxD);
               }

               TIME(solver.solve(vectorInOut, vectorInOut));

               {
                  PLEGMA_Vector<double> vectorAuxD;
                  TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)));
                  vectorInOut.copy(vectorAuxD);
               }

               vectorInOut.scale(norm);
               PLEGMA_Vector<float> vectorAuxF;
               vectorAuxF.copy(vectorInOut);
               vec_SC.copy(vectorAuxF);
                
               vec_SC.apply_gamma(G5);
	  };

          auto computeThreep_contract = [&](PLEGMA_ScattCorrelator<float> corrScatt, PLEGMA_Vector<float>& seqProp, PLEGMA_Vector<float> &propF, int i_gamma_i2, int i_gamma_f2, int i_pf1){

               std::vector<std::vector<int>> mpc = sourcemomentumList_threept.uniq_p(3);
               momList list_mpc(1,{mpc,},{0,});

	       seqProp.conjugate();

	       PLEGMA_ScattCorrelator<float> corr(source, list_mpc, tsinkMtsource+1);
	       TIME(corr.contractNucleonThrp_local(seqProp, propF, 0, gammas, false));

               corrScatt.absorbGammai2Gammaf2momentumf2(corr, i_gamma_i2, i_gamma_f2, i_pf1 );

          };

          auto computeThreep_meson = [&](double run_mu, PLEGMA_Vector3D<float>& prop, int nSmear, WHICHFLAVOR fl, std::string flstring, std::string name) {
	       std::vector<PLEGMA_Vector<float>*>  inversionThroughSink;
 
	       std::vector<std::vector<int>> pf1_filt= sourcemomentumList_threept.uniq_p(1);
               int length_fini_mom=pf1_filt.size();
               int momgammaif=length_fini_mom*glist_source_meson.size()*glist_sink_meson.size();

               for(int i=0; i< momgammaif; ++i) {
	         inversionThroughSink.push_back(new PLEGMA_Vector<float>(HOST));
               }
	       {
                 PLEGMA_Vector<float> vectorAuxF_SL;
                 for (int i_pf1=0; i_pf1<pf1_filt.size(); ++i_pf1 ){
                   auto &momentum_f1 =  pf1_filt[i_pf1];
                   for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2){
	             for (int i_gamma_f2=0; i_gamma_f2 < glist_sink_meson.size(); ++i_gamma_f2){
	     	       TIME(computeOetInvThroughSink(vectorAuxF_SL, run_mu, prop, nSmear, LIGHT, momentum_f1, i_gamma_i2, i_gamma_f2 ));
		       vectorAuxF_SL.unload();
                       int index=i_pf1*glist_source_meson.size()*glist_sink_meson.size();
                       inversionThroughSink[index]->copy( vectorAuxF_SL, HOST);
                       vectorAuxF_SL.load();
                     }
	           }
	         }

	       }
	  
               for(int i_pi2=0; i_pi2<mpi2_threept.size(); ++i_pi2){
                 auto &momentum_i2 =  mpi2_threept[i_pi2];
                 momList filtered_sourcemomentumList = sourcemomentumList_twopt.extract(momentum_i2, 0);

                 PLEGMA_ScattCorrelator<float> corrMP(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
                 PLEGMA_ScattCorrelator<float> corrMN(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);

		 std::string label;
		 if ((flstring == "up") && (name == "pizero")){
		   label="UUJUU_UP";
		 }
		 else if ((flstring == "dn") && (name == "pizero")){
                   label="DDJDD_DN";
		 }
		 else if ((flstring == "up") && (name == "piplus")){
                   label="DUJUD_UP";
		 }
		 else if ((flstring == "dn") && (name == "piplus")){
                   label="DUJUD_DN";
                 }

     	         corrMP.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "32", "Mproton"+label);
 
 		 corrMN.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "32", "Mneutron"+label);


                 PLEGMA_ScattCorrelator<float> corrpjp(source,  filtered_sourcemomentumList, tsinkMtsource+1 );	    
                 corrpjp.initialize_diagram(glist_source_meson, glist_sink_meson, gammas_insertion, "PJP_STL");

	         PLEGMA_Vector<float> oet_fini_SL;
                 oet_fini_SL.unload();
	         if ((flstring == "up") ){
		   oet_fini_SL.copy(*oet_mom_fini_up_SL[i_pi2],HOST);
		 } 
                 else {
                   oet_fini_SL.copy(*oet_mom_fini_dn_SL[i_pi2],HOST);
		 }
	         oet_fini_SL.load();

                 for (int i_pf1=0; i_pf1<pf1_filt.size(); ++i_pf1 ){
                   auto &momentum_f1 =  pf1_filt[i_pf1];
                   for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2){
                     for (int i_gamma_f2=0; i_gamma_f2 < glist_sink_meson.size(); ++i_gamma_f2){
	               PLEGMA_Vector<float> sequential;
		       sequential.unload();
		       int index=i_pf1*glist_source_meson.size()*glist_sink_meson.size();
                       sequential.copy(*inversionThroughSink[index],HOST);
		       sequential.load();
                       TIME(computeThreep_contract(corrpjp, sequential, oet_fini_SL, i_gamma_i2, i_gamma_f2, i_pf1 ));
		     }
		   }
		 }
	       
                 float *sinkTimeSliceProton;
                 sinkTimeSliceProton=corrNP.get_time_slice(global_fixSinkTime);
                 //PLEGMA_printf("nucleon zero component %e %d \n",sinkTimeSliceProton[0], global_fixSinkTime);
	         TIME(corrMP.M_diagrams( corrNP, corrpjp, sinkTimeSliceProton ));
	         free(sinkTimeSliceProton);

	         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"nucleonpionJpion";
	         TIME(corrMP.apply_phase());
                 TIME(corrMP.apply_sign("NPJP"));
                 TIME(corrMP.applyBoundaryConditions(true));
                 TIME(corrMP.writeHDF5(outfilename));

                 sinkTimeSliceProton=corrN0.get_time_slice(global_fixSinkTime);
                 //PLEGMA_printf("nucleon zero component %e %d \n",sinkTimeSliceProton[0], global_fixSinkTime);
                 TIME(corrMN.M_diagrams( corrN0, corrpjp, sinkTimeSliceProton ));
                 free(sinkTimeSliceProton);

                 outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"nucleonpionJpion";
                 TIME(corrMN.apply_phase());
                 TIME(corrMN.apply_sign("NPJP"));
                 TIME(corrMN.applyBoundaryConditions(true));
                 TIME(corrMN.writeHDF5(outfilename));

//                 asprintf(&ssource,"pi2x%02dpi2y%02dpi2z%02d", momentum_i2[0], momentum_i2[1], momentum_i2[2]);
//                 std::string momtext= (std::string)"_" + ssource;
//                 free(ssource);


	         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"pionJpion"+label;
                 TIME(corrpjp.writeHDF5(outfilename));
	       }

	  };

          for(int i_pi2=0; i_pi2<mpi2_threept.size(); ++i_pi2){
            auto &momentum_i2 =  mpi2_threept[i_pi2];
            momList filtered_sourcemomentumList = sourcemomentumList_threept.extract(momentum_i2, 0);
	  
            PLEGMA_ScattCorrelator<float> corrM1(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM2(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM3(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM4(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM5(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM6(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);

            PLEGMA_ScattCorrelator<float> corrM7(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM8(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
	    PLEGMA_ScattCorrelator<float> corrM9(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);
            PLEGMA_ScattCorrelator<float> corrM10(sourcePositions[isource], filtered_sourcemomentumList, tsinkMtsource+1);


            if (nucleon==PROTON){
	      corrM1.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "12", "MNJNUU_UP");
              corrM2.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "12", "MNJNUU_DN");
              corrM3.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "12", "MNJNDD_UP");
              corrM4.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "12", "MNJNDD_DN");
	      corrM5.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "32", "MNJNDU_UP");
              corrM6.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "32", "MNJNDU_DN");


	    } else {
              corrM7.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "12", "MNJNUD_UP");
              corrM8.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, gammas_insertion, "12", "MNJNUD_DN");

	    }
 
            if (nucleon==PROTON){
              PLEGMA_Vector<float> oet_fini_up;
              oet_fini_up.unload();
	      oet_fini_up.copy(*oet_mom_fini_up_SS[i_pi2], HOST);
              oet_fini_up.load();

	      PLEGMA_Vector<float> oet_fini_dn;
              oet_fini_dn.unload();
              oet_fini_dn.copy(*oet_mom_fini_dn_SS[i_pi2], HOST);
              oet_fini_dn.load();

              TIME(corrM1.M_diagrams( corrUp, oet_fini_up, oet_mom_zero_dn_SS));
              TIME(corrM2.M_diagrams( corrDn, oet_fini_up, oet_mom_zero_dn_SS));
              TIME(corrM3.M_diagrams( corrUp, oet_fini_dn, oet_mom_zero_up_SS));
              TIME(corrM4.M_diagrams( corrDn, oet_fini_dn, oet_mom_zero_up_SS));
              TIME(corrM5.M_diagrams( corrUp, oet_fini_up, oet_mom_zero_up_SS));
              TIME(corrM6.M_diagrams( corrDn, oet_fini_up, oet_mom_zero_up_SS));


            
	      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protonup_pizeroup";
	      TIME(corrM1.apply_sign("NJNP")); 
              TIME(corrM1.apply_phase());
	      TIME(corrM1.writeHDF5(outfilename));

	      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protondn_pizeroup";
              TIME(corrM2.apply_sign("NJNP"));
              TIME(corrM2.apply_phase());
              TIME(corrM2.writeHDF5(outfilename));

              outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protonup_pizerodn";
              TIME(corrM3.apply_sign("NJNP"));
              TIME(corrM3.apply_phase());
              TIME(corrM3.writeHDF5(outfilename));

	      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protondn_pizerodn";

	      TIME(corrM4.apply_sign("NJNP"));
              TIME(corrM4.apply_phase());
              TIME(corrM4.writeHDF5(outfilename));

              outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protonup_piplusdn";
              TIME(corrM5.apply_sign("NJNP"));
              TIME(corrM5.apply_phase());
              TIME(corrM5.writeHDF5(outfilename));

              outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protondn_piplusdn";
	      TIME(corrM6.apply_sign("NJNP"));
              TIME(corrM6.apply_phase());
              TIME(corrM6.writeHDF5(outfilename));


	    }
	    else{

              PLEGMA_Vector<float> oet_fini_up;
              oet_fini_up.unload();
              oet_fini_up.copy(*oet_mom_fini_up_SS[i_pi2], HOST);
              oet_fini_up.load();

              TIME(corrM7.M_diagrams( corrUp, oet_mom_zero_up_SS, oet_fini_up ));
              TIME(corrM8.M_diagrams( corrDn, oet_mom_zero_up_SS, oet_fini_up ));

              outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"neutronup_piplus";
              TIME(corrM7.apply_sign("NJNP"));
              TIME(corrM7.apply_phase());
              TIME(corrM7.writeHDF5(outfilename));

	      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"neutrondn_piplus";
              TIME(corrM8.apply_sign("NJNP"));
              TIME(corrM8.apply_phase());
              TIME(corrM8.writeHDF5(outfilename));

             
	    }

	  }//momentum pi2

          //Note that we compute NJN for the set 
	  //of input momenta pf1 and perform the 
	  //fourier transform for the set of momenta
	  //pc, however pi is not neccessarily pf1-pc, so
	  //we do not apply here the momentum phase at
	  //the source, this hast to be done in postproduction
          //TIME(corrUp.apply_phase());

          //TIME(corrDn.apply_phase());

	  if (nucleon==PROTON){
           outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protonup";
          }
	  else{
           outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"neutronup";
	  }
          TIME(corrUp.writeHDF5(outfilename));

          if (nucleon==PROTON){
            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"protondn";
          }
          else{
           outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"neutrondn";
          }
          TIME(corrDn.writeHDF5(outfilename));	  

          {
	  PLEGMA_Vector3D<float> zero_momentum_light;
          zero_momentum_light.absorb(oet_mom_zero_up_SS, global_fixSinkTime);
	  computeThreep_meson(-mu_ud, zero_momentum_light, nsmearGauss, LIGHT,"up", "piplus");

	  zero_momentum_light.absorb(oet_mom_zero_dn_SS, global_fixSinkTime);
          computeThreep_meson(+mu_ud, zero_momentum_light, nsmearGauss, LIGHT,"dn", "piplus");

          zero_momentum_light.absorb(oet_mom_zero_dn_SS, global_fixSinkTime);
          computeThreep_meson(-mu_ud, zero_momentum_light, nsmearGauss, LIGHT,"up", "pizero");

          zero_momentum_light.absorb(oet_mom_zero_up_SS, global_fixSinkTime);
          computeThreep_meson(+mu_ud, zero_momentum_light, nsmearGauss, LIGHT,"dn", "pizero");
  	  }

#endif

        }//tsink


        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
        TIME(corrPPUP.apply_sign("P"));
        TIME(corrPPUP.writeHDF5( outfilename ));
        TIME(corrPPDN.apply_sign("P"));
        TIME(corrPPDN.writeHDF5( outfilename ));
        TIME(corrP0UP.apply_sign("P"));
        TIME(corrP0UP.writeHDF5( outfilename ));
        TIME(corrP0DN.apply_sign("P"));
        TIME(corrP0DN.writeHDF5( outfilename ));

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_N";
        TIME( corrN0.apply_phase());
        TIME( corrN0.apply_sign("N"));
        TIME( corrN0.applyBoundaryConditions( true ));
        TIME( corrN0.writeHDF5(outfilename));

        TIME( corrNP.apply_phase() );
        TIME( corrNP.apply_sign("N") );
        TIME( corrNP.applyBoundaryConditions( true ) );
        TIME( corrNP.writeHDF5(outfilename) );
        
      }  //keep track of
    }//source positions
  }//loop in finalize
  finalize();
  return 0;
}
