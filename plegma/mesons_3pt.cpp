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
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss","momlist-filename",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks", "threep-filename"};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_s;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int startSource = 0;
  int nroots=4;
  int rand_seed1=1234;
  int confnumber_int;
  int stoch_std=true;

  std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
  std::vector<GAMMAS_SCATT> glist_source_meson={G_5};
  std::vector<GAMMAS_SCATT> gammas_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};//,S12,S13,S23,S41,S42,S43};


  std::string srcInputFile = "./input.src";
  std::string outfilename;
  std::string outdiagramPrefix="";

  std::vector<int> sourceMom = {0,0,0};
    auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("source-mom", "The list of momenta components at the source. Every three makes a momentum", verbosity, sourceMom);
    options.set("stoch_std", "stoch_std = true calculates the meson 3pt function using the oet, otherwise we use the standard point to all method", verbosity, stoch_std);
    options.set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    options.set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);



		     };
  add_options(*HGC_options);
  //=========================================================================================================//
  // Only multiple of three accepted
  assert(sourceMom.size()%3==0);
  
  initializePLEGMA();

  PLEGMA_Vector<double> vectorSource_oet;
  vectorSource_oet.randInit(rand_seed1);


  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH,FIRST_CORNER);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
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


    //Reading the momentum lists
    //we have here three momenta 
    //pi1 meson momentum at the source
    //pc momentum at the insertion
    //pf meson momentum at the sink
    PLEGMA_printf("###Momentum list read from : %s", pathListMomenta.c_str());
    momList sourcemomentumList(2,pathListMomenta,{0,1});

    PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList.size());

  //Get the confnumber for latfile
    char *ssource;
    asprintf(&ssource,"%04d", confnumber_int);
    std::string confnumber= ssource;
    free(ssource);

    for(int isource = startSource; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);


      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
      momList list_mpi2(1,{mpi2,},{0,});
      PLEGMA_ScattCorrelator<float> corrP0DN(sourcePositions[isource], list_mpi2);
      PLEGMA_ScattCorrelator<float> corrPPDN(sourcePositions[isource], list_mpi2);
      PLEGMA_ScattCorrelator<float> corrP0UP(sourcePositions[isource], list_mpi2);
      PLEGMA_ScattCorrelator<float> corrPPUP(sourcePositions[isource], list_mpi2);
      PLEGMA_ScattCorrelator<float> corrKAON(sourcePositions[isource], list_mpi2);

      corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
      corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
      corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
      corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");
      corrKAON.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");


      vectorSource_oet.stochastic_Z(nroots);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto computePropagator_oet = [&](PLEGMA_Vector<float>& vec_SS, PLEGMA_Vector<float>& vec_SL,
                                   double run_mu, WHICHFLAVOR fl, int nSmear0, int nSmear1,  std::vector<int> sourceMom) {
                                 // ensuring mu value
                                 if(mu != run_mu) {
                                   updateOptions(fl);
                                   mu = run_mu;
                                   solver.UpdateSolver();
                                 }

                                 PLEGMA_Vector<double> vectorInOut;


                                 {  // absorbing the source and put momentum to the sink
                                   PLEGMA_Vector3D<double> vector1,vector2;
                                   vector1.absorb(vectorSource_oet,sourcePositions[isource][DIM_T]);
                                   vector1.mulMomentumPhases(sourceMom,-1);

				   TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear0, alphaGauss));


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
                                   TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear1, alphaGauss));
                                   vectorAuxF.copy(vectorAuxD);
                                   vec_SS.copy(vectorAuxF);
                                 }

      };

#if 0
      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear0, int nSmear1) {
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
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear0, alphaGauss));
				     vectorInOut.absorb(vector2,source[DIM_T]);
				   }
				   // Inverting
				   PLEGMA_printf("Going to invert %s for component %d\n",
						 fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   if(prop_SL.getAllocation() != NONE) {
				     PLEGMA_Vector<float> vectorAuxF;
				     vectorAuxF.copy(vectorInOut);
				     prop_SL.absorb(vectorAuxF, isc/3, isc%3);
				   }
				   { // Smearing the solution
				     PLEGMA_Vector<double> vectorAuxD;
				     PLEGMA_Vector<float> vectorAuxF;
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear1, alphaGauss));
				     vectorAuxF.copy(vectorAuxD);
				     prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				   }
				 }
			       };
#endif
      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string;
      threep_filename = given_threep_filename + src_string;
      free(src_string);
#if 0      
      PLEGMA_Propagator<float> propUP_wrong_smear;
      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propST_wrong_smear;
      PLEGMA_Propagator<float> propST;
#endif
      PLEGMA_Vector<float> oet_light_up_fini_SS;
      PLEGMA_Vector<float> oet_light_dn_fini_SS;
      PLEGMA_Vector<float> oet_strange_up_fini_SS;
      PLEGMA_Vector<float> oet_strange_dn_fini_SS;

      PLEGMA_Vector<float> oet_light_up_zero_SS;
      PLEGMA_Vector<float> oet_light_dn_zero_SS;
      PLEGMA_Vector<float> oet_strange_up_zero_SS;
      PLEGMA_Vector<float> oet_strange_dn_zero_SS;

      { // Whithin this scope we keep track also of the propagator non smeared on the sink
	//PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
	//PLEGMA_Propagator<float> propST_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
//	PLEGMA_Propagator<float> propUP_SL;//(BOTH, FIRST_CORNER);
//	PLEGMA_Propagator<float> propST_SL;//(BOTH, FIRST_CORNER);

   	PLEGMA_Vector<float> oet_light_up_fini_SL(BOTH, FIRST_CORNER);
	PLEGMA_Vector<float> oet_light_dn_fini_SL(BOTH, FIRST_CORNER);
        PLEGMA_Vector<float> oet_strange_up_fini_SL(BOTH, FIRST_CORNER);
        PLEGMA_Vector<float> oet_strange_dn_fini_SL(BOTH, FIRST_CORNER);


        PLEGMA_Vector<float> oet_light_up_zero_SL(NONE);
        PLEGMA_Vector<float> oet_light_dn_zero_SL(NONE);
        PLEGMA_Vector<float> oet_strange_up_zero_SL(NONE);
	PLEGMA_Vector<float> oet_strange_dn_zero_SL(NONE);

	if (stoch_std ==true){

          std::vector<int> zero_mom({0,0,0});

          TIME(computePropagator_oet(oet_light_up_zero_SS, oet_light_up_zero_SL,  mu_ud, LIGHT, nsmearGauss, nsmearGauss, zero_mom));

//        oet_light_up_zero_SS.writeHDF5("oet_light_UP_SS_zero");


          TIME(computePropagator_oet(oet_light_dn_zero_SS, oet_light_dn_zero_SL, -mu_ud, LIGHT, nsmearGauss, nsmearGauss,  zero_mom));

// oet_light_dn_zero_SS.writeHDF5("oet_light_DN_SS_zero");


          TIME(computePropagator_oet(oet_strange_up_zero_SS, oet_strange_up_zero_SL, mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s, zero_mom));

	  //oet_strange_up_zero_SS.writeHDF5("oet_strange_UP_SS_zero");


          TIME(computePropagator_oet(oet_strange_dn_zero_SS, oet_strange_dn_zero_SL, -mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s, zero_mom));

          //oet_strange_dn_zero_SS.writeHDF5("oet_strange_DN_SS_zero");

        } 
#if 0    
    	else {
	  TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, nsmearGauss));
	  TIME(computePropagator(propST, propST_SL, mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s));
	
	  // Correcting the smearing
	  for(int isc = 0 ; isc < 12 ; isc++){
	    PLEGMA_Vector<double> vectorAuxD, vectorAuxD2;
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxF.absorb(propUP_SL, isc/3, isc%3);
	    vectorAuxD2.copy(vectorAuxF);
	    TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss_s, alphaGauss));
	    vectorAuxF.copy(vectorAuxD);
	    propUP_wrong_smear.absorb(vectorAuxF, isc/3, isc%3);
	  }
	  // Correcting the smearing
  	  for(int isc = 0 ; isc < 12 ; isc++){
	    PLEGMA_Vector<double> vectorAuxD, vectorAuxD2;
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxF.absorb(propST_SL, isc/3, isc%3);
	    vectorAuxD2.copy(vectorAuxF);
	    TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
	    vectorAuxF.copy(vectorAuxD);
	    propST_wrong_smear.absorb(vectorAuxF, isc/3, isc%3);
	  }
	}
#endif

	

	std::vector<std::vector<int>> pi1_filt = sourcemomentumList.uniq_p(0);
	  
	for (int i_pi1=0; i_pi1<pi1_filt.size();++i_pi1){


          auto &momentum_i1 =  pi1_filt[i_pi1];
	  //printf("Momentum I1 %d %d %d\n",momentum_i1[0],momentum_i1[1], momentum_i1[2]);
	  momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i1, 0);

          PLEGMA_ScattCorrelator<float> corr_local(source,  filtered_sourcemomentumList );
          //PLEGMA_ScattCorrelator<float> corr_local2(source,  filtered_sourcemomentumList );

          PLEGMA_ScattCorrelator<float> corr_oneD( source,  filtered_sourcemomentumList );
          PLEGMA_ScattCorrelator<float> corr_twoD( source,  filtered_sourcemomentumList );


          corr_local.initialize_diagram(glist_source_meson, glist_sink_meson, gammas_insertion, "PJP_STL");
          //corr_local2.initialize_diagram(glist_source_meson, glist_sink_meson, gammas_insertion, "PJP_STL");


          corr_oneD.initialize_diagram( glist_source_meson, glist_sink_meson, gammas_insertion, "PJP_STD");
          //corr_twoD.initialize_diagram( glist_source_meson, glist_sink_meson, gammas_insertion, "PJP_TWOD");

          if (stoch_std==true){
    	    TIME(computePropagator_oet(oet_light_up_fini_SS, oet_light_up_fini_SL,  mu_ud, LIGHT, nsmearGauss, nsmearGauss, momentum_i1));
            //oet_light_up_fini_SS.writeHDF5("oet_light_UP_SS_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));
            //oet_light_up_fini_SL.writeHDF5("oet_light_UP_SL_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));

            TIME(computePropagator_oet(oet_light_dn_fini_SS, oet_light_dn_fini_SL,  -mu_ud, LIGHT, nsmearGauss, nsmearGauss, momentum_i1));
            //oet_light_dn_fini_SS.writeHDF5("oet_light_DN_SS_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));
            //oet_light_dn_fini_SL.writeHDF5("oet_light_DN_SL_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));

	    TIME(computePropagator_oet(oet_strange_up_fini_SS, oet_strange_up_fini_SL, mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s, momentum_i1));

            //oet_strange_up_fini_SS.writeHDF5("oet_strange_UP_SS_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));
            //oet_strange_up_fini_SL.writeHDF5("oet_strange_UP_SL_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));


            TIME(computePropagator_oet(oet_strange_dn_fini_SS, oet_strange_dn_fini_SL, -mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s, momentum_i1));
              //oet_strange_dn_fini_SS.writeHDF5("oet_strange_DN_SS_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));
              //oet_strange_dn_fini_SL.writeHDF5("oet_strange_DN_SL_mom_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));

	  }


          TIME(corrPPUP.P_diagrams( oet_light_up_zero_SS, oet_light_up_fini_SS, i_pi1, false));
          TIME(corrPPDN.P_diagrams( oet_light_dn_zero_SS, oet_light_dn_fini_SS, i_pi1, false));
          TIME(corrP0UP.P_diagrams( oet_light_dn_zero_SS, oet_light_up_fini_SS, i_pi1, false));
          TIME(corrP0DN.P_diagrams( oet_light_up_zero_SS, oet_light_dn_fini_SS, i_pi1, false));
	  TIME(corrKAON.P_diagrams( oet_light_up_zero_SS, oet_strange_up_fini_SS, i_pi1, false)); 


	  for(size_t its = 0; its < tSinks.size(); its++){
	    int tsinkMtsource = tSinks[its];
	    if(tsinkMtsource >= HGC_totalL[3])
	      PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	    int signPer = (tsinkMtsource+source[3]) >= HGC_totalL[3] ? -1 : +1;
	    int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3]; 

	    // 3D propagators at t_sink

	    PLEGMA_Gauge3D<double> smearedGauge3D_sink;
#if 0
	    PLEGMA_Propagator3D<float> propUP3D;
	    PLEGMA_Propagator3D<float> propST3D;

	    if (stoch_std ==false){
              propUP3D.absorb(propUP, global_fixSinkTime);
	      propST3D.absorb(propST, global_fixSinkTime);
	    }
#endif
	    smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

	    std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4};



	    std::vector<std::vector<int>> pf1_filt = filtered_sourcemomentumList.uniq_p(2);


            for(int i_pf1=0; i_pf1<pf1_filt.size(); ++i_pf1){

              auto &momentum_f1 =  pf1_filt[i_pf1];

#if 0
	      auto computeThreep = [&](double run_mu, PLEGMA_Propagator3D<float>& prop, PLEGMA_Propagator<float> &propF, int nSmear, WHICHFLAVOR fl, std::string name) {
         	char * mom_string;
	        asprintf(&mom_string, "_mx%+dmy%+dmz%+d", momentum_f1[0], momentum_f1[1], momentum_f1[2]);
	        std::string filename = threep_filename + mom_string + "_dt" + std::to_string(tsinkMtsource)+name;
	        free(mom_string);
	      
  	        int signProps = -run_mu/abs(run_mu);		   
	        PLEGMA_Propagator<float> seqProp(BOTH, FIRST_CORNER);
	        // ensuring mu positive
	        if(mu != run_mu) {
		  updateOptions(fl);
		  mu = run_mu;
		  solver.UpdateSolver();
	        }
				     
	        for(int nu = 0 ; nu < 4 ; nu++)
		  for(int c2 = 0 ; c2 < 3 ; c2++){
		    PLEGMA_Vector<double> vectorInOut;
		    {
		      PLEGMA_Vector3D<double> vectorAuxD1, vectorAuxD2;
		      PLEGMA_Vector3D<float> vectorAuxF;
		      vectorAuxF.absorb(prop, nu, c2);
		      vectorAuxF.apply_gamma(G5);
		      vectorAuxF.mulMomentumPhases(momentum_f1,+1); // put momentum at the sink
		      std::complex<float> Isingle(0,1);
		      float phase = 2.*PI*(((float) momentum_i1[0] * source[0])/HGC_totalL[0]
					 + ((float) momentum_i1[1] * source[1])/HGC_totalL[1]
					 + ((float) momentum_i1[2] * source[2])/HGC_totalL[2]);
		      vectorAuxF.cscale(std::exp<float>(-phase*Isingle)); // put momentum from the point source
		      vectorAuxD1.copy(vectorAuxF);
		      TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nSmear, alphaGauss));
		      vectorInOut.absorb(vectorAuxD2, global_fixSinkTime);
		    }
		    double norm = vectorInOut.norm();
		    vectorInOut.scale(1/norm);
		    TIME(solver.solve(vectorInOut, vectorInOut));
		    vectorInOut.scale(norm);
		    PLEGMA_Vector<float> vectorAuxF;
		    vectorAuxF.copy(vectorInOut);
		    seqProp.absorb(vectorAuxF, nu, c2);
		  }
	        seqProp.apply_gamma(G5);
	        seqProp.conjugate();

                std::vector<std::vector<int>> mpc = filtered_sourcemomentumList.uniq_p(2);
                momList list_mpc(1,{mpc,},{0,});

				     
	        PLEGMA_ScattCorrelator<float> corr(source, list_mpc, tsinkMtsource+1);
	  
	        // LOCAL contractions
	        TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
	        if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
	        TIME(corr.writeFile(filename, corr_file_format));			     
	        // ONED contractions
	        TIME(corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas));
	        if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	        TIME(corr.writeFile( filename, corr_file_format));
				     
	        // noe contractions
	        TIME(corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps));
	        if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	        TIME(corr.writeFile( filename, corr_file_format));
	      
	        // TWOD contractions
	        TIME(corr.contractNucleonThrp_twoD(seqProp, propF, contractGauge, signProps, gammas));
	        if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	        TIME(corr.writeFile( filename, corr_file_format));
	      };
#endif
	      auto computeThreep_oet = [&](double run_mu, PLEGMA_Vector3D<float>& prop, PLEGMA_Vector<float> &propF, int nSmear, WHICHFLAVOR fl, std::string name) {
                 PLEGMA_Vector<float> seqProp(BOTH, FIRST_CORNER);
                 // ensuring mu positive

 		 int signProps = -run_mu/abs(run_mu);

                 if(mu != run_mu) {
                   updateOptions(fl);
                   mu = run_mu;
                   solver.UpdateSolver();
                 }

                 for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2){
                   for (int i_gamma_f2=0; i_gamma_f2 < glist_sink_meson.size(); ++i_gamma_f2){

                     PLEGMA_Vector<double> vectorInOut;
                     {
                       PLEGMA_Vector3D<double> vectorAuxD1, vectorAuxD2;
                       PLEGMA_Vector3D<float> vectorAuxF;
                       vectorAuxF.copy(prop);

                       vectorAuxF.mulMomentumPhases(momentum_f1,-1); // put momentum at the sink
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
                     seqProp.copy(vectorAuxF);
                     seqProp.apply_gamma(G5);

                     //seqProp.writeHDF5("oet_seq_"+name+std::to_string(momentum_f1[0])+std::to_string(momentum_f1[1])+std::to_string(momentum_f1[2]));

                     std::vector<std::vector<int>> mpc = filtered_sourcemomentumList.uniq_p(1);
                     momList list_mpc(1,{mpc,},{0,});

		     PLEGMA_ScattCorrelator<float> corr4(source, list_mpc, tsinkMtsource+1);
                     TIME(corr4.contractMesonThrp_local(seqProp, propF,  gammas_insertion));
                     if(signPer < 0) for(size_t iv = 0 ; iv < corr4.getTotalSize()*2; iv++) corr4.H_elem()[iv] *= signPer;
                     //PLEGMA_printf("LOCAL size %d\n",corr1.getTotalSize());
                     //corr_local2.absorbGammai2Gammaf2momentumf2(corr4, i_gamma_i2, i_gamma_f2, i_pf1 );


		     seqProp.conjugate();


                     //seqProp.conjugate();
 

		     //local contractions
                     PLEGMA_ScattCorrelator<float> corr1(source, list_mpc);//, tsinkMtsource+1);
                     TIME(corr1.contractNucleonThrp_local(seqProp, propF, 0, gammas, false));
                     if(signPer < 0) for(size_t iv = 0 ; iv < corr1.getTotalSize()*2; iv++) corr1.H_elem()[iv] *= signPer;
		     //PLEGMA_printf("LOCAL size %d\n",corr1.getTotalSize());
                     corr_local.absorbGammai2Gammaf2momentumf2(corr1, i_gamma_i2, i_gamma_f2, i_pf1 );

//                     corr1.writeHDF5(threep_filename+name+std::to_string(i_gamma_i2)+"_"+std::to_string(i_gamma_f2)+"_"+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));


		     // ONED contractions
                     PLEGMA_ScattCorrelator<float> corr2(source, list_mpc);//, tsinkMtsource+1);
                     TIME(corr2.contractNucleonThrp_oneD(seqProp, propF, contractGauge,0, gammas));
                     if(signPer < 0) for(size_t iv = 0 ; iv < corr2.getTotalSize()*2; iv++) corr2.H_elem()[iv] *= signPer;
                     //PLEGMA_printf("ONED size %d\n",corr2.getTotalSize());


 //                    corr2.writeHDF5(threep_filename+name+std::to_string(i_gamma_i2)+"_"+std::to_string(i_gamma_f2)+"_"+std::to_string(momentum_i1[0])+std::to_string(momentum_i1[1])+std::to_string(momentum_i1[2]));
		     corr_oneD.absorbGammai2Gammaf2momentumf2(corr2, i_gamma_i2, i_gamma_f2, i_pf1 );
		              
	      	     // TWOD contractions
                     /*PLEGMA_ScattCorrelator<float> corr3(source, list_mpc, tsinkMtsource+1);
     		     TIME(corr3.contractNucleonThrp_twoD(seqProp, propF, contractGauge, 0, gammas));
                     if(signPer < 0) for(size_t iv = 0 ; iv < corr3.getTotalSize()*2; iv++) corr3.H_elem()[iv] *= signPer;
                     corr_twoD.absorbGammai2Gammaf2momentumf2(corr2, i_gamma_i2, i_gamma_f2, i_pf1 );*/

                     //TIME(corr.writeFile( filename, corr_file_format));



                   }//igamma1
                 }//igamma2

              };

	    if (stoch_std == true){

              PLEGMA_Vector3D<float> zero_momentum_light;
	      zero_momentum_light.absorb(oet_light_up_zero_SS, global_fixSinkTime);

              TIME(computeThreep_oet(-mu_ud, zero_momentum_light, oet_light_up_fini_SL, nsmearGauss, LIGHT, "_up_pion"));

              std::string filename;
	      filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
	      //TIME(corr_local2.apply_sign("P"));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion2_local";
              //TIME(corr_local2.writeFile( filename, corr_file_format));

              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion_twoD";
              //TIME(corr_twoD.writeFile( filename, corr_file_format));


	      /*
	      zero_momentum_light.absorb(oet_light_dn_zero_SS, global_fixSinkTime);

              TIME(computeThreep_oet(mu_ud, zero_momentum_light, oet_light_dn_fini_SL, nsmearGauss, LIGHT, "_up_pion_inverted"));

              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion_inverted_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion_inverted_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_pion_inverted_twoD";
              TIME(corr_twoD.writeFile( filename, corr_file_format));*/


              PLEGMA_Vector3D<float> zero_momentum_strange;
              zero_momentum_strange.absorb(oet_strange_up_zero_SS,global_fixSinkTime);

              TIME(computeThreep_oet(-mu_ud, zero_momentum_strange, oet_light_up_fini_SL, nsmearGauss, LIGHT, "_up_kaon"));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_kaon_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_kaon_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_kaon_twoD";
              //TIME(corr_twoD.writeFile( filename, corr_file_format));


	      zero_momentum_strange.absorb(oet_strange_dn_zero_SS,global_fixSinkTime);

              TIME(computeThreep_oet(mu_ud, zero_momentum_strange, oet_light_dn_fini_SL, nsmearGauss, LIGHT, "_up_kaon_inverted"));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_kaon_inverted_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_kaon_inverted_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_up_kaon_inverted_twoD";
              //TIME(corr_twoD.writeFile( filename, corr_file_format));



              zero_momentum_light.absorb(oet_light_dn_zero_SS,global_fixSinkTime);

              TIME(computeThreep_oet(mu_ud, zero_momentum_light, oet_light_dn_fini_SL, nsmearGauss, LIGHT, "_dn_pion"));

              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_dn_pion_local";
	      TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_dn_pion_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_dn_pion_twoD";
              //TIME(corr_twoD.writeFile( filename, corr_file_format));

              /*
	      zero_momentum_light.absorb(oet_light_up_zero_SS,global_fixSinkTime);

              TIME(computeThreep_oet(-mu_ud, zero_momentum_light, oet_light_up_fini_SL, nsmearGauss, LIGHT, "_dn_pion_inverted"));

              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_dn_pion_inverted_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_dn_pion_inverted_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_dn_pion_inverted_twoD";
              TIME(corr_twoD.writeFile( filename, corr_file_format));*/


	      zero_momentum_light.absorb(oet_light_dn_zero_SS,global_fixSinkTime);

              TIME(computeThreep_oet(mu_s, zero_momentum_light, oet_strange_dn_fini_SL, nsmearGauss, STRANGE, "_st_kaon"));

              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_st_kaon_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_st_kaon_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_st_kaon_twoD";
              //TIME(corr_twoD.writeFile( filename, corr_file_format));


              zero_momentum_light.absorb(oet_light_up_zero_SS,global_fixSinkTime);

              TIME(computeThreep_oet(-mu_s, zero_momentum_light, oet_strange_up_fini_SL, nsmearGauss, STRANGE, "_st_kaon_inverted"));

              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_st_kaon_inverted_local";
              TIME(corr_local.writeFile( filename, corr_file_format));
              filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_st_kaon_inverted_oneD";
              TIME(corr_oneD.writeFile( filename, corr_file_format));
              //filename = threep_filename + "_dt" + std::to_string(tsinkMtsource)+"_st_kaon_inverted_twoD";
              //TIME(corr_twoD.writeFile( filename, corr_file_format));


	    } //stochastic standard
#if 0	    
	    else{
	      TIME(computeThreep(-mu_ud, propUP3D, propUP_SL, nsmearGauss, LIGHT, "_up_pion"));
	      TIME(computeThreep(-mu_ud, propST3D, propUP_SL, nsmearGauss, LIGHT, "_up_kaon"));
	      TIME(computeThreep(-mu_s, propUP3D, propST_SL, nsmearGauss_s, STRANGE, "_st_kaon"));
	    } //std or oet
#endif	   
	  } //i pf momentum
	}// tsinks 
      } // i_pi momentum

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrPPUP.apply_sign("P"));
      TIME(corrPPUP.writeHDF5( outfilename ));
      TIME(corrPPDN.apply_sign("P"));
      TIME(corrPPDN.writeHDF5( outfilename ));
      TIME(corrP0UP.apply_sign("P"));
      TIME(corrP0UP.writeHDF5( outfilename ));
      TIME(corrP0DN.apply_sign("P"));
      TIME(corrP0DN.writeHDF5( outfilename ));

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_K";
      TIME(corrKAON.apply_sign("P"));
      TIME(corrKAON.writeHDF5( outfilename ));



      }
/*
      propUP_wrong_smear.rotateToPhysicalBase_device(mu_ud/abs(mu_ud));
      propST_wrong_smear.rotateToPhysicalBase_device(mu_s/abs(mu_s));
      propUP.rotateToPhysicalBase_device(mu_ud/abs(mu_ud));
      propST.rotateToPhysicalBase_device(mu_s/abs(mu_s));
      propUP_wrong_smear.applyBoundaries_device(source[3]);
      propST_wrong_smear.applyBoundaries_device(source[3]);
      propUP.applyBoundaries_device(source[3]);
      propST.applyBoundaries_device(source[3]);

      {
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	TIME(corr.contractMesonsNew(propUP, propUP));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_uu"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));

	TIME(corr.contractMesonsNew(propUP, propST));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_us"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractMesonsNew(propUP_wrong_smear, propST));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_u's"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractMesonsNew(propUP, propST_wrong_smear));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_us'"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));

      	TIME(corr.contractMesonsNew(propUP_wrong_smear, propST_wrong_smear));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_u's'"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
      }*/
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }
    
  finalize();
  return 0;
}
  
