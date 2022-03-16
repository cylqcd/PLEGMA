#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

  using namespace plegma;
  using namespace quda;

  std::vector<double> runtime_inside;
#define TIME_INSIDE(fnc)  runtime_inside.push_back(MPI_Wtime()); fnc; \
    PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime_inside.back()); \
    runtime_inside.pop_back()

  std::vector<double> runtime;
#define TIME(fnc,isospin)  runtime.push_back(MPI_Wtime()); fnc;\
    PLEGMA_printf("TIME "#isospin" for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
    runtime.pop_back()

  std::vector<std::thread> threads;
  //#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME_INSIDE(fnc); }))
  #define THREAD(fnc) TIME_INSIDE(fnc)

  extern int device;
  static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename","time-dilution","confnumber"};
  // Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

  void produceOutput( PLEGMA_ScattCorrelator<float> source,
		      std::string outputFilename,
		      std::string diagram_name,
		      int n_stochastic_samples,
		      int n_coherent_source,
		      int *coherent_source_table_timeslice ){
    TIME_INSIDE(source.apply_phase());
    TIME_INSIDE(source.apply_sign(diagram_name));
    TIME_INSIDE(source.applyBoundaryConditions( true ,  n_coherent_source, coherent_source_table_timeslice));
    TIME_INSIDE(source.normalize_nstoch(n_stochastic_samples));
    TIME_INSIDE(source.writeHDF5( outputFilename ));

  }


  void produceOutput( PLEGMA_ScattCorrelator<float> source,
		      std::string outputFilename,
		      std::string diagram_name,
		      int n_coherent_source, 
		      int *coherent_source_table_timeslice
		    ){
    TIME_INSIDE(source.apply_phase());
    TIME_INSIDE(source.apply_sign(diagram_name));
    TIME_INSIDE(source.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice ));
    TIME_INSIDE(source.writeHDF5( outputFilename ));

  }
  void produceOutput( PLEGMA_ScattCorrelator<float> source,
		      std::string outputFilename,
		      std::string diagram_name
		    ){
    TIME_INSIDE(source.apply_phase());
    TIME_INSIDE(source.apply_sign(diagram_name));
    TIME_INSIDE(source.applyBoundaryConditions( true ));
    TIME_INSIDE(source.writeHDF5( outputFilename ));
  }

  int main(int argc, char **argv)
  {
    initializeOptions(argc, argv, true, listOpt);
    //================ Add your options in this between initializeOptions and initializePLEGMA ================//
    double mu_ud = mu;
    double mu_ud_factor[QUDA_MAX_MG_LEVEL];
    for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
    bool timedilution;
    bool read_stochastic_oet;
    bool do_contraction_std;
    bool do_stochastic; 
    bool do_stochastic_oet;
    int n_stochastic_samples;
    int nroots=4;
    int confnumber_int;
    int rand_seed1=1234;
    int rand_seed2=1234;
    std::string outfilename;
    std::string outfile_V="";
    std::string outfile_upS="";
    std::string outfile_dnS="";
    std::string outfile_SEQ="";
    std::string outdiagramPrefix="";
    HGC_options->set("contractionstoch", "We are performing stochastic contraction for the piN-piN diagrams B,W,D1ii,D1ff,T", verbosity, do_stochastic);
    HGC_options->set("contractionstd", "We are performing std contractions for nucleon and delta in both isospin channels", verbosity, do_contraction_std);
    HGC_options->set("contractionoet", "We are performing oet contraction for Z diagrams and for pions", verbosity, do_stochastic_oet);
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    HGC_options->set("read_stochastic_oet", "Flag for switching read/building stochastic propagators", verbosity, read_stochastic_oet);
    HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);
    HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
    HGC_options->set("outPropUP", "Path for saving the up propagator used", verbosity, outfile_upS);
    HGC_options->set("outPropDN", "Path for saving the dn propagator used", verbosity, outfile_dnS);
    HGC_options->set("outPropSeq", "Path for saving the sequential propagator used", verbosity, outfile_SEQ);
    HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
    HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
    HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);

    PLEGMA_printf("Automatic seed mmm\n");

    //=========================================================================================================//
    initializePLEGMA();
    {
      /*******************************************************************************
       *
       *
       *  Initialization: (1) reading the sourcepositions and the source momentum list 
       *                  (2) reading the gaugefield uploading the quda
       *                      produce the smeared gauge field
       *                      only the smeared gauge field will be stored in PLEGMA
       *                  (3) setting up the list of gammas
       *                      (1) we use three triplets for the delta Cgi, Cgigt, Cgigtg5
       *                      (2) we use C,Cg5, Cg4,Cg5g4 for the nucleon
       *                      (3) we use g5 for meson(pion)
       *
       *******************************************************************************/
      //Storing only the smeared gauge
      PLEGMA_Gauge<double> smearedGauge;

      {
	// Reading from Lime file and loading to device
	PLEGMA_Gauge<double> gauge;
	gauge.readFile(latfile, LIME_FORMAT);
	gauge.calculatePlaq();

	// Loading to QUDA and computing plaquette also there
	initGaugeQuda(gauge, true);
	plaqQuda();

	// Smearing
	TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3),"ISOSPIN32");
	PLEGMA_printf("Plaquette after smearing:\n");
	smearedGauge.calculatePlaq();
      }


      updateOptions(LIGHT);
      TIME(QUDA_solver solver(mu),"ISOSPIN32");



      //Get the confnumber for latfile
      char *ssource;
      asprintf(&ssource,"%04d", confnumber_int);
      std::string confnumber= ssource;
      free(ssource);

      //Reading the momentum lists
      PLEGMA_printf("###Momentum list read from : %s", pathListMomenta.c_str());
      momList sourcemomentumList(3,pathListMomenta,{1,2});
      PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList.size());

      if(sourcemomentumList.empty())
	PLEGMA_error("momentumList empty");


      //List of gammas
      std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3};
      std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3};
      std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5};
      std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5};
      std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
      std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};

      std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
      std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

      std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
      std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

      
      std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
      std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

  /******************************************************************************************
  *
  *  In the first part of the code we compute oet propagators
  *      (1) we compute oet propagators for zero momentum for each timeslice
  *      (2) for the other momenta only timeslice to timeslice
  *  Note that in both cases we store a standard vector of PLEGMA_Vectors on the host, 
  *  and need to load to the device in case we need them
  *
  *
  *******************************************************************************************/

      PLEGMA_Vector<float> stochastic_oet_prop_d_fini_mom_source_to_sink;
      PLEGMA_Vector<float> stochastic_oet_prop_u_fini_mom_source_to_sink;




  /********************************************************************************************
  *
  *
  *   In the second part of the code we compute point source propagators for
  *   UP and DN flavour and perform all the contractions necessary for I=3/2 and 1/2
  *   that does not require sequential source propagator and allocate space for the 
  *   sequential propagator
  *   Producing diagrams (1) N (Nucleon 2pt) (including both N+ with U,D,U and N- with D,U,D
  *                      (2) D (Delta 2pt) 
  *                      (3) delta -->> pi + N (2pt)
  *   Producing factors requiring only D or U for momenta pf1,pf2 or {0,0,0}
  *
  *********************************************************************************************/

      PLEGMA_Vector<double> vectorSource_oet;
      vectorSource_oet.randInit(rand_seed2);
      vectorSource_oet.stochastic_Z(nroots);
      double tmp=vectorSource_oet.norm();
      PLEGMA_printf("Normtest12 %e\n", tmp);

      //loop over the soure positions
      for(int isource = 0 ; isource < numSourcePositions; isource++){

        site source=site({0,0,0,sourcePositions[isource][DIM_T]});

        site actualSource=site({sourcePositions[isource][DIM_X],
	                        sourcePositions[isource][DIM_Y],
	                        sourcePositions[isource][DIM_Z],
		                sourcePositions[isource][DIM_T]});



	//Calculations for source-position Calculations for source-position 
	PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		      isource, sourcePositions[isource][0], sourcePositions[isource][1],
		      sourcePositions[isource][2], sourcePositions[isource][3]);
   
	//Create Propagator
	PLEGMA_Propagator<float> propUP(BOTH); //To be saved for all the coherent sources.
	PLEGMA_Propagator<float> propDN(BOTH);

	std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
	momList list_mpf1(1,{mpf1,},{0,});
	PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1 );
	TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"),"ISOSPIN32");

	PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1 );
	TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"),"ISOSPIN12");

	asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
	std::string sourcepositiontext_inside= (std::string)"_" + ssource; 
	free(ssource);

        PLEGMA_Gauge3D<double> smearedGauge3D;
        smearedGauge3D.absorb(smearedGauge, sourcePositions[isource][DIM_T]);

	// ensuring mu positive
	if(mu<0) {
	  mu*=-1.;
	  solver.UpdateSolver();
	}

	site src;
	src=site({sourcePositions[isource][0],
		  sourcePositions[isource][1],
		  sourcePositions[isource][2],
		  sourcePositions[isource][3]});
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut;
	  PLEGMA_Vector<float>  vectorAuxF;
	  PLEGMA_Vector<double> vectorAuxD;
	  { // Smearing the source
	    PLEGMA_Vector3D<double> vector1, vector2;
	    vector1.pointSource(src, isc/3, isc%3, DEVICE);
	    TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
	    vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
	  }

	  //Rotation to the physical basis
	  TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1),"ISOSPIN32");       

	  //Inversion
	  PLEGMA_printf("Going to invert UP for component %d\n", isc);
	  TIME(solver.solve(vectorAuxD, vectorAuxD),"ISOSPIN32");

	  //Rotation to the physical basis
	  TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1),"ISOSPIN32");

	  //Smearing at the sink
	  TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

	  vectorAuxF.copy(vectorAuxD);
	  propUP.absorb(vectorAuxF, isc/3, isc%3);

	}
	if(outfile_upS!="")
	{
	    PLEGMA_printf("Save propagator for the up quark\n");
	    PLEGMA_Vector<float> vectorAuxPrint(BOTH);
	    for(int isc = 0 ; isc < 12 ; isc++){
	      std::string spin=std::to_string(isc/3);
	      std::string col=std::to_string(isc%3);

	      vectorAuxPrint.absorb(propUP,isc/3,isc%3);
	      vectorAuxPrint.unload();
	      vectorAuxPrint.writeLIME(outfile_upS+confnumber+sourcepositiontext_inside+"_s"+spin+"_c"+col);
	      //vectorAuxPrint.writeHDF5(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
	    }
	}
	// ensuring mu negative
	if(mu>0) {
	  mu*=-1.;
	  solver.UpdateSolver();
	}

	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut;
	  PLEGMA_Vector<float>  vectorAuxF;
	  PLEGMA_Vector<double> vectorAuxD;

	  {  // Smearing the source
	      PLEGMA_Vector3D<double> vector1, vector2;
	      vector1.pointSource(src, isc/3, isc%3, DEVICE);
	      TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
	      vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
	  }

	  //(3 step) rotation to the physical basis
	  TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1),"ISOSPIN32");

	  //(4 step) doing the inversion
	  PLEGMA_printf("Going to invert DN for component %d\n", isc);
	  TIME(solver.solve(vectorAuxD, vectorAuxD),"ISOSPIN32");

	  //(5 step) rotating to the physical base
	  TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1),"ISOSPIN32");

	  //(6 step) doing the smearing on the propagator
	  TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

	  vectorAuxF.copy(vectorAuxD);

	  propDN.absorb(vectorAuxF, isc/3, isc%3);
	}
	  
	if(outfile_dnS!="")
        {
            PLEGMA_printf("Save propagator for the dn quark\n");
            PLEGMA_Vector<float> vectorAuxPrint(BOTH);
            for(int isc = 0 ; isc < 12 ; isc++){
              std::string spin=std::to_string(isc/3);
              std::string col=std::to_string(isc%3);

              vectorAuxPrint.absorb(propDN,isc/3,isc%3);
              vectorAuxPrint.unload();
              vectorAuxPrint.writeLIME(outfile_dnS+confnumber+sourcepositiontext_inside+"_s"+spin+"_c"+col);
              //vectorAuxPrint.writeHDF5(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            }
        }



        std::vector<int> mom={0,0,0};
	
	std::string outfilename;

	   
        //N diagram
	if (do_contraction_std==true){
	  std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
	  momList list_mpf1(1,{mpf1,},{0,});

	  //initialize diagram
	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_N";

	
	  //Computing T reductions+recombination
	  { 
	    PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
	    PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
	    //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
	    TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP), "ISOSPIN32");

	    PLEGMA_printf("Nucleon T2 reduction\n");
	    TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP), "ISOSPIN32");
	    PLEGMA_printf("Nucleon T2 reduction ready\n");

	    TIME(corrNP.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN32");
	    PLEGMA_printf("Nucleon diagram ready\n");

// PLEGMA_SCATTERING_SPIN12 
	    //Secondly compute N_0 (neutron)(we need for M diagram (N0p+))
	    TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN), "ISOSPIN12");
	  
	    TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN), "ISOSPIN12");

	    TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN12");

	  }//end of T reduction 
        }//end of if(do_contraction_std)


	asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
        std::string sourcepositiontext= (std::string)"_" + ssource;
        free(ssource);

        //P diagram
        std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
        momList list_mpi2(1,{mpi2,},{0,});
        PLEGMA_ScattCorrelator<float> corrP0UP(sourcePositions[isource], list_mpi2);
        PLEGMA_ScattCorrelator<float> corrP0DN(sourcePositions[isource], list_mpi2);
        PLEGMA_ScattCorrelator<float> corrPPUP(sourcePositions[isource], list_mpi2);
        PLEGMA_ScattCorrelator<float> corrPPDN(sourcePositions[isource], list_mpi2);

        corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
        corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
        corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
        corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");

        PLEGMA_Vector<float> stochastic_oet_prop_u_zero_mom_source_to_sink;
        PLEGMA_Vector<float> stochastic_oet_prop_d_zero_mom_source_to_sink;

        {
        PLEGMA_Vector<double> vectortmp1;
        PLEGMA_Vector<double> vectortmp2;

        // ensuring mu positive
        if(mu<0) {
           mu*=-1.;
           solver.UpdateSolver();
        }

        vectortmp2.copy(vectorSource_oet);
        vectortmp1.absorbTimeslice(vectortmp2, sourcePositions[isource][DIM_T]);


        vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);
        //Multiplying by the appropriate momentum phase


        //Doing the inversion
        TIME(solver.solve(vectortmp2, vectortmp2),"ISOSPIN32");


        //Rotate back immediately to the physical basis
        vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);


        //performing smearing
	    
        TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

        stochastic_oet_prop_u_zero_mom_source_to_sink.copy(vectortmp2);
        // ensuring mu negative
        if(mu>0) {
           mu*=-1.;
           solver.UpdateSolver();
        }

        vectortmp2.copy(vectorSource_oet);
        vectortmp1.absorbTimeslice(vectortmp2, sourcePositions[isource][DIM_T]);

        vectortmp2.rotateToPhysicalBasis(vectortmp1,-1);

        //Doing the inversion
        TIME(solver.solve(vectortmp2, vectortmp2),"ISOSPIN32");

        //Rotate back immediately to the physical basis
        vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);

        //performing smearing
        TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");
        stochastic_oet_prop_d_zero_mom_source_to_sink.copy(vectortmp1);

	}




	{ //W,Z diagrams

        PLEGMA_Vector<float> spropagator_V6;

	PLEGMA_ScattCorrelator<float> reductionsV3_phiui2_U(source, sourcemomentumList.uniq_p(2));
	PLEGMA_ScattCorrelator<float> reductionsV3_phidi2_U(source, sourcemomentumList.uniq_p(2));
        PLEGMA_ScattCorrelator<float> reductionsV3_phiui2_D(source, sourcemomentumList.uniq_p(2));

	
	//for the Z diagram we need V3 reduction for momentum pf2
        PLEGMA_ScattCorrelator<float> reductionsV2_phidi2_UU(source, sourcemomentumList.uniq_p(1));
	PLEGMA_ScattCorrelator<float> reductionsV2_phiui2_DU(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV2_phiui2_DD(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV2_phidi2_UD(source, sourcemomentumList.uniq_p(1));

	//for the Z diagram we need V2 reduction for momentum pf1
	PLEGMA_ScattCorrelator<float> reductionsV4_phiui2_DU(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV4_phidi2_DU(source, sourcemomentumList.uniq_p(1));

	//for the Z diagram we need V4 reduction for momentum pf1
        PLEGMA_ScattCorrelator<float> reductionsV6_W_phiui2_phidf2U(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV6_W_phiuf2_phiui2D(source, sourcemomentumList.uniq_p(1));
	PLEGMA_ScattCorrelator<float> reductionsV6_W_phiui2_phiuf2D(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV6_W_phiui2phidf2_U(source, sourcemomentumList.uniq_p(1));
  	PLEGMA_ScattCorrelator<float> reductionsV6_W_phiuf2phidi2_U(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV6_W_phiui2phiuf2_U(source, sourcemomentumList.uniq_p(1));
	PLEGMA_ScattCorrelator<float> reductionsV6_W_phidf2phiui2_D(source, sourcemomentumList.uniq_p(1));
	PLEGMA_ScattCorrelator<float> reductionsV6_W_phiuf2_phidi2U(source, sourcemomentumList.uniq_p(1));



        PLEGMA_ScattCorrelator<float> reductionsV6_W_phidf2_phiui2D(source, sourcemomentumList.uniq_p(1));
	PLEGMA_ScattCorrelator<float> reductionsV6_W_phidf2_phidi2U(source, sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV6_W_phidi2_phidf2U(source, sourcemomentumList.uniq_p(1));


	PLEGMA_ScattCorrelator<float> reductionsV2_phiui2_UUT(source,  sourcemomentumList.uniq_p(3));
        PLEGMA_ScattCorrelator<float> reductionsV2_phiui2_UDT(source,  sourcemomentumList.uniq_p(3));
        PLEGMA_ScattCorrelator<float> reductionsV2_phiui2_DUT(source, sourcemomentumList.uniq_p(3));
        PLEGMA_ScattCorrelator<float> reductionsV2_phidi2_UUT(source, sourcemomentumList.uniq_p(3));


	//for the T diagram we need V2 reduction for momentum p total
	
	PLEGMA_ScattCorrelator<float> reductionsV4_phiui2_UUT(source,  sourcemomentumList.uniq_p(3));
        PLEGMA_ScattCorrelator<float> reductionsV4_phidi2_UUT(source,  sourcemomentumList.uniq_p(3));
        PLEGMA_ScattCorrelator<float> reductionsV4_phiui2_DUT(source, sourcemomentumList.uniq_p(3));
        PLEGMA_ScattCorrelator<float> reductionsV4_phiui2_UDT(source,  sourcemomentumList.uniq_p(3));

	//for the T diagram we need V4 reduction for momentum p total

	//Factors for the T diagram (zero momentum i2)

        TIME(reductionsV4_phiui2_UUT.V4( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_delta, propUP, propUP, true),"ISOSPIN12");
        TIME(reductionsV4_phiui2_UDT.V4( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_delta, propUP, propDN, true),"ISOSPIN12");
	TIME(reductionsV4_phiui2_DUT.V4( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_delta, propDN, propUP, true),"ISOSPIN12");


	//THREAD(reductionsV4_phiui2_UU.writeHDF5("V4redforTdiagram"));

        TIME(reductionsV2_phiui2_UUT.V2( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_delta, propUP, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_UDT.V2( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_delta, propUP, propDN, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_DUT.V2( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_delta, propDN, propUP, true),"ISOSPIN12");


	//THREAD(reductionsV2_phiui2_UUT.writeHDF5("V2redforTdiagram"));

	//Factors for the Z diagram
	TIME(reductionsV4_phiui2_DU.V4( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_nucleon, propDN, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_DU.V2( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_nucleon, propDN, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_DD.V2( stochastic_oet_prop_u_zero_mom_source_to_sink, glist_sink_nucleon, propDN, propDN, true),"ISOSPIN12");	

 
        TIME(reductionsV2_phidi2_UD.V2( stochastic_oet_prop_d_zero_mom_source_to_sink, glist_sink_nucleon, propUP, propDN, true),"ISOSPIN12");
        TIME(reductionsV2_phidi2_UU.V2( stochastic_oet_prop_d_zero_mom_source_to_sink, glist_sink_nucleon, propUP, propUP, true),"ISOSPIN12");
        TIME(reductionsV4_phidi2_DU.V4( stochastic_oet_prop_d_zero_mom_source_to_sink, glist_sink_nucleon, propDN, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phidi2_UUT.V2( stochastic_oet_prop_d_zero_mom_source_to_sink, glist_sink_delta,propUP, propUP, true),"ISOSPIN12");
	TIME(reductionsV4_phidi2_UUT.V4( stochastic_oet_prop_d_zero_mom_source_to_sink, glist_sink_delta, propUP, propUP, true),"ISOSPIN32");

//	reductionsV4.writeHDF5("V4redforZdiagram");

//      reductionsV2.writeHDF5("V2redforZdiagram");
	//Loop over the source meson momentum
        for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);

          PLEGMA_ScattCorrelator<float> corrZ1(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ2(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ3(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ4(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrZ5(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ6(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ7(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ8(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ9(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ10(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ11(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ12(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ13(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ14(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrZ15(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ16(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ17(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ18(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ19(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ20(sourcePositions[isource], filtered_sourcemomentumList);
/*
          PLEGMA_ScattCorrelator<float> corrZ1_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ2_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ3_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ4_check(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrZ5_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ6_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ7_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ8_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ9_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ10_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ11_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ12_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ13_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ14_check(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrZ15_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ16_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ17_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ18_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ19_check(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ20_check(sourcePositions[isource], filtered_sourcemomentumList);

*/

	  PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList);
	  PLEGMA_ScattCorrelator<float> corrD1if12(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrD1if34(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrD1if56(sourcePositions[isource], filtered_sourcemomentumList);

	  
          

          corrZ1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z1");
          corrZ2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z2");
          corrZ3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z3");
          corrZ4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z4");
          corrZ5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z5");
          corrZ6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z6");
          corrZ7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z7");
          corrZ8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z8");
          corrZ9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z9");
          corrZ10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z10");
          corrZ11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z11");
          corrZ12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z12");
          corrZ13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z13");
          corrZ14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z14");
          corrZ15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z15");
          corrZ16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z16");
          corrZ17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z17");
          corrZ18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z18");
          corrZ19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z19");
          corrZ20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z20");
/*
          corrZ1_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z1");
          corrZ2_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z2");
          corrZ3_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z3");
          corrZ4_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z4");
          corrZ5_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z5");
          corrZ6_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z6");
          corrZ7_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z7");
          corrZ8_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z8");
          corrZ9_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z9");
          corrZ10_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z10");
          corrZ11_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z11");
          corrZ12_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z12");
          corrZ13_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z13");
          corrZ14_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z14");
          corrZ15_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z15");
          corrZ16_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z16");
          corrZ17_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z17");
          corrZ18_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z18");
          corrZ19_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z19");
          corrZ20_check.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z20");
*/
          corrM.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "MNPPP");

	  TIME(corrD1if34.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP01"),"ISOSPIN12");

          TIME(corrD1if12.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP02"),"ISOSPIN12");

          TIME(corrD1if56.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MN0PP"),"ISOSPIN12");


	  if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){

            PLEGMA_Vector<double> vectortmp1;
            PLEGMA_Vector<double> vectortmp2;

            // ensuring mu positive
            if(mu<0) {
              mu*=-1.;
              solver.UpdateSolver();
            }
            double tmp;

            vectortmp2.copy(vectorSource_oet);
            vectortmp1.absorbTimeslice(vectortmp2, sourcePositions[isource][DIM_T]);


            vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);


            //Multiplying by the appropriate momentum phase

            std::vector<int> tmp_4Dmom= momentum_i2 ;
            tmp_4Dmom.push_back(0);
            vectortmp2.mulMomentumPhases(tmp_4Dmom,-1);


            //Doing the inversion
            TIME(solver.solve(vectortmp2, vectortmp2),"ISOSPIN32");


            //Rotate back immediately to the physical basis
            vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);

            //performing smearing
	    
            TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

            stochastic_oet_prop_u_fini_mom_source_to_sink.copy(vectortmp2);

            // ensuring mu negative
            if(mu>0) {
              mu*=-1.;
              solver.UpdateSolver();
            }

	    vectortmp2.copy(vectorSource_oet);
            vectortmp1.absorbTimeslice(vectortmp2, sourcePositions[isource][DIM_T]);

            vectortmp2.rotateToPhysicalBasis(vectortmp1,-1);

            //Multiplying by the appropriate momentum phase

            vectortmp2.mulMomentumPhases(tmp_4Dmom,-1);


            //Doing the inversion
            TIME(solver.solve(vectortmp2, vectortmp2),"ISOSPIN32");

            //Rotate back immediately to the physical basis
            vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);

            //performing smearing
            TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");
            stochastic_oet_prop_d_fini_mom_source_to_sink.copy(vectortmp1);
/*
            {
              PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(stochastic_oet_prop_u_fini_mom_source_to_sink);
              vectorAuxF.unload();
              vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_u_propagator_oet_mpi2_momstoch_mom_pi2x"+std::to_string(momentum_i2[0])+"_pi2y"+std::to_string(momentum_i2[1])+"_pi2z"+std::to_string(momentum_i2[2])+"_"+confnumber);
            }
	    {
              PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(stochastic_oet_prop_d_fini_mom_source_to_sink);
              vectorAuxF.unload();
              vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_d_propagator_oet_mpi2_momstoch_mom_pi2x"+std::to_string(momentum_i2[0])+"_pi2y"+std::to_string(momentum_i2[1])+"_pi2z"+std::to_string(momentum_i2[2])+"_"+confnumber);
            }
 
*/
          }


	  std::shared_ptr<float> Phi0;



	  //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.


	  if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
	    
	    TIME(corrPPUP.P_diagrams( stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_u_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

            TIME(corrP0DN.P_diagrams( stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_d_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

	 
	    TIME(corrM.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_u_fini_mom_source_to_sink ),"ISOSPIN32");
            TIME(corrD1if12.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_d_fini_mom_source_to_sink),"ISOSPIN12");

	    TIME(corrD1if56.M_diagrams( corrN0, stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_u_fini_mom_source_to_sink),"ISOSPIN12");
	  
	  }
	  else{

	    TIME(corrPPUP.P_diagrams( stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_u_zero_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");
            TIME(corrP0DN.P_diagrams( stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_d_zero_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");
	    TIME(corrM.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom_source_to_sink,stochastic_oet_prop_u_zero_mom_source_to_sink ),"ISOSPIN32");
            TIME(corrD1if12.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom_source_to_sink,stochastic_oet_prop_d_zero_mom_source_to_sink),"ISOSPIN12");

            TIME(corrD1if56.M_diagrams( corrN0, stochastic_oet_prop_u_zero_mom_source_to_sink, stochastic_oet_prop_u_zero_mom_source_to_sink),"ISOSPIN12");
	  
	  }


          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){

            TIME(corrPPDN.P_diagrams( stochastic_oet_prop_d_zero_mom_source_to_sink, stochastic_oet_prop_d_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

            TIME(corrP0UP.P_diagrams( stochastic_oet_prop_d_zero_mom_source_to_sink, stochastic_oet_prop_u_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

	    TIME(corrD1if34.M_diagrams( corrNP, stochastic_oet_prop_d_zero_mom_source_to_sink, stochastic_oet_prop_u_fini_mom_source_to_sink ),"ISOSPIN12");

          }
          else{

            TIME(corrPPDN.P_diagrams( stochastic_oet_prop_d_zero_mom_source_to_sink, stochastic_oet_prop_d_zero_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");
            TIME(corrP0UP.P_diagrams( stochastic_oet_prop_d_zero_mom_source_to_sink, stochastic_oet_prop_u_zero_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");
            TIME(corrD1if34.M_diagrams( corrNP, stochastic_oet_prop_d_zero_mom_source_to_sink, stochastic_oet_prop_u_zero_mom_source_to_sink ),"ISOSPIN12");


          }



          for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2) {


            GAMMAS_SCATT gamma_i2 = glist_sink_meson[i_gamma_i2];
            GAMMAS_SCATT gamma_i2_t_gamma5= apply_g5( gamma_i2, RIGHT );

            std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
	    

	    if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){

              spropagator_V6.copy(stochastic_oet_prop_u_fini_mom_source_to_sink);
              spropagator_V6.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);


              TIME(reductionsV3_phiui2_U.V3( spropagator_V6, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");
              TIME(reductionsV3_phiui2_D.V3( spropagator_V6, gamma_5_t_sinkmeson, propDN, true),"ISOSPIN32");

              spropagator_V6.copy(stochastic_oet_prop_d_fini_mom_source_to_sink);
              spropagator_V6.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);
              TIME(reductionsV3_phidi2_U.V3( spropagator_V6, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");


//              THREAD(reductionsV3_phidi2_U.writeHDF5("V3redforZdiagramU"));


            }
            else{

              spropagator_V6.copy(stochastic_oet_prop_u_zero_mom_source_to_sink);;
              spropagator_V6.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);

              TIME(reductionsV3_phiui2_U.V3( spropagator_V6, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");
              TIME(reductionsV3_phiui2_D.V3( spropagator_V6, gamma_5_t_sinkmeson, propDN, true),"ISOSPIN32");

	      spropagator_V6.copy(stochastic_oet_prop_d_zero_mom_source_to_sink);;
              spropagator_V6.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);

              TIME(reductionsV3_phidi2_U.V3( spropagator_V6, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");

              //THREAD(reductionsV3.writeHDF5("V3redforZdiagram"));

            }

/*

	        void Z_diagrams_without_dilution_check(PLEGMA_ScattCorrelator<Float> &srcV3,
                                                PLEGMA_ScattCorrelator<Float> &srcV2, int i_g_i2,
                                                std::string proporder, int diagram_number, bool transp_i1, bool transp_f1, bool accum );*/

            TIME(corrZ1.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phiui2_DU, i_gamma_i2, 1 ),"ISOSPIN32");
//            TIME(corrZ1_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV4_phiui2_DU, i_gamma_i2, "PSS",1,true, false,false),"ISOSPIN32");

            TIME(corrZ2.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phiui2_DU, i_gamma_i2, 2 ),"ISOSPIN32");
//            TIME(corrZ2_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV4_phiui2_DU, i_gamma_i2, "PSS",2,true, false,false),"ISOSPIN32");


            TIME(corrZ3.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DU, i_gamma_i2, 3 ),"ISOSPIN32");
//            TIME(corrZ3_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV2_phiui2_DU, i_gamma_i2, "SSP",1,true, false,false),"ISOSPIN32");

            TIME(corrZ4.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DU, i_gamma_i2, 4 ),"ISOSPIN32");
//            TIME(corrZ4_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV2_phiui2_DU, i_gamma_i2, "SSP",2,true, false,false),"ISOSPIN32");

            TIME(corrZ5.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV2_phiui2_DU, i_gamma_i2, 5 ),"ISOSPIN12");
//            TIME(corrZ5_check.Z_diagrams_without_dilution_check( reductionsV3_phidi2_U, reductionsV2_phiui2_DU, i_gamma_i2, "SSP",2,true, false,false),"ISOSPIN32");

            TIME(corrZ6.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV4_phiui2_DU, i_gamma_i2, 6 ),"ISOSPIN12");
//            TIME(corrZ6_check.Z_diagrams_without_dilution_check( reductionsV3_phidi2_U, reductionsV4_phiui2_DU, i_gamma_i2, "PSS",1,true, false,false),"ISOSPIN32");

            TIME(corrZ7.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV2_phiui2_DU, i_gamma_i2, 7 ),"ISOSPIN12");
//            TIME(corrZ7_check.Z_diagrams_without_dilution_check( reductionsV3_phidi2_U, reductionsV2_phiui2_DU, i_gamma_i2, "SSP",1,true, false,false),"ISOSPIN32");

            TIME(corrZ8.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV4_phiui2_DU, i_gamma_i2, 8 ),"ISOSPIN12");
//            TIME(corrZ8_check.Z_diagrams_without_dilution_check( reductionsV3_phidi2_U, reductionsV4_phiui2_DU, i_gamma_i2, "PSS",2,true, false,false),"ISOSPIN32");

            TIME(corrZ9.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phidi2_UU, i_gamma_i2, 9 ),"ISOSPIN12");
//            TIME(corrZ9_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_D, reductionsV2_phidi2_UU, i_gamma_i2, "SPS",1,false,true,false),"ISOSPIN32");

            TIME(corrZ10.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phidi2_UU, i_gamma_i2, 10 ),"ISOSPIN12");
//            TIME(corrZ10_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_D, reductionsV2_phidi2_UU, i_gamma_i2, "SPS",2,false,true,false),"ISOSPIN32");

            TIME(corrZ11.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phiui2_DU, i_gamma_i2, 11 ),"ISOSPIN12");
//            TIME(corrZ11_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_D, reductionsV2_phiui2_DU, i_gamma_i2, "SPS",1,true, false,false) ,"ISOSPIN12");

            TIME(corrZ12.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV4_phiui2_DU, i_gamma_i2, 12 ),"ISOSPIN12");
//            TIME(corrZ12_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_D, reductionsV4_phiui2_DU, i_gamma_i2, "PSS",1,true, false,false) ,"ISOSPIN12");

            TIME(corrZ13.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phiui2_DU, i_gamma_i2, 13 ),"ISOSPIN12");
//            TIME(corrZ13_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_D, reductionsV2_phiui2_DU, i_gamma_i2, "SSP",1,false, false,false) ,"ISOSPIN12");

            TIME(corrZ14.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV4_phiui2_DU, i_gamma_i2, 14 ),"ISOSPIN12");
//            TIME(corrZ14_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_D, reductionsV4_phiui2_DU, i_gamma_i2, "PSS",2,false, false,false) ,"ISOSPIN12");

	    TIME(corrZ15.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DD, i_gamma_i2, 15 ),"ISOSPIN12");
//            TIME(corrZ15_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV2_phiui2_DD, i_gamma_i2, "SPS",2,false, true,false) ,"ISOSPIN12");

            TIME(corrZ16.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DD, i_gamma_i2, 16 ),"ISOSPIN12");
//            TIME(corrZ16_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV2_phiui2_DD, i_gamma_i2, "SPS",1,false, true,false) ,"ISOSPIN12");

            TIME(corrZ17.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phidi2_UD, i_gamma_i2, 17 ),"ISOSPIN12");
//            TIME(corrZ17_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV2_phidi2_UD, i_gamma_i2, "SPS", 1, true, false,false ),"ISOSPIN12");

            TIME(corrZ18.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phidi2_UD, i_gamma_i2, 18 ),"ISOSPIN12");
//            TIME(corrZ18_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV2_phidi2_UD, i_gamma_i2, "SSP",1,false,false ,false),"ISOSPIN12");

            TIME(corrZ19.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phidi2_DU, i_gamma_i2, 19 ),"ISOSPIN12");
//            TIME(corrZ19_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV4_phidi2_DU, i_gamma_i2, "PSS",1, true, false,false ),"ISOSPIN12");

            TIME(corrZ20.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phidi2_DU, i_gamma_i2, 20 ),"ISOSPIN12");
//            TIME(corrZ20_check.Z_diagrams_without_dilution_check( reductionsV3_phiui2_U, reductionsV4_phidi2_DU, i_gamma_i2, "PSS",2,false,false,false ),"ISOSPIN12");




	  }

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

          TIME(produceOutput(corrZ1, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ2, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ3, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ4, outfilename, "4pt",1, NULL),"ISOSPIN32");
	  TIME(produceOutput(corrZ5, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ6, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ7, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ8, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ9, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ10, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ11, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ12, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ13, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ14, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ15, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ16, outfilename, "4pt",1,NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ17, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ18, outfilename, "4pt",1,NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ19, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ20, outfilename, "4pt",1, NULL),"ISOSPIN12");
/*
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Zcheck";

          TIME(produceOutput(corrZ1_check, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ2_check, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ3_check, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ4_check, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrZ5_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ6_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ7_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ8_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ9_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ10_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ11_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ12_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ13_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ14_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ15_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ16_check, outfilename, "4pt",1,NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ17_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ18_check, outfilename, "4pt",1,NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ19_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrZ20_check, outfilename, "4pt",1, NULL),"ISOSPIN12");
*/
  
	  //## M
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";
          TIME(produceOutput(corrM, outfilename,"4pt",1,NULL),"ISOSPIN32");
          TIME(produceOutput(corrD1if12, outfilename,"4pt",1,NULL),"ISOSPIN32");
          TIME(produceOutput(corrD1if34, outfilename,"4pt",1,NULL),"ISOSPIN32");
          TIME(produceOutput(corrD1if56, outfilename,"4pt",1,NULL),"ISOSPIN32");



      }//loop over unique set of momenta for p_i2

      //write P
      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrPPUP.apply_sign("P"),"ISOSPIN32");
      TIME_INSIDE(corrPPUP.writeHDF5( outfilename ));
      TIME(corrPPDN.apply_sign("P"),"ISOSPIN32");
      TIME_INSIDE(corrPPDN.writeHDF5( outfilename ));
      TIME(corrP0UP.apply_sign("P"),"ISOSPIN32");
      TIME_INSIDE(corrP0UP.writeHDF5( outfilename ));
      TIME(corrP0DN.apply_sign("P"),"ISOSPIN32");
      TIME_INSIDE(corrP0DN.writeHDF5( outfilename ));      

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_N";
      TIME( corrN0.apply_phase(),"ISOSPIN12" );
      TIME( corrN0.apply_sign("N"),"ISOSPIN12" );
      TIME( corrN0.applyBoundaryConditions( true ),"ISOSPIN12" );
      TIME( corrN0.writeHDF5(outfilename) ,"ISOSPIN12");

      TIME( corrNP.apply_phase(),"ISOSPIN32" );
      TIME( corrNP.apply_sign("N"),"ISOSPIN32" );
      TIME( corrNP.applyBoundaryConditions( true ),"ISOSPIN32" );
      TIME( corrNP.writeHDF5(outfilename),"ISOSPIN32" );




      }//Z,W,P,M diagrams


    }



//     while(not threads.empty()) {threads.back().join(); threads.pop_back();}


  } 

  finalize();
  
  return 0;
}
