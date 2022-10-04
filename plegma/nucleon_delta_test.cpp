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
      std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5,CG_5_G_4};
      std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5,CG_5_G_4};
      std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
      std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};

      std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
      std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

      std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
      std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

      
      std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
      std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);
      
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

	   

	//D diagram
	if (do_contraction_std == true)
	{
	  std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
	  momList list_mtot(1,{mtot,},{0,});
	  PLEGMA_ScattCorrelator<float> corrD(src,list_mtot);

	  //initialize diagram
	  corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,"D");
	
	  PLEGMA_ScattCorrelator<float> reductionsT1(source, mtot);
	  PLEGMA_ScattCorrelator<float> reductionsT2(source, mtot);

	  TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP),"ISOSPIN32");

	  TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP),"ISOSPIN32");

	  //write D
	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_D";
	  
	  TIME( corrD.D_diagrams( reductionsT1, reductionsT2 ),"ISOSPIN32");
	  TIME( corrD.apply_phase(),"ISOSPIN32" );
	  TIME( corrD.apply_sign("D") ,"ISOSPIN32");
	  TIME( corrD.applyBoundaryConditions( true ),"ISOSPIN32" );
	  TIME( corrD.writeHDF5(outfilename),"ISOSPIN32" );


	  //D I=1/2 I_3=+1/2
          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_DNUPUP_T2";
          TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propUP), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT2, -1 ), "ISOSPIN12"); //The argument -1 indicates the corrD does not contain any mesonic
          //gamma structure
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_DNUPUP_T1";
          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propUP, propUP), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT1, -1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPUPDN_T1";
          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propDN), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPDNUP_T1";
          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propDN, propUP), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPUPDN_T2";
          TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propDN), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT2,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

	  //N to D I=1/2 I_3=+1/2
          PLEGMA_ScattCorrelator<float> corrDN(src,list_mtot);

          //initialize diagram
          corrDN.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_delta_unpaired,glist_source_nucleon, glist_sink_delta,"D");

	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_NUCLEON_DELTA_UPUPDN_T2";
          TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_delta, propUP, propUP, propDN), "ISOSPIN12");
          TIME( corrDN.convertTreductiontoDiagram( reductionsT2, -1 ), "ISOSPIN12"); //The argument -1 indicates the corrD does not contain any mesonic
          //gamma structure
          TIME( produceOutput(corrDN, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_NUCLEON_DELTA_DNUPUP_T1";
          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propDN, propUP, propUP), "ISOSPIN12");
          TIME( corrDN.convertTreductiontoDiagram( reductionsT1, -1 ), "ISOSPIN12");
          TIME( produceOutput(corrDN, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_NUCLEON_DELTA_UPDNUP_T1";
          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propUP, propDN, propUP), "ISOSPIN12");
          TIME( corrDN.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrDN, outfilename, "D" ) ,"ISOSPIN12" );


	  //N diagram
	  std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
	  momList list_mpf1(1,{mpf1,},{0,});

	  //initialize diagram
	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_N";

	
	  PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
	  PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
	  //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
	  TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP), "ISOSPIN32");

	  //PLEGMA_printf("Nucleon T2 reduction\n");
	  TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP), "ISOSPIN32");
	  //PLEGMA_printf("Nucleon T2 reduction ready\n");

	  TIME(corrNP.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN32");
	  //PLEGMA_printf("Nucleon diagram ready\n");

// PLEGMA_SCATTERING_SPIN12 
	  //Secondly compute N_0 (neutron)(we need for M diagram (N0p+))
	  TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN), "ISOSPIN12");
	  
	  TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN), "ISOSPIN12");

	  TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN12");

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext_inside+"_N";
          TIME( corrN0.apply_phase(),"ISOSPIN12" );
          TIME( corrN0.apply_sign("N"),"ISOSPIN12" );
          TIME( corrN0.applyBoundaryConditions( true ),"ISOSPIN12" );
          TIME( corrN0.writeHDF5(outfilename) ,"ISOSPIN12");

          TIME( corrNP.apply_phase(),"ISOSPIN32" );
          TIME( corrNP.apply_sign("N"),"ISOSPIN32" );
          TIME( corrNP.applyBoundaryConditions( true ),"ISOSPIN32" );
          TIME( corrNP.writeHDF5(outfilename),"ISOSPIN32" );
	}//end of if(do_contraction_std)

      }//loop over source positions

    } //initialize PLEGMA

  finalize();
  
  return 0;
}
