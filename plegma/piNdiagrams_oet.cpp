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
      std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_zero_mom;
      std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_fini_mom;

      std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_zero_mom;
      std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_fini_mom;

      PLEGMA_Vector<float> stochastic_oet_prop_d_fini_mom_source_to_sink;
      PLEGMA_Vector<float> stochastic_oet_prop_u_fini_mom_source_to_sink;



      for(int i=0; i< HGC_totalL[DIM_T]; ++i) {
	stochastic_oet_prop_d_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
        stochastic_oet_prop_u_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));

      }
      //length of pf2
      int length_fini_mom=(sourcemomentumList.uniq_p(2)).size();
      for(int i=0; i< length_fini_mom; ++i) {
	stochastic_oet_prop_d_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));
        stochastic_oet_prop_u_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));

      }


      PLEGMA_printf("Start producing stochastic oet propagators\n");
      //Note that we replace the f1<-f2 DN propagator with a stochastic one
      //phi^f2(xf1)
      //In the same time we replace 
      //U(xf2,xi2) with gamma5 D(xi2,xf2)^DAGGER gamma5 = gamma5 phi*^(f2)(xi2) gamma5


      std::vector<std::vector<int>> mpf2 = sourcemomentumList.uniq_p(2);

     
      if (read_stochastic_oet==false){
	
	PLEGMA_Vector<double> vectorAuxD1(BOTH);//For storing the source (rotated and smeared)
	PLEGMA_Vector<double> vectorAuxD2(BOTH);//For storing the propagotor for the time-slices
	PLEGMA_Vector<double> vectorAuxD3(BOTH);
	PLEGMA_Vector<double> vectorAuxD4(BOTH);
	PLEGMA_Vector<double> vectorInOut; //temporary vector using in solve

	
	// ensuring mu negative
        if(mu>0) {
          mu*=-1.;
          solver.UpdateSolver();
        }

	PLEGMA_Vector<double> vectorSource_oet;
	vectorSource_oet.randInit(rand_seed1);
	vectorSource_oet.stochastic_Z(nroots);

        TIME(vectorInOut.gaussianSmearing(vectorSource_oet, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

	for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){

	  auto &momentum_f2 = mpf2[i_mpf2];

	  std::vector<int> tmp_4Dmom= momentum_f2 ;
	  tmp_4Dmom.push_back(0);
	  
	  //Step(1) multiply with the momentum phase
	  vectorAuxD1.copy(vectorInOut);
	  vectorAuxD1.mulMomentumPhases(tmp_4Dmom,-1);

	  //In vectorAuxD2 we store the results for the inversion for sink to sink
	  vectorAuxD2.scale(0.0);
 
	  //Step(2) We rotate the source to the physical basis
	  TIME(vectorAuxD3.rotateToPhysicalBasis(vectorAuxD1,-1),"ISOSPIN32");
   
	  for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

	    //Step(3) pick out a particular timeslice from the source
	    vectorAuxD1.absorbTimeslice(vectorAuxD3, timeidx);

	    //Step(4) Solve
	    TIME(solver.solve(vectorAuxD1, vectorAuxD1),"ISOSPIN32");

	    //Step(5) absorbing the particular timeslice to a 4d vector
	    vectorAuxD2.absorbTimeslice(vectorAuxD1, timeidx, false);

	    if ((momentum_f2[0] == 0) && (momentum_f2[1] == 0) && (momentum_f2[2] == 0)){
	

	      //Step(6) We rotate back the propagator to the physical basis
	      TIME(vectorAuxD4.rotateToPhysicalBasis(vectorAuxD1,-1),"ISOSPIN32");

	      //Step(7) Smearing all the time slice in the propagator
	      TIME(vectorAuxD1.gaussianSmearing(vectorAuxD4, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

	      //Step(8) Save the propagator to the disk
	      {
	       PLEGMA_Vector<float> vectorAuxF;
	       vectorAuxF.copy(vectorAuxD1);
//	       vectorAuxF.unload();
	       vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_d_propagator_oet_stoch_time_"+std::to_string(timeidx)+"_"+confnumber);
	      }

	      //Step(9) Save the propagator to the host memory
	      vectorAuxD1.unload();
	      stochastic_oet_prop_d_zero_mom[timeidx]->copy(vectorAuxD1,HOST);
	      vectorAuxD1.load();
	    }
	  
	  } //timeidx

	  //Step(10) We rotate back the propagator to the physical basis
	  TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,-1),"ISOSPIN32");

	  //Step(11) Smearing all the time slice in the propagator
	  TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

	  //Step(12) Save the propagator to the disk
	  {
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxF.copy(vectorAuxD2);
	    vectorAuxF.unload();
	    vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_d_propagator_oet_stoch_mom_pf2x_"+std::to_string(momentum_f2[0])+"_pf2y"+std::to_string(momentum_f2[1])+"_pf2z"+std::to_string(momentum_f2[2])+"_"+confnumber);
	  }

	  //Step(13) Save the propagator to the host memory
	  vectorAuxD2.unload();
	  stochastic_oet_prop_d_fini_mom[i_mpf2]->copy(vectorAuxD2,HOST);
	  vectorAuxD2.load();
	}


	if(mu<0) {
          mu*=-1.;
          solver.UpdateSolver();
        }

        for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){

          auto &momentum_f2 = mpf2[i_mpf2];

          vectorAuxD1.copy(vectorInOut);

          std::vector<int> tmp_4Dmom= momentum_f2 ;
          tmp_4Dmom.push_back(0);

          //Step(1) multiply with the momentum phase
          vectorAuxD1.mulMomentumPhases(tmp_4Dmom,-1);

          //In vectorAuxD2 we store the results for the inversion for sink to sink
          vectorAuxD2.scale(0.0);

          //Step(2) We rotate the source to the physical basis
          TIME(vectorAuxD3.rotateToPhysicalBasis(vectorAuxD1,+1),"ISOSPIN12");

          for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

            //Step(4) pick out a particular timeslice from the source
            vectorAuxD1.absorbTimeslice(vectorAuxD3, timeidx);

            //Step(5) Solve
            TIME(solver.solve(vectorAuxD1, vectorAuxD1),"ISOSPIN12");

            //Step(6) absorbing the particular timeslice to a 4d vector
            vectorAuxD2.absorbTimeslice(vectorAuxD1, timeidx, false);

            if ((momentum_f2[0] == 0) && (momentum_f2[1] == 0) && (momentum_f2[2] == 0)){

              //Step(7) We rotate back the propagator to the physical basis
              TIME(vectorAuxD4.rotateToPhysicalBasis(vectorAuxD1,+1),"ISOSPIN32");

              //Step(8) Smearing all the time slice in the propagator
              TIME(vectorAuxD1.gaussianSmearing(vectorAuxD4, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

              //Step(9) Save the propagator to the disk
              {
               PLEGMA_Vector<float> vectorAuxF;
               vectorAuxF.copy(vectorAuxD1);
//             vectorAuxF.unload();
               vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_u_propagator_oet_stoch_time_"+std::to_string(timeidx)+"_"+confnumber);
              }

              //Step(9) Save the propagator to the host memory
              vectorAuxD1.unload();
              stochastic_oet_prop_u_zero_mom[timeidx]->copy(vectorAuxD1,HOST);
              vectorAuxD1.load();
            }

          } //timeidx

          //Step(10) We rotate back the propagator to the physical basis
          TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1),"ISOSPIN32");

          //Step(11) Smearing all the time slice in the propagator
          TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

          //Step(12) Save the propagator to the disk
          {
            PLEGMA_Vector<float> vectorAuxF;
            vectorAuxF.copy(vectorAuxD2);
            vectorAuxF.unload();
            vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_u_propagator_oet_stoch_mom_pf2x_"+std::to_string(momentum_f2[0])+"_pf2y"+std::to_string(momentum_f2[1])+"_pf2z"+std::to_string(momentum_f2[2])+"_"+confnumber);
          }

          //Step(13) Save the propagator to the host memory
          vectorAuxD2.unload();
          stochastic_oet_prop_u_fini_mom[i_mpf2]->copy(vectorAuxD2,HOST);
          vectorAuxD2.load();
        }
	
      }
      else{
	PLEGMA_Vector<float> vectorRead(BOTH);
	for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){
	  auto &momentum_f2 = mpf2[i_mpf2];
	  if ((momentum_f2[0] == 0) || (momentum_f2[1] == 0) || (momentum_f2[2] == 0)){

	    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

	      std::string inputfilename=outfile_V+"globalTfulltimedilution_d_propagator_oet_stoch_time_"+std::to_string(timeidx)+"_"+confnumber;
	      PLEGMA_printf("Read stochastic oet source (zero momentum source to sink) from: %s\n",inputfilename.c_str());
	  
	      vectorRead.readFile(inputfilename,LIME_FORMAT);
	  
	      stochastic_oet_prop_d_zero_mom[timeidx]->copy(vectorRead,HOST);
	
	    }
	    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

              std::string inputfilename=outfile_V+"globalTfulltimedilution_u_propagator_oet_stoch_time_"+std::to_string(timeidx)+"_"+confnumber;
              PLEGMA_printf("Read stochastic oet source (zero momentum source to sink) from: %s\n",inputfilename.c_str());

              vectorRead.readFile(inputfilename,LIME_FORMAT);

              stochastic_oet_prop_u_zero_mom[timeidx]->copy(vectorRead,HOST);

            }
	  }
	  
	  std::string inputfilename=outfile_V+"globalTfulltimedilution_d_propagator_oet_stoch_mom_pf2x_"+std::to_string(momentum_f2[0])+"_pf2y"+std::to_string(momentum_f2[1])+"_pf2z"+std::to_string(momentum_f2[2])+"_"+confnumber;

	  PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
	    
	  vectorRead.readFile(inputfilename,LIME_FORMAT);
	    
	  stochastic_oet_prop_d_fini_mom[i_mpf2]->copy(vectorRead,HOST);

	  inputfilename=outfile_V+"globalTfulltimedilution_u_propagator_oet_stoch_mom_pf2x_"+std::to_string(momentum_f2[0])+"_pf2y"+std::to_string(momentum_f2[1])+"_pf2z"+std::to_string(momentum_f2[2])+"_"+confnumber;

          PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());

          vectorRead.readFile(inputfilename,LIME_FORMAT);

          stochastic_oet_prop_u_fini_mom[i_mpf2]->copy(vectorRead,HOST);

	}
      } 
      
      
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

// PLEGMA_SCATTERING_SPIN12
	  //For I=1/2 I_3=+1/2
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

	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPDNUP_T2";
	  TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propDN), "ISOSPIN12");
	  TIME( corrD.convertTreductiontoDiagram( reductionsT2,-1 ), "ISOSPIN12");
	  TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

	}
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



        std::vector<GAMMAS_SCATT> gamma_5_t_sourcemeson=apply_gamma5_scatt_gamma(glist_source_meson,LEFT);
	/********************************************************/
	/*
	 *
	 *
	 *
	 * Computing B diagrams for a fixed sink pion momentum
	 *
	 *
	 *
	 * ******************************************************/

	for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){

	  auto &momentum_f2 =  mpf2[i_mpf2];
	  //List of momenta corresponding to a fix value of p_i2
	  momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_f2, 2);

	  PLEGMA_Vector<float> spropagator_V3_u;
          PLEGMA_Vector<float> spropagator_V3_d;

	  PLEGMA_Vector<float> spropagator_V2;

	  std::string pf2x=std::to_string(momentum_f2[0]);
	  std::string pf2y=std::to_string(momentum_f2[1]);
	  std::string pf2z=std::to_string(momentum_f2[2]);
	    
	  PLEGMA_ScattCorrelator<float> corrB1(sourcePositions[isource], filtered_sourcemomentumList);
	  PLEGMA_ScattCorrelator<float> corrB2(sourcePositions[isource], filtered_sourcemomentumList);


          PLEGMA_ScattCorrelator<float> corrB3(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB4(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB5(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB6(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrB7(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB8(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB9(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB10(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrB11(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB12(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrB13(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB14(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB15(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB16(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrB17(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB18(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB19(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB20(sourcePositions[isource], filtered_sourcemomentumList);
	  
	  //initialize diagrams
	  corrB1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B1", true);
	  corrB2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B2", true);


          corrB3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B3", true);
          corrB4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B4", true);
          corrB5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B5", true);
          corrB6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B6", true);
          corrB7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B7", true);
          corrB8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B8", true);
	  corrB9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B9", true);
          corrB10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B10", true);
          corrB11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B11", true);
          corrB12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B12", true);
	  corrB13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B13", true);
          corrB14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B14", true);
          corrB15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B15", true);
          corrB16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B16", true);
          corrB17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B17", true);
          corrB18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B18", true);
          corrB19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B19", true);
          corrB20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B20", true);


	  site source=site({0,0,0,sourcePositions[isource][DIM_T]});	  

	  PLEGMA_ScattCorrelator<float> reductionsV2_phid_UU(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
	  PLEGMA_ScattCorrelator<float> reductionsV3_phid_D(source, filtered_sourcemomentumList.uniq_p(0));//V3 reduction for momentum pi2


          PLEGMA_ScattCorrelator<float> reductionsV2_phiu_DU(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
          PLEGMA_ScattCorrelator<float> reductionsV4_phiu_UD(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
          PLEGMA_ScattCorrelator<float> reductionsV4_phid_UU(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
          PLEGMA_ScattCorrelator<float> reductionsV2_phid_UD(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
	  PLEGMA_ScattCorrelator<float> reductionsV4_phid_UD(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
          PLEGMA_ScattCorrelator<float> reductionsV3_phiu_D(source, filtered_sourcemomentumList.uniq_p(0));//V3 reduction for momentum pi2
          PLEGMA_ScattCorrelator<float> reductionsV3_phid_U(source, filtered_sourcemomentumList.uniq_p(0));//V3 reduction for momentum pi2




	  spropagator_V2.copy(*stochastic_oet_prop_d_fini_mom[i_mpf2], HOST);
	  spropagator_V2.load();

	  reductionsV2_phid_UU.V2(spropagator_V2, glist_sink_nucleon, propUP, propUP, true);

	  reductionsV4_phid_UU.V4(spropagator_V2, glist_sink_nucleon, propUP, propUP, true);

          reductionsV4_phid_UD.V4(spropagator_V2, glist_sink_nucleon, propUP, propDN, true);

          reductionsV2_phid_UD.V2(spropagator_V2, glist_sink_nucleon, propUP, propDN, true);

          spropagator_V2.copy(*stochastic_oet_prop_u_fini_mom[i_mpf2], HOST);
          spropagator_V2.load();

          reductionsV2_phiu_DU.V2(spropagator_V2, glist_sink_nucleon, propDN, propUP, true);

          reductionsV4_phiu_UD.V4(spropagator_V2, glist_sink_nucleon, propUP, propDN, true);

//          THREAD(reductionsV2_phiu_DU.writeHDF5("V2redforBdiagram"+std::to_string(i_mpf2)));

          //Loop over the different gamma structure for the source meson
          for (int i_gamma_f2=0; i_gamma_f2<glist_sink_meson.size(); ++i_gamma_f2) {


	    GAMMAS_SCATT gamma_f2 = glist_sink_meson[i_gamma_f2];
            GAMMAS_SCATT gamma_f2_t_gamma5= apply_g5( gamma_f2, RIGHT );

	    spropagator_V3_d.zero_where(DEVICE);
	    spropagator_V3_d.zero_where(HOST);

	    spropagator_V3_u.zero_where(DEVICE);
	    spropagator_V3_u.zero_where(HOST);
	  
	    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
  
              PLEGMA_Vector<float> spropagator;
              PLEGMA_Vector3D<float> vector1;
 	      spropagator.copy(*stochastic_oet_prop_d_zero_mom[timeidx],HOST);	
	      spropagator.load();
          
	      vector1.absorb(spropagator,  sourcePositions[isource][3], true);
              spropagator_V3_d.absorb(vector1,  timeidx, false);

              spropagator.copy(*stochastic_oet_prop_u_zero_mom[timeidx],HOST);
              spropagator.load();

              vector1.absorb(spropagator,  sourcePositions[isource][3], true);
              spropagator_V3_u.absorb(vector1,  timeidx, false);

	    }

            spropagator_V3_d.apply_gamma_scatt(gamma_f2_t_gamma5,RIGHT);
            spropagator_V3_u.apply_gamma_scatt(gamma_f2_t_gamma5,RIGHT);

	    {

            PLEGMA_Propagator<float> propDN_source_to_source;

	    for (int isc=0; isc<12; ++isc){

              PLEGMA_Vector<float> stmp;
              PLEGMA_Vector3D<float> vector1;

              stmp.zero_where(DEVICE);
              stmp.zero_where(HOST);

	      vector1.absorb(propDN, sourcePositions[isource][3],isc/3, isc%3,true);

	      for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
                stmp.absorb(vector1, timeidx, false); 
	      }

	        
	      propDN_source_to_source.absorb(stmp, isc/3, isc%3);

	    }


	    reductionsV3_phid_D.V3(spropagator_V3_d, gamma_5_t_sourcemeson, propDN_source_to_source, false);
            reductionsV3_phiu_D.V3(spropagator_V3_u, gamma_5_t_sourcemeson, propDN_source_to_source, false);


	    }
            {

            PLEGMA_Propagator<float> propUP_source_to_source;

            for (int isc=0; isc<12; ++isc){

              PLEGMA_Vector<float> stmp;
              PLEGMA_Vector3D<float> vector1;

	      stmp.zero_where(DEVICE);
	      stmp.zero_where(HOST);

              vector1.absorb(propUP, sourcePositions[isource][3],isc/3, isc%3, true);
              for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
                stmp.absorb(vector1, timeidx, false);

              }
           
              propUP_source_to_source.absorb(stmp, isc/3, isc%3);

            }

            reductionsV3_phid_U.V3(spropagator_V3_d, gamma_5_t_sourcemeson, propUP_source_to_source, false);

            }

	    TIME(corrB1.B_diagrams(reductionsV3_phid_D, reductionsV2_phid_UU, i_gamma_f2, 1, true, true),"ISOSPIN32");
	    
	    TIME(corrB2.B_diagrams(reductionsV3_phid_D, reductionsV2_phid_UU, i_gamma_f2, 2, true, true),"ISOSPIN32");

            TIME(corrB3.B_diagrams(reductionsV3_phid_U, reductionsV2_phiu_DU, i_gamma_f2, 3, true, true),"ISOSPIN32");

            TIME(corrB4.B_diagrams(reductionsV3_phid_U, reductionsV4_phiu_UD, i_gamma_f2, 4, true, true),"ISOSPIN32");

            TIME(corrB5.B_diagrams(reductionsV3_phid_U, reductionsV2_phiu_DU, i_gamma_f2, 5, true, true),"ISOSPIN32");

            TIME(corrB6.B_diagrams(reductionsV3_phid_U, reductionsV4_phiu_UD, i_gamma_f2, 6, true, true),"ISOSPIN32");

            TIME(corrB7.B_diagrams(reductionsV3_phiu_D, reductionsV4_phid_UU, i_gamma_f2, 7, true, true),"ISOSPIN32");

            TIME(corrB8.B_diagrams(reductionsV3_phiu_D, reductionsV2_phid_UU, i_gamma_f2, 8, true, true),"ISOSPIN32");

            TIME(corrB9.B_diagrams(reductionsV3_phid_D, reductionsV2_phiu_DU, i_gamma_f2, 9, true, true),"ISOSPIN32");

	    TIME(corrB10.B_diagrams(reductionsV3_phid_D, reductionsV4_phiu_UD, i_gamma_f2, 10, true, true),"ISOSPIN32");

            TIME(corrB11.B_diagrams(reductionsV3_phid_D, reductionsV2_phiu_DU, i_gamma_f2, 11, true, true),"ISOSPIN32");

            TIME(corrB12.B_diagrams(reductionsV3_phid_D, reductionsV4_phiu_UD, i_gamma_f2, 12, true, true),"ISOSPIN32");

            TIME(corrB13.B_diagrams(reductionsV3_phid_D, reductionsV4_phid_UD, i_gamma_f2, 13, true, true),"ISOSPIN32");

            TIME(corrB14.B_diagrams(reductionsV3_phid_D, reductionsV4_phid_UD, i_gamma_f2, 14, true, true),"ISOSPIN32");

            TIME(corrB15.B_diagrams(reductionsV3_phid_D, reductionsV2_phid_UD, i_gamma_f2, 15, true, true),"ISOSPIN32");

            TIME(corrB16.B_diagrams(reductionsV3_phid_D, reductionsV2_phid_UD, i_gamma_f2, 16, true, true),"ISOSPIN32");

	    TIME(corrB17.B_diagrams(reductionsV3_phid_U, reductionsV4_phid_UD, i_gamma_f2, 17, true, true),"ISOSPIN32");

            TIME(corrB18.B_diagrams(reductionsV3_phid_U, reductionsV4_phid_UD, i_gamma_f2, 18, true, true),"ISOSPIN32");

            TIME(corrB19.B_diagrams(reductionsV3_phid_U, reductionsV2_phid_UD, i_gamma_f2, 19, true, true),"ISOSPIN32");

            TIME(corrB20.B_diagrams(reductionsV3_phid_U, reductionsV2_phid_UD, i_gamma_f2, 20, true, true),"ISOSPIN32");
	  }//gamma_f2
	
	         
	  //## B

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
  

//          while(not threads.empty()) {threads.back().join(); threads.pop_back();}

          TIME(produceOutput(corrB1, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB2, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
	  TIME(produceOutput(corrB3, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB4, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB5, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB6, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB7, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB8, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB9, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB10, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB11, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrB12, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32");
	  TIME(produceOutput(corrB13, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
	  TIME(produceOutput(corrB14, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
          TIME(produceOutput(corrB15, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
          TIME(produceOutput(corrB16, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
	  TIME(produceOutput(corrB17, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
          TIME(produceOutput(corrB18, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
          TIME(produceOutput(corrB19, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");
          TIME(produceOutput(corrB20, outfilename, "4pt", 1, 1, NULL),"ISOSPIN32=");

	}//i_mpf2


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

        spropagator_V6.copy(*stochastic_oet_prop_u_zero_mom[sourcePositions[isource][DIM_T]],HOST);
        spropagator_V6.load();

        TIME(reductionsV4_phiui2_UUT.V4( spropagator_V6, glist_sink_delta, propUP, propUP, true),"ISOSPIN12");
        TIME(reductionsV4_phiui2_UDT.V4( spropagator_V6, glist_sink_delta, propUP, propDN, true),"ISOSPIN12");
	TIME(reductionsV4_phiui2_DUT.V4( spropagator_V6, glist_sink_delta, propDN, propUP, true),"ISOSPIN12");


	//THREAD(reductionsV4_phiui2_UU.writeHDF5("V4redforTdiagram"));

        TIME(reductionsV2_phiui2_UUT.V2( spropagator_V6, glist_sink_delta, propUP, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_UDT.V2( spropagator_V6, glist_sink_delta, propUP, propDN, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_DUT.V2( spropagator_V6, glist_sink_delta, propDN, propUP, true),"ISOSPIN12");


	//THREAD(reductionsV2_phiui2_UUT.writeHDF5("V2redforTdiagram"));

	//Factors for the Z diagram
	TIME(reductionsV4_phiui2_DU.V4( spropagator_V6, glist_sink_nucleon, propDN, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_DU.V2( spropagator_V6, glist_sink_nucleon, propDN, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phiui2_DD.V2( spropagator_V6, glist_sink_nucleon, propDN, propDN, true),"ISOSPIN12");	

        spropagator_V6.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);
        spropagator_V6.load();
 
        TIME(reductionsV2_phidi2_UD.V2( spropagator_V6, glist_sink_nucleon, propUP, propDN, true),"ISOSPIN12");
        TIME(reductionsV2_phidi2_UU.V2( spropagator_V6, glist_sink_nucleon, propUP, propUP, true),"ISOSPIN12");
        TIME(reductionsV4_phidi2_DU.V4( spropagator_V6, glist_sink_nucleon, propDN, propUP, true),"ISOSPIN12");
        TIME(reductionsV2_phidi2_UUT.V2( spropagator_V6, glist_sink_delta,propUP, propUP, true),"ISOSPIN12");
	TIME(reductionsV4_phidi2_UUT.V4( spropagator_V6, glist_sink_delta, propUP, propUP, true),"ISOSPIN32");

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


          PLEGMA_ScattCorrelator<float> corrW1(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW2(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW3(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW4(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW5(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW6(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW7(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW8(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW9(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW10(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW11(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW12(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW13(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW14(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW15(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW16(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW17(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW18(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW19(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW20(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW21(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW22(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW23(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW24(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW25(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW26(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW27(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW28(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW29(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW30(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW31(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW32(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW33(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW34(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW35(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW36(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList);
	  PLEGMA_ScattCorrelator<float> corrD1if12(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrD1if34(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrD1if56(sourcePositions[isource], filtered_sourcemomentumList);

	  
	  //T diagrams
          std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
          std::vector<std::vector<int>> mpi2_filt;
          mpi2_filt.assign(mptot_filt.size(),momentum_i2);
          momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});


          PLEGMA_ScattCorrelator<float> corrT1(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT2(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT3(sourcePositions[isource], list_mpi2ptot);

          PLEGMA_ScattCorrelator<float> corrT7(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT9(sourcePositions[isource], list_mpi2ptot);

          PLEGMA_ScattCorrelator<float> corrT11(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT12(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT13(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT14(sourcePositions[isource], list_mpi2ptot);

          PLEGMA_ScattCorrelator<float> corrT15(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT17(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT19(sourcePositions[isource], list_mpi2ptot);

          PLEGMA_ScattCorrelator<float> corrT21(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT22(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT23(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT24(sourcePositions[isource], list_mpi2ptot);

          PLEGMA_ScattCorrelator<float> corrT25(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT26(sourcePositions[isource], list_mpi2ptot);


          corrT1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "T1oet");
          corrT2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "T2oet");
          corrT3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "T3oet");

          corrT7.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T7oet");
          corrT9.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T9oet");


          corrT11.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T11oet");
          corrT12.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T12oet");
          corrT13.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T13oet");
          corrT14.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T14oet");

          corrT15.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T15oet");
          corrT17.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T17oet");
          corrT19.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T19oet");

          corrT21.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T21oet");
          corrT22.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T22oet");
          corrT23.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T23oet");
          corrT24.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T24oet");

          corrT25.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T25oet");
          corrT26.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "T26oet");

          
	  //initialize diagrams
          corrW1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W1");
          corrW2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W2");
          corrW3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W3");
          corrW4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W4");

          corrW5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W5");
          corrW6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W6");
          corrW7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W7");
          corrW8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W8");
          corrW9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W9");
          corrW10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W10");
          corrW11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W11");
          corrW12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W12");
          corrW13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W13");
          corrW14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W14");
          corrW15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W15");
          corrW16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W16");
          corrW17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W17");
          corrW18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W18");
          corrW19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W19");
          corrW20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W20");
          corrW21.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W21");
          corrW22.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W22");
          corrW23.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W23");
	  corrW24.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W24");
          corrW25.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W25");
	  corrW26.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W26");
          corrW27.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W27");
          corrW28.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W28");
          corrW29.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W29");
          corrW30.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W30");
          corrW31.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W31");
          corrW32.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W32");
          corrW33.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W33");
          corrW34.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W34");
          corrW35.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W35");
          corrW36.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W36");


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
 

          }


	  std::shared_ptr<float> Phi0;

	  /********************************************************************
	   *
	   * WARNING: The code assumes you have the same list of momentum for pi2 and
	   * pf2
	   *
	   * ******************************************************************/

	  spropagator_V6.copy(*stochastic_oet_prop_u_fini_mom[i_mpi2],HOST);

	  spropagator_V6.unload();

          Phi0 = spropagator_V6.getPointSource(actualSource,HOST);

          spropagator_V6.load();



	  TIME(corrT1.T_diagrams_oet(reductionsV4_phiui2_UUT, Phi0, 1, true),"ISOSPIN32");

	  TIME(corrT2.T_diagrams_oet(reductionsV2_phiui2_UUT, Phi0, 2, true),"ISOSPIN32");

          TIME(corrT3.T_diagrams_oet(reductionsV2_phiui2_UUT, Phi0, 3, true),"ISOSPIN32");

          TIME(corrT7.T_diagrams_oet(reductionsV2_phiui2_UDT, Phi0, 7, true),"ISOSPIN32");

          TIME(corrT9.T_diagrams_oet(reductionsV2_phiui2_UDT, Phi0, 9, true),"ISOSPIN32");

          TIME(corrT11.T_diagrams_oet(reductionsV4_phiui2_UDT, Phi0, 11, true),"ISOSPIN32");

          TIME(corrT12.T_diagrams_oet(reductionsV4_phiui2_UDT, Phi0, 12, true),"ISOSPIN32");

          TIME(corrT13.T_diagrams_oet(reductionsV2_phiui2_DUT, Phi0, 13, true),"ISOSPIN32");

	  TIME(corrT14.T_diagrams_oet(reductionsV2_phiui2_DUT, Phi0, 14, true),"ISOSPIN32");

	  TIME(corrT19.T_diagrams_oet(reductionsV4_phidi2_UUT, Phi0, 19, true),"ISOSPIN32");

          TIME(corrT25.T_diagrams_oet(reductionsV2_phidi2_UUT, Phi0, 25, true),"ISOSPIN32");

          TIME(corrT26.T_diagrams_oet(reductionsV2_phidi2_UUT, Phi0, 26, true),"ISOSPIN32");


          spropagator_V6.copy(*stochastic_oet_prop_d_fini_mom[i_mpi2],HOST);

          spropagator_V6.unload();

          Phi0 = spropagator_V6.getPointSource(actualSource,HOST);

          spropagator_V6.load();


          TIME(corrT15.T_diagrams_oet(reductionsV2_phiui2_UDT, Phi0, 15, true),"ISOSPIN32");

          TIME(corrT17.T_diagrams_oet(reductionsV2_phiui2_UDT, Phi0, 17, true),"ISOSPIN32");

	  TIME(corrT21.T_diagrams_oet(reductionsV4_phiui2_DUT, Phi0, 21, true),"ISOSPIN32");

	  TIME(corrT22.T_diagrams_oet(reductionsV4_phiui2_DUT, Phi0, 22, true),"ISOSPIN32");

          TIME(corrT23.T_diagrams_oet(reductionsV2_phiui2_DUT, Phi0, 23, true),"ISOSPIN32");

          TIME(corrT24.T_diagrams_oet(reductionsV2_phiui2_DUT, Phi0, 24, true),"ISOSPIN32");


	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
	  
	  TIME(corrT1.apply_phase(),"ISOSPIN32");
	  TIME(corrT1.apply_sign("T"),"ISOSPIN32");
	  TIME(corrT1.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN32");
	  TIME(corrT1.writeHDF5(outfilename),"ISOSPIN32");


	  TIME(corrT2.apply_phase(),"ISOSPIN32");
          TIME(corrT2.apply_sign("T"),"ISOSPIN32");
          TIME(corrT2.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN32");
          TIME(corrT2.writeHDF5(outfilename),"ISOSPIN32");


	  TIME(corrT3.apply_phase(),"ISOSPIN32");
          TIME(corrT3.apply_sign("T"),"ISOSPIN32");
          TIME(corrT3.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN32");
          TIME(corrT3.writeHDF5(outfilename),"ISOSPIN32");

          TIME(corrT7.apply_phase(),"ISOSPIN12");
          TIME(corrT7.apply_sign("T"),"ISOSPIN12");
          TIME(corrT7.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT7.writeHDF5(outfilename),"ISOSPIN12");
	  
	  TIME(corrT9.apply_phase(),"ISOSPIN12");
          TIME(corrT9.apply_sign("T"),"ISOSPIN12");
          TIME(corrT9.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT9.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT11.apply_phase(),"ISOSPIN12");
          TIME(corrT11.apply_sign("T"),"ISOSPIN12");
          TIME(corrT11.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT11.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT12.apply_phase(),"ISOSPIN12");
          TIME(corrT12.apply_sign("T"),"ISOSPIN12");
          TIME(corrT12.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT12.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT13.apply_phase(),"ISOSPIN12");
          TIME(corrT13.apply_sign("T"),"ISOSPIN12");
          TIME(corrT13.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT13.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT14.apply_phase(),"ISOSPIN12");
          TIME(corrT14.apply_sign("T"),"ISOSPIN12");
          TIME(corrT14.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT14.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT15.apply_phase(),"ISOSPIN12");
          TIME(corrT15.apply_sign("T"),"ISOSPIN12");
          TIME(corrT15.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT15.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT17.apply_phase(),"ISOSPIN12");
          TIME(corrT17.apply_sign("T"),"ISOSPIN12");
          TIME(corrT17.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT17.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT19.apply_phase(),"ISOSPIN12");
          TIME(corrT19.apply_sign("T"),"ISOSPIN12");
          TIME(corrT19.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT19.writeHDF5(outfilename),"ISOSPIN12");

	  TIME(corrT21.apply_phase(),"ISOSPIN12");
          TIME(corrT21.apply_sign("T"),"ISOSPIN12");
          TIME(corrT21.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT21.writeHDF5(outfilename),"ISOSPIN12");

	  TIME(corrT22.apply_phase(),"ISOSPIN12");
          TIME(corrT22.apply_sign("T"),"ISOSPIN12");
          TIME(corrT22.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT22.writeHDF5(outfilename),"ISOSPIN12");

	  TIME(corrT23.apply_phase(),"ISOSPIN12");
          TIME(corrT23.apply_sign("T"),"ISOSPIN12");
          TIME(corrT23.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT23.writeHDF5(outfilename),"ISOSPIN12");

	  TIME(corrT24.apply_phase(),"ISOSPIN12");
          TIME(corrT24.apply_sign("T"),"ISOSPIN12");
          TIME(corrT24.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT24.writeHDF5(outfilename),"ISOSPIN12");

	  TIME(corrT25.apply_phase(),"ISOSPIN12");
          TIME(corrT25.apply_sign("T"),"ISOSPIN12");
          TIME(corrT25.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT25.writeHDF5(outfilename),"ISOSPIN12");

          TIME(corrT26.apply_phase(),"ISOSPIN12");
          TIME(corrT26.apply_sign("T"),"ISOSPIN12");
          TIME(corrT26.applyBoundaryConditions( true,  1, NULL ),"ISOSPIN12");
          TIME(corrT26.writeHDF5(outfilename),"ISOSPIN12");


	  //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
	  spropagator_V6.copy(*stochastic_oet_prop_u_zero_mom[sourcePositions[isource][DIM_T]],HOST);
          spropagator_V6.load();

	  if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
	    
	    TIME(corrPPUP.P_diagrams( spropagator_V6, stochastic_oet_prop_u_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

            TIME(corrP0DN.P_diagrams( spropagator_V6, stochastic_oet_prop_d_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");
	 
	    TIME(corrM.M_diagrams( corrNP, spropagator_V6, stochastic_oet_prop_u_fini_mom_source_to_sink ),"ISOSPIN32");

            TIME(corrD1if12.M_diagrams( corrNP, spropagator_V6, stochastic_oet_prop_d_fini_mom_source_to_sink),"ISOSPIN12");

	    TIME(corrD1if56.M_diagrams( corrN0, spropagator_V6, stochastic_oet_prop_u_fini_mom_source_to_sink),"ISOSPIN12");
	  
	  }
	  else{

	    PLEGMA_Vector<float> stmp;
	    stmp.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);
            stmp.load();

	    TIME(corrPPUP.P_diagrams( spropagator_V6, spropagator_V6, i_mpi2, true),"ISOSPIN32");
            TIME(corrP0DN.P_diagrams( spropagator_V6, stmp, i_mpi2, true),"ISOSPIN32");
	    TIME(corrM.M_diagrams( corrNP, spropagator_V6, spropagator_V6 ),"ISOSPIN32");
            TIME(corrD1if12.M_diagrams( corrNP, spropagator_V6, stmp),"ISOSPIN12");

            TIME(corrD1if56.M_diagrams( corrN0, spropagator_V6, spropagator_V6),"ISOSPIN12");
	  
	  }

	  spropagator_V6.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);
          spropagator_V6.load();

          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){

            TIME(corrPPDN.P_diagrams( spropagator_V6, stochastic_oet_prop_d_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

            TIME(corrP0UP.P_diagrams( spropagator_V6, stochastic_oet_prop_u_fini_mom_source_to_sink, i_mpi2, true),"ISOSPIN32");

	    TIME(corrD1if34.M_diagrams( corrNP, spropagator_V6, stochastic_oet_prop_u_fini_mom_source_to_sink ),"ISOSPIN12");

          }
          else{

            PLEGMA_Vector<float> stmp;
            stmp.copy(*stochastic_oet_prop_u_zero_mom[sourcePositions[isource][DIM_T]],HOST);
            stmp.load();

            TIME(corrPPDN.P_diagrams( spropagator_V6, spropagator_V6, i_mpi2, true),"ISOSPIN32");
            TIME(corrP0UP.P_diagrams( spropagator_V6, stmp, i_mpi2, true),"ISOSPIN32");
            TIME(corrD1if34.M_diagrams( corrNP, spropagator_V6, stmp ),"ISOSPIN12");


          }


       

          std::vector<std::vector<int>> mpf2_forfixmpi2 = filtered_sourcemomentumList.uniq_p(2);


          for  (int i_mpf2=0; i_mpf2<mpf2_forfixmpi2.size(); ++i_mpf2){

            for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2) {

              for (int i_gamma_f2=0; i_gamma_f2 < glist_sink_meson.size(); ++i_gamma_f2) {

		PLEGMA_Vector<float> szerotmp;

		szerotmp.copy(*stochastic_oet_prop_u_zero_mom[sourcePositions[isource][DIM_T]],HOST);
		szerotmp.load();

                spropagator_V6.copy(*stochastic_oet_prop_d_fini_mom[i_mpf2],HOST);
	        spropagator_V6.load();

                reductionsV6_W_phiui2_phidf2U.V6_RED(szerotmp, spropagator_V6, glist_sink_nucleon,propUP,1,2,true);

	        reductionsV6_W_phidf2_phiui2D.V6_RED(spropagator_V6, szerotmp, glist_sink_nucleon,propDN,1,2,true);

		reductionsV6_W_phiui2phidf2_U.V6_RED(szerotmp, spropagator_V6, glist_sink_nucleon,propUP,0,1,true);

		reductionsV6_W_phidf2phiui2_D.V6_RED(spropagator_V6,szerotmp,glist_sink_nucleon,propUP,0,1,true);

		szerotmp.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);
                szerotmp.load();
		reductionsV6_W_phidf2_phidi2U.V6_RED(spropagator_V6, szerotmp, glist_sink_nucleon,propUP,1,2,true);
                reductionsV6_W_phidi2_phidf2U.V6_RED(szerotmp, spropagator_V6, glist_sink_nucleon,propUP,1,2,true);

                spropagator_V6.copy(*stochastic_oet_prop_u_fini_mom[i_mpf2],HOST);
                spropagator_V6.load();
                szerotmp.copy(*stochastic_oet_prop_u_zero_mom[sourcePositions[isource][DIM_T]],HOST);
                szerotmp.load();


		reductionsV6_W_phiuf2_phiui2D.V6_RED(spropagator_V6, szerotmp, glist_sink_nucleon,propUP,1,2,true);
                reductionsV6_W_phiui2_phiuf2D.V6_RED(szerotmp, spropagator_V6, glist_sink_nucleon,propUP,1,2,true);


                szerotmp.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);
                szerotmp.load();

                spropagator_V6.copy(*stochastic_oet_prop_u_fini_mom[i_mpf2],HOST);
                spropagator_V6.load();


                reductionsV6_W_phiuf2phidi2_U.V6_RED(szerotmp, spropagator_V6, glist_sink_nucleon,propUP,0,1,true);
		reductionsV6_W_phiuf2_phidi2U.V6_RED(szerotmp, spropagator_V6, glist_sink_nucleon,propUP,1,2,true);

		szerotmp.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);
                szerotmp.load();
		  
                //reductionsV6.writeHDF5("V6redforWdiagram12_mpi2_"+std::to_string(i_mpi2)+"mpf2_"+std::to_string(i_mpf2));
                spropagator_V6.copy(*stochastic_oet_prop_u_fini_mom[i_mpi2],HOST);

                spropagator_V6.unload();

                Phi0 = spropagator_V6.getPointSource(actualSource,HOST);

                spropagator_V6.load();

//	        reductionsV6_W_phiui2_phidf2U.writeHDF5("V6redforWdiagram12_mpi2_"+std::to_string(i_mpi2)+"mpf2_"+std::to_string(i_mpf2));

		TIME(corrW1.W_diagrams_oet(reductionsV6_W_phiui2_phidf2U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 1, true),"ISOSPIN32");

		TIME(corrW2.W_diagrams_oet(reductionsV6_W_phiui2_phidf2U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 2, true),"ISOSPIN32");

		TIME(corrW3.W_diagrams_oet(reductionsV6_W_phiui2phidf2_U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 3, true),"ISOSPIN32");

		TIME(corrW4.W_diagrams_oet(reductionsV6_W_phiui2phidf2_U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 4, true),"ISOSPIN32");

                TIME(corrW9.W_diagrams_oet(reductionsV6_W_phiuf2phidi2_U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 9, true),"ISOSPIN12");

                TIME(corrW10.W_diagrams_oet(reductionsV6_W_phiuf2_phidi2U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 10, true),"ISOSPIN12");

                TIME(corrW11.W_diagrams_oet(reductionsV6_W_phiuf2phidi2_U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 11, true),"ISOSPIN12");

		TIME(corrW12.W_diagrams_oet(reductionsV6_W_phiuf2_phidi2U, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 12, true),"ISOSPIN12");

                TIME(corrW17.W_diagrams_oet(reductionsV6_W_phiuf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 17, true),"ISOSPIN12");

		TIME(corrW18.W_diagrams_oet(reductionsV6_W_phiui2_phiuf2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 18, true),"ISOSPIN12");

		TIME(corrW19.W_diagrams_oet(reductionsV6_W_phiuf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 19, true),"ISOSPIN12");

                TIME(corrW20.W_diagrams_oet(reductionsV6_W_phiui2_phiuf2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 20, true),"ISOSPIN12");

		TIME(corrW21.W_diagrams_oet(reductionsV6_W_phiui2phidf2_U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 21, true),"ISOSPIN12");

		TIME(corrW22.W_diagrams_oet(reductionsV6_W_phiui2_phidf2U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 22, true),"ISOSPIN12");

		TIME(corrW23.W_diagrams_oet(reductionsV6_W_phiui2phidf2_U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 23, true),"ISOSPIN12");

		TIME(corrW24.W_diagrams_oet(reductionsV6_W_phiui2_phidf2U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 24, true),"ISOSPIN12");

		TIME(corrW25.W_diagrams_oet(reductionsV6_W_phidf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 25, true),"ISOSPIN32");

                TIME(corrW26.W_diagrams_oet(reductionsV6_W_phidf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 26, true),"ISOSPIN32");

                TIME(corrW27.W_diagrams_oet(reductionsV6_W_phidf2phiui2_D, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 27, true),"ISOSPIN32");

                TIME(corrW28.W_diagrams_oet(reductionsV6_W_phidf2phiui2_D, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 28, true),"ISOSPIN32");

                TIME(corrW33.W_diagrams_oet(reductionsV6_W_phidf2_phidi2U, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 33, true),"ISOSPIN32");

                TIME(corrW34.W_diagrams_oet(reductionsV6_W_phidf2_phidi2U, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 34, true),"ISOSPIN32");

                TIME(corrW35.W_diagrams_oet(reductionsV6_W_phidi2_phidf2U, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 35, true),"ISOSPIN32");

		TIME(corrW36.W_diagrams_oet(reductionsV6_W_phidi2_phidf2U, Phi0, stochastic_oet_prop_d_zero_mom,  i_mpf2, 36, true),"ISOSPIN32");


                spropagator_V6.copy(*stochastic_oet_prop_d_fini_mom[i_mpi2],HOST);

                spropagator_V6.unload();

                Phi0 = spropagator_V6.getPointSource(actualSource,HOST);

                spropagator_V6.load();

                TIME(corrW5.W_diagrams_oet(reductionsV6_W_phiuf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 5, true),"ISOSPIN12");

                TIME(corrW6.W_diagrams_oet(reductionsV6_W_phiui2_phiuf2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 6, true),"ISOSPIN12");
		
		TIME(corrW7.W_diagrams_oet(reductionsV6_W_phiuf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 7, true),"ISOSPIN12");
                
		TIME(corrW8.W_diagrams_oet(reductionsV6_W_phiui2_phiuf2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 12, true),"ISOSPIN12");

		TIME(corrW13.W_diagrams_oet(reductionsV6_W_phiui2phidf2_U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2,13, true),"ISOSPIN12");

                TIME(corrW14.W_diagrams_oet(reductionsV6_W_phiui2_phidf2U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 14, true),"ISOSPIN12");

                TIME(corrW15.W_diagrams_oet(reductionsV6_W_phiui2phidf2_U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 15, true),"ISOSPIN12");
 
		TIME(corrW16.W_diagrams_oet(reductionsV6_W_phiui2_phidf2U, Phi0, stochastic_oet_prop_u_zero_mom, i_mpf2, 16, true),"ISOSPIN12");

                TIME(corrW29.W_diagrams_oet(reductionsV6_W_phidf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 29, true),"ISOSPIN12");

		TIME(corrW30.W_diagrams_oet(reductionsV6_W_phidf2_phiui2D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 30, true),"ISOSPIN12");

		TIME(corrW31.W_diagrams_oet(reductionsV6_W_phidf2phiui2_D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 31, true),"ISOSPIN12");
		
		TIME(corrW32.W_diagrams_oet(reductionsV6_W_phidf2phiui2_D, Phi0, stochastic_oet_prop_d_zero_mom, i_mpf2, 32, true),"ISOSPIN12");

	      }

	    }

	  }

	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";


          TIME(produceOutput(corrW1, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrW2, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrW3, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrW4, outfilename, "4pt",1, NULL),"ISOSPIN32");
          TIME(produceOutput(corrW5, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW6, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW7, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW8, outfilename, "4pt",1, NULL),"ISOSPIN12");

          TIME(produceOutput(corrW9, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW10, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW11, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW12, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW13, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW14, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW15, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW16, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW17, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW18, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW19, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW20, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW21, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW22, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW23, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW24, outfilename, "4pt",1, NULL),"ISOSPIN12");
	  TIME(produceOutput(corrW25, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW26, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW27, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW28, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW29, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW30, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW31, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW32, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW33, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW34, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW35, outfilename, "4pt",1, NULL),"ISOSPIN12");
          TIME(produceOutput(corrW36, outfilename, "4pt",1, NULL),"ISOSPIN12");



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


              //THREAD(reductionsV3.writeHDF5("V3redforZdiagram"));


            }
            else{

              spropagator_V6.copy(*stochastic_oet_prop_u_zero_mom[sourcePositions[isource][DIM_T]],HOST);;
              spropagator_V6.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);

              TIME(reductionsV3_phiui2_U.V3( spropagator_V6, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");
              TIME(reductionsV3_phiui2_D.V3( spropagator_V6, gamma_5_t_sinkmeson, propDN, true),"ISOSPIN32");

	      spropagator_V6.copy(*stochastic_oet_prop_d_zero_mom[sourcePositions[isource][DIM_T]],HOST);;
              spropagator_V6.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);

              TIME(reductionsV3_phidi2_U.V3( spropagator_V6, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");

              //THREAD(reductionsV3.writeHDF5("V3redforZdiagram"));

            }



            TIME(corrZ1.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phiui2_DU, i_gamma_i2, 1 ),"ISOSPIN32");
            TIME(corrZ2.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phiui2_DU, i_gamma_i2, 2 ),"ISOSPIN32");
            TIME(corrZ3.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DU, i_gamma_i2, 3 ),"ISOSPIN32");
            TIME(corrZ4.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DU, i_gamma_i2, 4 ),"ISOSPIN32");
            TIME(corrZ5.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV2_phiui2_DU, i_gamma_i2, 5 ),"ISOSPIN12");
            TIME(corrZ6.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV4_phiui2_DU, i_gamma_i2, 6 ),"ISOSPIN12");
            TIME(corrZ7.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV2_phiui2_DU, i_gamma_i2, 7 ),"ISOSPIN12");
            TIME(corrZ8.Z_diagrams_without_dilution( reductionsV3_phidi2_U, reductionsV4_phiui2_DU, i_gamma_i2, 8 ),"ISOSPIN12");
            TIME(corrZ9.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phidi2_UU, i_gamma_i2, 9 ),"ISOSPIN12");
            TIME(corrZ10.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phidi2_UU, i_gamma_i2, 10 ),"ISOSPIN12");
            TIME(corrZ11.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phiui2_DU, i_gamma_i2, 11 ),"ISOSPIN12");
            TIME(corrZ12.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV4_phiui2_DU, i_gamma_i2, 12 ),"ISOSPIN12");
            TIME(corrZ13.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV2_phiui2_DU, i_gamma_i2, 13 ),"ISOSPIN12");
            TIME(corrZ14.Z_diagrams_without_dilution( reductionsV3_phiui2_D, reductionsV4_phiui2_DU, i_gamma_i2, 14 ),"ISOSPIN12");
	    TIME(corrZ15.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DD, i_gamma_i2, 15 ),"ISOSPIN12");
            TIME(corrZ16.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phiui2_DD, i_gamma_i2, 16 ),"ISOSPIN12");
            TIME(corrZ17.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phidi2_UD, i_gamma_i2, 17 ),"ISOSPIN12");
            TIME(corrZ18.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV2_phidi2_UD, i_gamma_i2, 18 ),"ISOSPIN12");
            TIME(corrZ19.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phidi2_DU, i_gamma_i2, 19 ),"ISOSPIN12");
            TIME(corrZ20.Z_diagrams_without_dilution( reductionsV3_phiui2_U, reductionsV4_phidi2_DU, i_gamma_i2, 20 ),"ISOSPIN12");



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


    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
      stochastic_oet_prop_u_zero_mom.pop_back();
      stochastic_oet_prop_d_zero_mom.pop_back();
    }
    
    for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){
      
      auto &momentum_f2 = mpf2[i_mpf2];
      stochastic_oet_prop_u_fini_mom.pop_back();
      stochastic_oet_prop_d_fini_mom.pop_back();   

    }

//     while(not threads.empty()) {threads.back().join(); threads.pop_back();}


  } 

  finalize();
  
  return 0;
}
