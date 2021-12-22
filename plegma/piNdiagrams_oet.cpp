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
#define TIME(fnc,isospin)  runtime.push_back(MPI_Wtime()); fnc;                 \
    PLEGMA_printf("TIME "#isospin" for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
    runtime.pop_back()

  //std::vector<std::thread> threads;
  //#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
  //#define THREAD(fnc) TIME(fnc)

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
    int n_coherent_source;
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
    HGC_options->set("n_coherent_source", "Flag for switching time-dilution in stochastic propagators", verbosity, n_coherent_source);
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

      for(int i=0; i< HGC_totalL[DIM_T]; ++i) {
	stochastic_oet_prop_d_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
      }
      //length of pf2
      int length_fini_mom=(sourcemomentumList.uniq_p(2)).size();
      for(int i=0; i< length_fini_mom; ++i) {
	stochastic_oet_prop_d_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));
      }


      PLEGMA_printf("Start producing stochastic oet propagators\n");
      //Note that we replace the f1<-f2 DN propagator with a stochastic one
      //phi^f2(xf1)
      //In the same time we replace 
      //U(xf2,xi2) with gamma5 U(xi2,xf2) gamma5 = gamma5 phi*^(f2)(i2) gamma5


      std::vector<std::vector<int>> mpf2 = sourcemomentumList.uniq_p(2);

      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(2);

     
      if (read_stochastic_oet==false){
	
	PLEGMA_Vector<double> vectorAuxD1(BOTH);//For storing the source (rotated and smeared)
	PLEGMA_Vector<double> vectorAuxD2(BOTH);//For storing the propagotor for the time-slices
	PLEGMA_Vector<double> vectorInOut; //temporary vector using in solve

	
	// ensuring mu negative
          if(mu<0) {
            mu*=-1.;
            solver.UpdateSolver();
          }


	//Step(1) Creating the time-diluted stochastic source

	PLEGMA_Vector<double> vectorSource_oet;
	vectorSource_oet.randInit(rand_seed1);
	vectorSource_oet.stochastic_Z(nroots);

	for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){

	  auto &momentum_f2 = mpf2[i_mpf2];

	  vectorAuxD1.copy(vectorSource_oet);

	  std::vector<int> tmp_4Dmom= momentum_f2 ;
	  tmp_4Dmom.push_back(0);
	  
	  //Step(1) multiply with the momentum phase
	  vectorAuxD1.mulMomentumPhases(tmp_4Dmom,-1);

	  //In vectorAuxD2 we store the results for the inversion for sink to sink
	  vectorAuxD2.scale(0.0);

	  //Step(2) Smearing all the time slice
	  TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");
   
	  //Step(3) We rotate the source to the physical basis
	  TIME(vectorAuxD1.rotateToPhysicalBasis(vectorInOut,+1),"ISOSPIN32");
   
	  for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

	    //Step(4) pick out a particular timeslice from the source
	    vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);

	    //Step(5) Solve
	    TIME(solver.solve(vectorInOut, vectorInOut),"ISOSPIN32");

	    //Step(6) absorbing the particular timeslice to a 4d vector
	    vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);

	    if ((momentum_f2[0] == 0) && (momentum_f2[1] == 0) && (momentum_f2[2] == 0)){

	      //Step(7) We rotate back the propagator to the physical basis
	      TIME(vectorAuxD1.rotateToPhysicalBasis(vectorInOut,+1),"ISOSPIN32");

	      //Step(8) Smearing all the time slice in the propagator
	      TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

	      //Step(9) Save the propagator to the disk
	      {
	       PLEGMA_Vector<float> vectorAuxF;
	       vectorAuxF.copy(vectorInOut);
	       vectorAuxF.unload();
	       vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_oet_stoch_time_"+std::to_string(timeidx)+"_"+confnumber);
	      }

	      //Step(9) Save the propagator to the host memory
	      vectorInOut.unload();
	      stochastic_oet_prop_d_zero_mom[timeidx]->copy(vectorInOut,HOST);
	      vectorInOut.load();
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
	    vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_oet_stoch_mom"+std::to_string(i_mpf2)+"_"+confnumber);
	  }

	  //Step(13) Save the propagator to the host memory
	  vectorAuxD2.unload();
	  stochastic_oet_prop_d_fini_mom[i_mpf2]->copy(vectorAuxD2,HOST);
	  vectorAuxD2.load();
	}
	
      }
      else{
	PLEGMA_Vector<float> vectorRead(BOTH);
	for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){
	  auto &momentum_f2 = mpf2[i_mpf2];
	  if ((momentum_f2[0] == 0) || (momentum_f2[1] == 0) || (momentum_f2[2] == 0)){

	    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

	      std::string inputfilename=outfile_V+"globalTfulltimedilution_propagator_oet_stoch_time_"+std::to_string(timeidx)+"_"+confnumber;
	      PLEGMA_printf("Read stochastic oet source (zero momentum source to sink) from: %s\n",inputfilename.c_str());
	  
	      vectorRead.readFile(inputfilename,LIME_FORMAT);
	  
	      stochastic_oet_prop_d_zero_mom[timeidx]->copy(vectorRead,HOST);
	
	    }
	  }
	  
	  std::string inputfilename=outfile_V+"globalTfulltimedilution_propagator_oet_stoch_mom_"+std::to_string(i_mpf2)+"_"+confnumber;

	  PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
	    
	  vectorRead.readFile(inputfilename,LIME_FORMAT);
	    
	  stochastic_oet_prop_d_fini_mom[i_mpf2]->copy(vectorRead,HOST);
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



	//Creating look up tables for the coherent time-slice sources
	int *coherent_source_table=NULL;
	int *coherent_source_table_timeslice=NULL;
	int **coherent_look_up_table=NULL;

	coherent_source_table=(int *)malloc(sizeof(int)*n_coherent_source);
	for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){
	  coherent_source_table[i_coherent_source]=(sourcePositions[isource][DIM_T]+i_coherent_source*HGC_totalL[DIM_T]/n_coherent_source)%HGC_totalL[DIM_T];
	        site source=site({0,0,0,sourcePositions[isource][DIM_T]});
}
	coherent_source_table_timeslice=(int *)malloc(sizeof(int)*HGC_totalL[DIM_T]);
	for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){
	  for (int i=0; i<=HGC_totalL[DIM_T]/(2*n_coherent_source); ++i){
	    coherent_source_table_timeslice[(coherent_source_table[i_coherent_source]+i)%HGC_totalL[DIM_T]]=i_coherent_source;
	  }
	  for (int i=1; i<(HGC_totalL[DIM_T]/(2*n_coherent_source));++i){
	    coherent_source_table_timeslice[(coherent_source_table[i_coherent_source]-i+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]]=i_coherent_source;
	  }
	}
	coherent_look_up_table=(int **)malloc(sizeof(int*)*n_coherent_source);
	for (int i=0; i<n_coherent_source; ++i){
	  int k=0;
	  coherent_look_up_table[i]=(int *)malloc(sizeof(int)*HGC_totalL[DIM_T]/n_coherent_source);
	  for (int j=0; j<=HGC_totalL[DIM_T]/(2*n_coherent_source); ++j){
	    coherent_look_up_table[i][k]=(coherent_source_table[i]+j)%HGC_totalL[DIM_T];
	    k++;
	  }
	  for (int j=1;j<(HGC_totalL[DIM_T]/(2*n_coherent_source));++j){
	    coherent_look_up_table[i][k]=(coherent_source_table[i]-j+HGC_totalL[DIM_T])%HGC_totalL[DIM_T];
	    k++;    
	  }
	}

	//Calculations for source-position Calculations for source-position 
	PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		      isource, sourcePositions[isource][0], sourcePositions[isource][1],
		      sourcePositions[isource][2], sourcePositions[isource][3]);
	for(int icoherentsource; icoherentsource < n_coherent_source; ++icoherentsource){
	  PLEGMA_printf("\n ### Calculations for coherent-source-number %d - timeslice %03d begin now ###\n\n",
		      icoherentsource, sourcePositions[isource][3]+icoherentsource*HGC_totalL[DIM_T]/n_coherent_source);
	}
   
	//Create Propagator
	PLEGMA_Propagator<float> propUP(BOTH); //To be saved for all the coherent sources.
	PLEGMA_Propagator<float> propDN(BOTH);

	PLEGMA_Propagator<float> propTS(BOTH);

	std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
	momList list_mpf1(1,{mpf1,},{0,});
	PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1 );
	TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"),"ISOSPIN32");

#ifdef PLEGMA_SCATTERING_SPIN12
	PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1 );
	TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"),"ISOSPIN12");
#endif


	for(int icoherentsource=0;icoherentsource < n_coherent_source; ++icoherentsource)
	{

	  PLEGMA_Propagator<float> propUP_coherent;
	  PLEGMA_Propagator<float> propDN_coherent;

	  asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], coherent_source_table[icoherentsource]);
	  std::string sourcepositiontext_inside= (std::string)"_" + ssource; 
	  free(ssource);

	  PLEGMA_Gauge3D<double> smearedGauge3D;
	  smearedGauge3D.absorb(smearedGauge, coherent_source_table[icoherentsource]);

	  // ensuring mu positive
	  if(mu<0) {
	    mu*=-1.;
	    solver.UpdateSolver();
	  }

	  site src;
	  src=site({sourcePositions[isource][0],
		    sourcePositions[isource][1],
		    sourcePositions[isource][2],
		    coherent_source_table[icoherentsource]});
	  for(int isc = 0 ; isc < 12 ; isc++){
	    PLEGMA_Vector<double> vectorInOut;
	    PLEGMA_Vector<float>  vectorAuxF;
	    PLEGMA_Vector<double> vectorAuxD;
	    { // Smearing the source
	      PLEGMA_Vector3D<double> vector1, vector2;
	      vector1.pointSource(src, isc/3, isc%3, DEVICE);
	      TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
	      vectorInOut.absorb(vector2,coherent_source_table[icoherentsource]);
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
	    propUP_coherent.absorb(vectorAuxF, isc/3, isc%3);

	  }
	  for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
	    propUP.absorbTimeslice(propUP_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
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
	      vectorInOut.absorb(vector2,coherent_source_table[icoherentsource]);
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

	    propDN_coherent.absorb(vectorAuxF, isc/3, isc%3);
	  }
	  for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
	    propDN.absorbTimeslice(propDN_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
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
	
	  site source=site({0,0,0, coherent_source_table[icoherentsource]});
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

	    TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propUP_coherent),"ISOSPIN32");

	    TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propUP_coherent),"ISOSPIN32");

	    //write D
	    outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_D";
	  
	    TIME( corrD.D_diagrams( reductionsT1, reductionsT2 ),"ISOSPIN32");
	    TIME( corrD.apply_phase(),"ISOSPIN32" );
	    TIME( corrD.apply_sign("D") ,"ISOSPIN32");
	    TIME( corrD.applyBoundaryConditions( true ),"ISOSPIN32" );
	    TIME( corrD.writeHDF5(outfilename),"ISOSPIN32" );

#ifdef PLEGMA_SCATTERING_SPIN12
	    //For I=1/2 I_3=+1/2
	    outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_DNUPUP_T2";
	    TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN_coherent, propUP_coherent, propUP_coherent), "ISOSPIN12");
	    TIME( corrD.convertTreductiontoDiagram( reductionsT2, -1 ), "ISOSPIN12"); //The argument -1 indicates the corrD does not contain any mesonic 
	    //gamma structure
	    TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

	    outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_DNUPUP_T1";
	    TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN_coherent, propUP_coherent, propUP_coherent), "ISOSPIN12");
	    TIME( corrD.convertTreductiontoDiagram( reductionsT1, -1 ), "ISOSPIN12");
	    TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

	    outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPUPDN_T1";
	    TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propDN_coherent), "ISOSPIN12");
	    TIME( corrD.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
	    TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

	    outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPDNUP_T1";
	    TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP_coherent, propDN_coherent, propUP_coherent), "ISOSPIN12");
	    TIME( corrD.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
	    TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

	    outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_DELTA_UPDNUP_T2";
	    TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propDN_coherent), "ISOSPIN12");
	    TIME( corrD.convertTreductiontoDiagram( reductionsT2,-1 ), "ISOSPIN12");
	    TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );
#endif

	  }
	
	  //N diagram
	  if (do_contraction_std==true){
	  std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
	  momList list_mpf1(1,{mpf1,},{0,});
	  PLEGMA_ScattCorrelator<float> corrNP_coherent(src, list_mpf1 );
#ifdef PLEGMA_SCATTERING_SPIN12
	  PLEGMA_ScattCorrelator<float> corrN0_coherent(src, list_mpf1 );
#endif

	  //initialize diagram
	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext_inside+"_N";

	  corrNP_coherent.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP");
#ifdef PLEGMA_SCATTERING_SPIN12
	  corrN0_coherent.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0");
#endif

	
	  //Computing T reductions+recombination
	  { 
	    PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
	    PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
	    //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
	    TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP_coherent, propDN_coherent, propUP_coherent), "ISOSPIN32");

	    PLEGMA_printf("Nucleon T2 reduction\n");
	    TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP_coherent, propDN_coherent, propUP_coherent), "ISOSPIN32");
	    PLEGMA_printf("Nucleon T2 reduction ready\n");

	    TIME(corrNP_coherent.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN32");
	    PLEGMA_printf("Nucleon diagram ready\n");

	    for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
	      corrNP.absorbTimeslice(corrNP_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
	    }

	    TIME( corrNP_coherent.apply_phase(),"ISOSPIN32" );
	    TIME( corrNP_coherent.apply_sign("N"),"ISOSPIN32" );
	    TIME( corrNP_coherent.applyBoundaryConditions( true ),"ISOSPIN32" );
	    TIME( corrNP_coherent.writeHDF5(outfilename),"ISOSPIN32" );


#ifdef PLEGMA_SCATTERING_SPIN12 
	    //Secondly compute N_0 (neutron)(we need for M diagram (N0p+))
	    TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN_coherent, propUP_coherent, propDN_coherent), "ISOSPIN12");
	  
	    TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN_coherent, propUP_coherent, propDN_coherent), "ISOSPIN12");

	    TIME(corrN0_coherent.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN12");

	    for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
	      corrN0.absorbTimeslice(corrN0_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
	    }

#endif

	    TIME( corrN0_coherent.apply_phase(),"ISOSPIN12" );
	    TIME( corrN0_coherent.apply_sign("N"),"ISOSPIN12" );
	    TIME( corrN0_coherent.applyBoundaryConditions( true ),"ISOSPIN12" );
	    TIME( corrN0_coherent.writeHDF5(outfilename) ,"ISOSPIN12");

	  }//end of T reduction 
	  }//end of if(do_contraction_std)

	} //End of loop on coherent sources


	asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
        std::string sourcepositiontext= (std::string)"_" + ssource;
        free(ssource);

        //P diagram
        std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
        momList list_mpi2(1,{mpi2,},{0,});
        PLEGMA_ScattCorrelator<float> corrP(sourcePositions[isource], list_mpi2);
        corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");


        std::vector<GAMMAS_SCATT> gamma_5_t_sourcemeson=apply_gamma5_scatt_gamma(glist_source_meson,LEFT);


	for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){

	  auto &momentum_f2 =  mpf2[i_mpf2];
	  //List of momenta corresponding to a fix value of p_i2
	  momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_f2, 2);

	  PLEGMA_Vector<float> spropagator_V3;
	  PLEGMA_Vector<float> spropagator_V2;

	  std::string pf2x=std::to_string(momentum_f2[0]);
	  std::string pf2y=std::to_string(momentum_f2[1]);
	  std::string pf2z=std::to_string(momentum_f2[2]);

	    
	  spropagator_V2.copy(*stochastic_oet_prop_d_fini_mom[i_mpf2], HOST);
	  spropagator_V2.load();


	  PLEGMA_ScattCorrelator<float> corrB1(sourcePositions[isource], filtered_sourcemomentumList);
	  PLEGMA_ScattCorrelator<float> corrB2(sourcePositions[isource], filtered_sourcemomentumList);


	  
	  //initialize diagrams
	  corrB1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B1", true);
	  corrB2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B2", true);

	  site source=site({0,0,0,sourcePositions[isource][DIM_T]});

	  PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList.uniq_p(1));//V2 reduction for momentum pf1
	  PLEGMA_ScattCorrelator<float> reductionsV3(source, filtered_sourcemomentumList.uniq_p(0));//V3 reduction for momentum pi2


	  reductionsV2.V2(spropagator_V2,glist_sink_nucleon, propUP, propUP, false);
	  reductionsV2.writeHDF5("V2redforBdiagram"+std::to_string(i_mpf2));

          //Loop over the different gamma structure for the source meson
          for (int i_gamma_f2=0; i_gamma_f2<glist_sink_meson.size(); ++i_gamma_f2) {


	    GAMMAS_SCATT gamma_f2 = glist_sink_meson[i_gamma_f2];
            GAMMAS_SCATT gamma_f2_t_gamma5= apply_g5( gamma_f2, RIGHT );

            PLEGMA_Propagator<float> propDN_source_to_source;
	  
	    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
  
              PLEGMA_Vector<float> spropagator;
              PLEGMA_Vector3D<float> vector1;
 	      spropagator.copy(*stochastic_oet_prop_d_zero_mom[timeidx],HOST);	
	      spropagator.load();
          
	      vector1.absorb(spropagator,  sourcePositions[isource][3]);
              spropagator_V3.absorb(vector1,  timeidx, false);

	    }

	    for (int isc=0; isc<12; ++isc){

              PLEGMA_Vector<float> spropagator;
              PLEGMA_Vector<float> stmp;
              PLEGMA_Vector3D<float> vector1;

              spropagator.absorb(propDN, isc/3, isc%3);
	      vector1.absorb(spropagator, sourcePositions[isource][3]);

	      for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
                stmp.absorb(vector1, timeidx, false); 
	      }

	        
	      propDN_source_to_source.absorb(stmp, isc/3, isc%3);

	    }

            spropagator_V3.apply_gamma_scatt(gamma_f2_t_gamma5,RIGHT);


	    reductionsV3.V3(spropagator_V3, gamma_5_t_sourcemeson, propDN_source_to_source, true);

	    reductionsV3.writeHDF5("V3redforBdiagram");

	    TIME(corrB1.B_diagrams(reductionsV3, reductionsV2, i_gamma_f2, 1, true),"ISOSPIN32");


	    
	    TIME(corrB2.B_diagrams(reductionsV3, reductionsV2, i_gamma_f2, 2, true),"ISOSPIN32");

	     

	  }//gamma_f2
	         
	  //## B

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
   
          TIME(produceOutput(corrB1, outfilename, "4pt", 1, n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrB2, outfilename, "4pt", 1, n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");


	}//i_mpf2



	{ //W,Z diagrams

	//We draw a different random vector for every source position
	vectorSource_oet.stochastic_Z(nroots);

        PLEGMA_Vector<float> spropagator_V6;



	//Doing for +mu for the UP propagator spin dilution oet
	if(mu<0) {
	   mu*=-1.;
	   solver.UpdateSolver();
	}

	PLEGMA_Vector<float> stochastic_oet_prop_u_zero_mom;
	PLEGMA_Vector<float> stochastic_oet_prop_u_fini_mom;


	// Smearing the source and invert
	{
	
	PLEGMA_Vector<double> vectortmp1;
	PLEGMA_Vector<double> vectortmp2;

	for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){
	  vectortmp1.absorbTimeslice(vectorSource_oet, coherent_source_table[i_coherent_source]);
	  PLEGMA_Vector3D<double> vector1, vector2;
	  vector1.absorb(vectortmp1, coherent_source_table[i_coherent_source]);

	  PLEGMA_Gauge3D<double> smearedGauge3D;
	  smearedGauge3D.absorb(smearedGauge, coherent_source_table[i_coherent_source]);

	  TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
	  vectortmp1.absorb(vector2,coherent_source_table[i_coherent_source]);
	  vectortmp2.absorbTimeslice(vectortmp1,coherent_source_table[i_coherent_source],false);
	}


	//Save the smeared,transformed and diluted source for non-zero momentum oet.
	vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
	vectorSource_oet.copy(vectortmp1);


	TIME(solver.solve(vectortmp2, vectortmp2), "ISOSPIN32");

	vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);

	TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss), "ISOSPIN32");
	
	stochastic_oet_prop_u_zero_mom.copy(vectortmp2);

        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(stochastic_oet_prop_u_zero_mom);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_oet_mpi2_zero_momstoch_mom_"+confnumber);
        }


	}

	PLEGMA_ScattCorrelator<float> reductionsV3(source, sourcemomentumList.uniq_p(2));
	//for the Z diagram we need V3 reduction for momentum pf2
	PLEGMA_ScattCorrelator<float> reductionsV2(source, sourcemomentumList.uniq_p(1));
	//for the Z diagram we need V2 reduction for momentum pf1
	PLEGMA_ScattCorrelator<float> reductionsV4(source, sourcemomentumList.uniq_p(1));
	//for the Z diagram we need V4 reduction for momentum pf1
        PLEGMA_ScattCorrelator<float> reductionsV6(source, sourcemomentumList.uniq_p(1));

	PLEGMA_ScattCorrelator<float> reductionsV2T(source,  sourcemomentumList.uniq_p(3));
	//for the T diagram we need V2 reduction for momentum p total
	
	PLEGMA_ScattCorrelator<float> reductionsV4T(source,  sourcemomentumList.uniq_p(3));
	//for the T diagram we need V4 reduction for momentum p total

	//Factors for the T diagram (zero momentum i2)
        TIME(reductionsV4T.V4( stochastic_oet_prop_u_zero_mom, glist_sink_delta, propUP, propUP),"ISOSPIN32");

	reductionsV4T.writeHDF5("V4redforTdiagram");

        TIME(reductionsV2T.V2( stochastic_oet_prop_u_zero_mom, glist_sink_delta, propUP, propUP),"ISOSPIN32");

	reductionsV2T.writeHDF5("V2redforTdiagram");




	//Factors for the Z diagram
	TIME(reductionsV4.V4( stochastic_oet_prop_u_zero_mom, glist_sink_nucleon, propDN, propUP),"ISOSPIN32");
 
	reductionsV4.writeHDF5("V4redforZdiagram");

        TIME(reductionsV2.V2( stochastic_oet_prop_u_zero_mom, glist_sink_nucleon, propDN, propUP),"ISOSPIN32");

        reductionsV2.writeHDF5("V2redforZdiagram");



	//Loop over the source meson momentum
        for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);

          PLEGMA_ScattCorrelator<float> corrZ1(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ2(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ3(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ4(sourcePositions[isource], filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW1(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW2(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW3(sourcePositions[isource], filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW4(sourcePositions[isource], filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList);
	  
	  //T diagrams
          std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
          std::vector<std::vector<int>> mpi2_filt;
          mpi2_filt.assign(mptot_filt.size(),momentum_i2);
          momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});


          PLEGMA_ScattCorrelator<float> corrT1(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT2(sourcePositions[isource], list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT3(sourcePositions[isource], list_mpi2ptot);

          corrT1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "T1oet");
          corrT2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "T2oet");
          corrT3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "T3oet");

          
	  //initialize diagrams
          corrW1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W1");
          corrW2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W2");
          corrW3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W3");
          corrW4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W4");


          corrZ1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z1");
          corrZ2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z2");
          corrZ3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z3");
          corrZ4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z4");
          corrM.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "MNPPP");



	  if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){

	    PLEGMA_Vector<double> vectortmp1;
	    PLEGMA_Vector<double> vectortmp2;
	    vectortmp1.copy(vectorSource_oet);
          
	    //Multiplying by the appropriate momentum phase
          
	    std::vector<int> tmp_4Dmom= momentum_i2 ;
	    tmp_4Dmom.push_back(0);
	    vectortmp1.mulMomentumPhases(tmp_4Dmom,-1);

	  
            //Doing the inversion
            TIME(solver.solve(vectortmp1, vectortmp1),"ISOSPIN32");

            //Rotate back immediately to the physical basis
            vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);

            //performing smearing
            TIME(vectortmp1.gaussianSmearing(vectortmp2, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

              
	    stochastic_oet_prop_u_fini_mom.copy(vectortmp1);

	    {
              PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(stochastic_oet_prop_u_fini_mom);
              vectorAuxF.unload();
              vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_oet_mpi2_momstoch_mom_"+std::to_string(i_mpi2)+"_"+confnumber);
            }




	  }

	  std::shared_ptr<float> Phi0;

          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){

	    stochastic_oet_prop_u_fini_mom.unload();

            Phi0 = stochastic_oet_prop_u_fini_mom.getPointSource(actualSource,HOST);

            stochastic_oet_prop_u_fini_mom.load();

	  }
	  else {

            stochastic_oet_prop_u_zero_mom.unload();

            Phi0 = stochastic_oet_prop_u_zero_mom.getPointSource(actualSource,HOST);

            stochastic_oet_prop_u_zero_mom.load();

	  }



	  TIME(corrT1.T_diagrams_oet(reductionsV4T, Phi0, i_mpi2, 1, true),"ISOSPIN32");

	  TIME(corrT2.T_diagrams_oet(reductionsV2T, Phi0, i_mpi2, 2, true),"ISOSPIN32");

          TIME(corrT3.T_diagrams_oet(reductionsV2T, Phi0, i_mpi2, 3, true),"ISOSPIN32");


	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
	  
	  TIME(corrT1.apply_phase(),"ISOSPIN32");
	  TIME(corrT1.apply_sign("T"),"ISOSPIN32");
	  TIME(corrT1.applyBoundaryConditions( true,  n_coherent_source, coherent_source_table_timeslice ),"ISOSPIN32");
	  TIME(corrT1.writeHDF5(outfilename),"ISOSPIN32");


	  TIME(corrT2.apply_phase(),"ISOSPIN32");
          TIME(corrT2.apply_sign("T"),"ISOSPIN32");
          TIME(corrT2.applyBoundaryConditions( true,  n_coherent_source, coherent_source_table_timeslice ),"ISOSPIN32");
          TIME(corrT2.writeHDF5(outfilename),"ISOSPIN32");


	  TIME(corrT3.apply_phase(),"ISOSPIN32");
          TIME(corrT3.apply_sign("T"),"ISOSPIN32");
          TIME(corrT3.applyBoundaryConditions( true,  n_coherent_source, coherent_source_table_timeslice ),"ISOSPIN32");
          TIME(corrT3.writeHDF5(outfilename),"ISOSPIN32");


	  //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       
	  if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
	    
	    TIME(corrP.P_diagrams( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom, i_mpi2, true),"ISOSPIN32");
	 
	    TIME(corrM.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom ),"ISOSPIN32");
	  
	  }
	  else{

	    TIME(corrP.P_diagrams( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom, i_mpi2, true),"ISOSPIN32");

	    TIME(corrM.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom ),"ISOSPIN32");
	  
	  }
        

	  PLEGMA_Vector<float> spropagator_zero;
	  PLEGMA_Vector<float> spropagator_fini;

          PLEGMA_Vector<float> vectortmp_zero;
          PLEGMA_Vector<float> vectortmp_fini;


	  for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

            PLEGMA_Vector<float> spropagator;
            PLEGMA_Vector3D<float> vector1;
            spropagator.copy(*stochastic_oet_prop_d_zero_mom[timeidx],HOST);
            spropagator.load();

            vector1.absorb(spropagator,  timeidx);
            spropagator_V6.absorb(vector1,  timeidx, false);

          }

          for  (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){


         /*   spropagator_fini.copy(*stochastic_oet_prop_d_fini_mom[i_mpf2],HOST);

            std::shared_ptr<float> Phi1 = spropagator_fini.getPointSource(actualSource,HOST);
	    for (int i=0;i<24;++i){
              printf("Phi0 %e\n",Phi0.get()[i]);
            }


	    for (int i=0;i<24;++i){
	      printf("Phi1 %e\n",Phi1.get()[i]);
	    }
            spropagator_fini.load();*/

            for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2) {

              for (int i_gamma_f2=0; i_gamma_f2 < glist_sink_meson.size(); ++i_gamma_f2) {


                reductionsV6.V6_RED(stochastic_oet_prop_u_zero_mom, spropagator_V6, glist_sink_nucleon,propUP,1,2,false);

                reductionsV6.writeHDF5("V6redforWdiagram12");


		TIME(corrW1.W_diagrams_oet(reductionsV6, Phi0, stochastic_oet_prop_d_zero_mom, i_mpi2,  i_mpf2, 1, true),"ISOSPIN32");

		TIME(corrW2.W_diagrams_oet(reductionsV6, Phi0, stochastic_oet_prop_d_zero_mom, i_mpi2,  i_mpf2, 2, true),"ISOSPIN32");

		reductionsV6.V6_RED(stochastic_oet_prop_u_zero_mom, spropagator_V6, glist_sink_nucleon,propUP,0,1,false);

		reductionsV6.writeHDF5("V6redforWdiagram34");

                  
		TIME(corrW3.W_diagrams_oet(reductionsV6, Phi0, stochastic_oet_prop_d_zero_mom, i_mpi2,  i_mpf2, 3, true),"ISOSPIN32");

		TIME(corrW4.W_diagrams_oet(reductionsV6, Phi0, stochastic_oet_prop_d_zero_mom, i_mpi2,  i_mpf2, 4, true),"ISOSPIN32");

		  
		

	      }

	    }

	  }

	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";


          TIME(produceOutput(corrW1, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrW2, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrW3, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrW4, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");


          for (int i_gamma_i2=0; i_gamma_i2 < glist_source_meson.size(); ++i_gamma_i2) {


            GAMMAS_SCATT gamma_i2 = glist_sink_meson[i_gamma_i2];
            GAMMAS_SCATT gamma_i2_t_gamma5= apply_g5( gamma_i2, RIGHT );

            std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
	    if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
              vectortmp_fini.copy(stochastic_oet_prop_u_fini_mom);
              vectortmp_fini.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);

              TIME(reductionsV3.V3( vectortmp_fini, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");
              reductionsV3.writeHDF5("V3redforZdiagram");


            }
            else{

              vectortmp_zero.copy(stochastic_oet_prop_u_zero_mom);
              vectortmp_zero.apply_gamma_scatt(gamma_i2_t_gamma5,RIGHT);

              TIME(reductionsV3.V3( vectortmp_zero, gamma_5_t_sinkmeson, propUP, true),"ISOSPIN32");
              reductionsV3.writeHDF5("V3redforZdiagram");


            }



            TIME(corrZ1.Z_diagrams_without_dilution( reductionsV3, reductionsV4, i_gamma_i2, 1 ),"ISOSPIN32");
            TIME(corrZ2.Z_diagrams_without_dilution( reductionsV3, reductionsV4, i_gamma_i2, 2 ),"ISOSPIN32");

            TIME(corrZ3.Z_diagrams_without_dilution( reductionsV3, reductionsV2, i_gamma_i2, 3 ),"ISOSPIN32");
            TIME(corrZ4.Z_diagrams_without_dilution( reductionsV3, reductionsV2, i_gamma_i2, 4 ),"ISOSPIN32");


	  }

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

          TIME(produceOutput(corrZ1, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrZ2, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrZ3, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");
          TIME(produceOutput(corrZ4, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");

    
	  //## M
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";
          TIME(produceOutput(corrM, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN32");

      }//loop over unique set of momenta for p_i2

      //write P
      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrP.apply_sign("P"),"ISOSPIN32");
      TIME(corrP.writeHDF5( outfilename ),"ISOSPIN32");
      


      }//Z,W,P,M diagrams


    }




    for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
      
      stochastic_oet_prop_d_zero_mom.pop_back();

    }
    
    for (int i_mpf2=0; i_mpf2<mpf2.size(); ++i_mpf2){

        auto &momentum_f2 = mpf2[i_mpf2];

        if ((momentum_f2[0] != 0) || (momentum_f2[1] != 0) || (momentum_f2[2] != 0)){

	  stochastic_oet_prop_d_fini_mom.pop_back();

        }

     }


  } 

  finalize();
  
  return 0;
}
