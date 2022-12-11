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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks","Projs", "threep-filename","confnumber", "nstochSamples"};

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_stochastic_samples){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions( true ));
  TIME(source.normalize_nstoch(n_stochastic_samples));
  TIME(source.writeHDF5( outputFilename ));

}

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name
                  ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions( true ));
  TIME(source.writeHDF5( outputFilename ));
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
  std::string srcInputFile = "./input.src";
  std::string outdiagramPrefix="";
  std::string outfilename;

  std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5};

  std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

  std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};//,S12,S13,S23,S41,S42,S43};

  int n_stochastic_samples;
  int max_source_sink_separations;


  auto add_options = [&](Options& options) {
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    options.set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    options.set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
    options.set("maxSourceSinkSeparations", "Maximal source sink separations", verbosity, max_source_sink_separations);

		     };
  //=========================================================================================================//
  initializePLEGMA();



  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<double> contractGauge(BOTH);
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

    std::vector<std::vector<int>> mpi2_twopt = sourcemomentumList_twopt.uniq_p(0);
    momList list_mpi2_twopt(1,{mpi2_twopt,},{0,});

    std::vector<std::vector<int>> mpi2_threept = sourcemomentumList_threept.uniq_p(0);
    momList list_mpi2_threept(1,{mpi2_threept,},{0,});

    std::vector<std::vector<int>> mpf1_threept = sourcemomentumList_threept.uniq_p(1);
    momList list_mpf1_threept(1,{mpf1_threept,},{0,});

    std::vector<std::vector<int>> mpf1_twopt = sourcemomentumList_twopt.uniq_p(1);
    momList list_mpf1_twopt(1,{mpf1_twopt,},{0,});

    std::vector<std::vector<int>> mpc = sourcemomentumList_threept.uniq_p(3);
    momList list_mpc(1,{mpc,},{0,});


    int parallel_sources=HGC_totalL[3]/max_source_sink_separations;
    std::vector<int> lookuptable;
    
    for (int i=0; i<HGC_totalL[3]; ++i)
      lookuptable.push_back(-1);
//    for (
//    for (int j=0; j< tSinks.size();++j){
//      for (int k=0; k< parallel_sources;++k){
//        lookuptable[(i*tSinks.size()*parallel_sources+j*parallel_sources+k);
//      }
//    }

    PLEGMA_Vector<float> vectorSource_stochastic;
    vectorSource_stochastic.randInit(rand_seed1);

    //Computing with ubaru insertion
    //Step (1) produce the stochastic sample
    //Step (2) produce the stochastic propagators
    //Step (3) produce standard point to all propagators
    //Step (4) produce factors with point to all
    //Step (5) produce sequential through the source with momenta pi2
    //Step (6) produce factors with the sequential
    //Step (7) doing the recombination
    //Step (8) doing the one end trick calculation for the Z diagrams

    std::vector<PLEGMA_Vector<float>*> stochastic_sources;

    std::vector<PLEGMA_Vector<float>*> stochastic_propags_UP;

    std::vector<PLEGMA_Vector<float>*> stochastic_propags_DN;


    /******************************************************
     *
     *Step 1: Producing the stochastic sources
     *
     ******************************************************/

    for (int i=0; i<n_stochastic_samples;++i){
   
      stochastic_sources.push_back(new PLEGMA_Vector<float>(HOST));
      vectorSource_stochastic.stochastic_Z(nroots);
      vectorSource_stochastic.unload();
      stochastic_sources[i]->copy(vectorSource_stochastic,HOST);
      vectorSource_stochastic.load();

    }

    /******************************************************
     *
     *Step 2: Producing the stochastic propagators
     *
     ******************************************************/

    if(mu<0) 
    {
      mu = -mu;
      solver.UpdateSolver();
    }

    int countindex=0;
    for (int isource=0; isource<numSourcePositions;++isource){
      site& source = sourcePositions[isource];
      for (int k=0; k<tSinks.size();++k){
        for (int l=0; l<parallel_sources;++l){
          if (lookuptable[(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]==-1){
            int timeSlice=(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];	     
	    PLEGMA_Gauge3D<double> smearedGauge3D;
            smearedGauge3D.absorb(smearedGauge, timeSlice );
            for(int i=0; i< n_stochastic_samples; ++i) {
	      vectorSource_stochastic.copy( *stochastic_sources[i], HOST);
              vectorSource_stochastic.load();
	      PLEGMA_Vector<double> vectorInOut, vectorAuxD1;
              { // Smearing the source
		PLEGMA_Vector3D<double> vector1, vector2;
		vectorInOut.copy(vectorSource_stochastic);
                vector1.absorb(vectorInOut,timeSlice);
                TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
                vectorInOut.absorb(vector2,timeSlice);
              }

              //Step(4) We rotate the source to the physical basis
              TIME(vectorAuxD1.rotateToPhysicalBasis(vectorInOut,+1));
 
	      //Step(5) pick out a particular timeslice from the source
              vectorInOut.absorbTimeslice(vectorAuxD1, timeSlice);
              
	      //Step(6) Solve
              TIME(solver.solve(vectorInOut, vectorInOut));
              
              //Step(6) We rotate back the propagator to the physical basis
              TIME(vectorAuxD1.rotateToPhysicalBasis(vectorInOut,+1));

              //Step(7) Smearing all the time slice in the propagator
              TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

	      vectorInOut.unload();
	      stochastic_propags_UP.push_back(new PLEGMA_Vector<float>(HOST));
              stochastic_propags_UP[countindex]->copy(vectorInOut, HOST);
	      countindex++;
              //(isource*tSinks.size()*parallel_sources+k*parallel_sources+l)*n_stochastic_samples+i]->copy(vectorInOut, HOST);
	      vectorInOut.load();

	    }
	    lookuptable[timeSlice]=countindex/n_stochastic_samples;
	  }
        }
      }
    }

    if(mu>0)
    {
      mu = -mu;
      solver.UpdateSolver();
    }


    countindex=0;
    for (int isource=0; isource<numSourcePositions;++isource){
      site& source = sourcePositions[isource];
      for (int k=0; k<tSinks.size();++k){
        for (int l=0; l<parallel_sources;++l){
          if (lookuptable[(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]==-1){
            int timeSlice=(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            PLEGMA_Gauge3D<double> smearedGauge3D;
            smearedGauge3D.absorb(smearedGauge, timeSlice );
            for(int i=0; i< n_stochastic_samples; ++i) {
              vectorSource_stochastic.copy( *stochastic_sources[i], HOST);
              vectorSource_stochastic.load();
              PLEGMA_Vector<double> vectorInOut,vectorAuxD1;
              { // Smearing the source
                PLEGMA_Vector3D<double> vector1, vector2;
		vectorInOut.copy(vectorSource_stochastic);
                vector1.absorb(vectorInOut,timeSlice);
                TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
                vectorInOut.absorb(vector2,timeSlice);
              }

              //Step(4) We rotate the source to the physical basis
              TIME(vectorAuxD1.rotateToPhysicalBasis(vectorInOut,-1));

              //Step(5) pick out a particular timeslice from the source
              vectorInOut.absorbTimeslice(vectorAuxD1, timeSlice);

              //Step(6) Solve
              TIME(solver.solve(vectorInOut, vectorInOut));

              //Step(6) We rotate back the propagator to the physical basis
              TIME(vectorAuxD1.rotateToPhysicalBasis(vectorInOut,-1));

              //Step(7) Smearing all the time slice in the propagator
              TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

              vectorInOut.unload();
              stochastic_propags_DN.push_back(new PLEGMA_Vector<float>(HOST));
	      stochastic_propags_DN[countindex]->copy(vectorInOut, HOST);
              countindex++;

              //stochastic_propags_DN[(isource*tSinks.size()*parallel_sources+k*parallel_sources+l)*n_stochastic_samples+i]->copy(vectorInOut, HOST);
              vectorInOut.load();

            }
            //lookuptable[(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]=isource*tSinks.size()*parallel_sources+k*parallel_sources+l;
            lookuptable[timeSlice]=countindex/n_stochastic_samples;
          }
        }
      }
    }


    /******************************************************
     *
     * Step 3: Computing the point to all propagators
     *
     ******************************************************/


    for(int isource = startSource; isource < numSourcePositions; isource++){

      site& source = sourcePositions[isource];

      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",isource, source[0], source[1], source[2], source[3]);
      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);


      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear,  site &source_location, bool finalize) {
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
				     vector1.pointSource(source_location, isc/3, isc%3, DEVICE);
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
				     vectorInOut.absorb(vector2,source[DIM_T]);
				   }
				   // Inverting
				   PLEGMA_printf("Going to invert %s for component %d\n",
						 fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
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

      PLEGMA_Propagator<float> propUP_SS_packed;
      PLEGMA_Propagator<float> propDN_SS_packed;

      PLEGMA_Propagator<float> propUP_SL_packed;
      PLEGMA_Propagator<float> propDN_SL_packed;

      


      for (int i=0; i<HGC_totalL[3]/max_source_sink_separations;++i){
	PLEGMA_Propagator<float> propUP_SS;
        PLEGMA_Propagator<float> propDN_SS;

	PLEGMA_Propagator<float> propUP_SL;
        PLEGMA_Propagator<float> propDN_SL;

        site& source_local = sourcePositions[isource];
	source_local[3]=(sourcePositions[isource][3]+i*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

	site source_local_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});
        source_local_reduction[3]=(sourcePositions[isource][3]+i*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];


  	PLEGMA_ScattCorrelator<float> corrNP(source_local, list_mpf1_twopt);
    	TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"));

        PLEGMA_ScattCorrelator<float> corrN0(source_local, list_mpf1_twopt);
        TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"));

        bool computed_light = false;
        // If twop_filename exists we hold the computation of the light props
        if(access( twop_filename.c_str(), F_OK ) == -1) {
	  TIME(computePropagator(propUP_SS, propUP_SL, mu_ud, LIGHT, nsmearGauss, source_local, false));
	  TIME(computePropagator(propDN_SS, propDN_SL, -mu_ud, LIGHT, nsmearGauss, source_local, false));
          computed_light = true; 
        }

        //Computing T reductions+recombination
        {
          PLEGMA_ScattCorrelator<float> reductionsT1N(source_local_reduction, sourcemomentumList_twopt.uniq_p(1));
          PLEGMA_ScattCorrelator<float> reductionsT2N(source_local_reduction, sourcemomentumList_twopt.uniq_p(1));
          //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP_SS, propDN_SS, propUP_SS));
          //PLEGMA_printf("Nucleon T2 reduction\n");
          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP_SS, propDN_SS, propUP_SS));
          //PLEGMA_printf("Nucleon T2 reduction ready\n");
          TIME(corrNP.N_diagrams( reductionsT1N, reductionsT2N ));
          //PLEGMA_printf("Nucleon diagram ready\n");

        }

	//Computing T reductions+recombination
        {
          PLEGMA_ScattCorrelator<float> reductionsT1N(source_local_reduction, sourcemomentumList_twopt.uniq_p(1));
          PLEGMA_ScattCorrelator<float> reductionsT2N(source_local_reduction, sourcemomentumList_twopt.uniq_p(1));
          //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN_SS, propUP_SS, propDN_SS));
          //PLEGMA_printf("Nucleon T2 reduction\n");
          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN_SS, propUP_SS, propDN_SS));
          //PLEGMA_printf("Nucleon T2 reduction ready\n");
          TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ));
	  //PLEGMA_printf("Nucleon diagram ready\n");
        }

        propUP_SS_packed.pack_propagator_from_source_to_sink(propUP_SS, source_local[3], max_source_sink_separations, i == 0 ? true : false);
        propDN_SS_packed.pack_propagator_from_source_to_sink(propDN_SS, source_local[3], max_source_sink_separations, i == 0 ? true : false);

        propUP_SL_packed.pack_propagator_from_source_to_sink(propUP_SS, source_local[3], max_source_sink_separations, i == 0 ? true : false);
        propDN_SL_packed.pack_propagator_from_source_to_sink(propDN_SS, source_local[3], max_source_sink_separations, i == 0 ? true : false);


      }

      //We implement the UD part first
      //The neutron piplus at the source
	

      int sequential_time_source=source[DIM_T];

      //We can compute V2 contractions for B and V3 contraction for W first
      //without having to compute it for all the iterations in the loop
      //over the sequential momentum
      //Provided we have the same pf1,pf2 pairs for all pi2 sequential momentum
      //For the saved V2 and V3 reductions we have to use all possible unique pf1 and pf2

      //Here the prefix UU means that reduction is based phi and xi, without the gamma_5
      //We replace U(x_f2,x_f1) with phi(x_f2) xi^dagger(x_f1)
      //phi goes to V2 reduction and xi goes to V3 reduction

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U;//implemented

      //Here the prefix DD means that reduction is based phi*g5 and xi*g5
      //We replace D(x_f2,x_f1) with xi(x_f2)*gamma_5* phi^dagger(x_f1) *gamma_5
      //phi goes to V3 reduction and xi goes to V2 reduction

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U;//implemented

      for(int i=0; i< n_stochastic_samples; ++i) {
	for (int k=0; k< tSinks.size(); ++k){
          try
          {
            reductions_DD_V2_GAMMAF1U_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
            reductions_UU_V2_GAMMAF1D_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
            reductions_UU_V4_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));

            reductions_UU_V3_GAMMAF2U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));
            reductions_DD_V3_GAMMAF2D.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));

          }
          catch(std::bad_alloc&){
            PLEGMA_printf("Memory allocation fails to store factors");
            exit(1);
          }
        }
      }

      for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){

        PLEGMA_Vector<float> stochastic_source;
        stochastic_source.copy(*stochastic_sources[i_sample],HOST);
        stochastic_source.load();


	for (int k=0; k< tSinks.size();++k){

          stochastic_source.apply_gamma5();

          //For U(xf1,xf2)
          //B3,B5,B9,B11
	  PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;
          {
            for (int j=0; j<HGC_totalL[3]/max_source_sink_separations;++j){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
              propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            }
          }

          TIME(reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k]->V2( stochastic_source,     glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, false));


          //B4,B6
          //B10,B12
          TIME(reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k]->V4( stochastic_source,     glist_sink_nucleon, propUPpacked_to_sink, propDNpacked_to_sink, false));

          //W5,W6,W7,W8
          //W9,W10,W11,W12,W17,W18,W19,W20
	  PLEGMA_Vector<float> stochastic_propagator_packed;
          for (int j=0; j<HGC_totalL[3]/max_source_sink_separations;++j){
            PLEGMA_Vector<float> temporary;
            temporary.copy(*stochastic_propags_DN[lookuptable[(source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	    temporary.load();
            stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, j == 0 ? true : false);
          }

          stochastic_propagator_packed.apply_gamma5();

	  
          TIME(reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k]->V3( stochastic_propagator_packed, glist_insertion,   propUP_SL_packed, true));

          //For D(xf1,xf2)

          stochastic_propagator_packed.apply_gamma5();


	  for (int j=0; j<HGC_totalL[3]/max_source_sink_separations;++j){
            PLEGMA_Vector<float> temporary;
            temporary.copy(*stochastic_propags_UP[lookuptable[(source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
            temporary.load();
            stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary,  (source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, j == 0 ? true : false);
          }

          stochastic_propagator_packed.apply_gamma5();


          //W13,W14,W15,W16
          //W21,W22,W23,W24
          TIME(reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k]->V3( stochastic_propagator_packed, glist_insertion,   propDN_SL_packed, true));

          //B7,B8
          //B1,B2
          TIME(reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k]->V2( stochastic_source, glist_sink_nucleon, propUPpacked_to_sink, propUPpacked_to_sink, false));

        } //end of for source sink separations
      } //end of for stochastic samples

      //We first have a loop over all unique the source meson momentum p_i2
      for (int i_mpi2=0; i_mpi2<mpi2_threept.size(); ++i_mpi2){

        auto &momentum_i2 =  mpi2_threept[i_mpi2];
        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList_threept.extract(momentum_i2, 0);

        PLEGMA_ScattCorrelator<float> reductionsV2(source_reduction, list_mpf1_threept);
		
	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(4);
        std::vector<std::vector<int>> mpi2_filt  ;
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);
        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

	PLEGMA_ScattCorrelator<float> reductionsT1(source_reduction, mpf1_threept);
        PLEGMA_ScattCorrelator<float> reductionsT2(source_reduction, mpf1_threept);

        PLEGMA_ScattCorrelator<float> reductionsV3(source_reduction, list_mpc);

	std::vector<PLEGMA_ScattCorrelator<float>*> corrB3,corrB4,corrB5,corrB6;
        std::vector<PLEGMA_ScattCorrelator<float>*> corrW5,corrW6,corrW7,corrW8;
        std::vector<PLEGMA_ScattCorrelator<float>*> corrW13,corrW14,corrW15,corrW16;


        for (int k=0; k< tSinks.size(); ++k){
          try
          {
            corrB3.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB4.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB5.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB6.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

	    corrW5.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW6.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW7.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW8.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

	    corrW13.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW14.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW15.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW16.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

          }
          catch(std::bad_alloc&){
            PLEGMA_printf("Memory allocation fails to store factors");
            exit(1);
          }
       
	  ssource=(char *)malloc(sizeof(char)*100);
	  sprintf(ssource,"B3_deltat_%d",tSinks[k]); 
	  corrB3[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B4_deltat_%d",tSinks[k]);
          corrB4[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B5_deltat_%d",tSinks[k]);
          corrB5[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B6_deltat_%d",tSinks[k]);
  	  corrB6[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

	  sprintf(ssource,"W5_deltat_%d",tSinks[k]);
          corrW5[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W6_deltat_%d",tSinks[k]);
          corrW6[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W7_deltat_%d",tSinks[k]);
          corrW7[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W8_deltat_%d",tSinks[k]);
          corrW8[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

	  sprintf(ssource,"W13_deltat_%d",tSinks[k]);
          corrW13[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W14_deltat_%d",tSinks[k]);
          corrW14[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W15_deltat_%d",tSinks[k]);
          corrW15[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W16_deltat_%d",tSinks[k]);
          corrW16[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
    
	  free(ssource);
        }
		 
	PLEGMA_Propagator<float> propTS_SS_packed;

	for (int i_source_parallel=0; i_source_parallel < HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){

	  PLEGMA_Propagator<float> propTS;

          //we first implemenet UD
	  //Ensure mu is positive
	  if(mu<0) {
            mu*=-1.;
            solver.UpdateSolver();
          }

	  //pi plus at the source 
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            //Performing the smearing
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propUP_SS_packed, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vector2.mulMomentumPhases(momentum_i2,1);
              vectorAuxD.absorb(vector2, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
	    }
            
	    vectorAuxD2.absorbTimeslice(vectorAuxD, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], false);

            //Perform multiplication with glist_insertion[0]
            vectorAuxD2.apply_gamma_scatt(glist_insertion[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_insertion[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1);
            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS.absorb(vectorAuxF, isc/3, isc%3);

          }

	  propTS_SS_packed.pack_propagator_from_source_to_sink(propTS, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
	}



	PLEGMA_ScattCorrelator<float> corrTproton_protonpizero1(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero2(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero3(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero4(source, list_mpi2ptot);

        corrTproton_protonpizero1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq21");

        corrTproton_protonpizero2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq22");

        corrTproton_protonpizero3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq23");

        corrTproton_protonpizero4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq24");

        TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed, propUP_SS_packed, propDN_SS_packed));
        TIME(corrTproton_protonpizero1.convertTreductiontoDiagram( reductionsT1, false, true, false ));

        TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed, propDN_SS_packed, propUP_SS_packed));
        TIME(corrTproton_protonpizero2.convertTreductiontoDiagram( reductionsT2, false, false, true ));

        TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
        TIME(corrTproton_protonpizero3.convertTreductiontoDiagram( reductionsT1, false, false, false ));

        TIME(reductionsT1.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
        TIME(corrTproton_protonpizero4.convertTreductiontoDiagram( reductionsT1, false, false, true));

	//creating factors

	PLEGMA_Vector<float> stochastic_propagator_packed;
	for (int k=0; k<tSinks.size(); ++k){

	  PLEGMA_Propagator<float> propTS_SS_packed_to_sink;
          {
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propTS_SS_packed_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, i_source_parallel==0 ? true : false);
            }
          }

	  PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;
          {
            for (int j=0; j<HGC_totalL[3]/max_source_sink_separations;++j){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
              propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            }
          }

          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
	      PLEGMA_Vector<float> temporary;
	      temporary.copy(*stochastic_propags_DN[lookuptable[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);

	      stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
	    }

            stochastic_propagator_packed.apply_gamma5();
	    
	    //V3
            TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion,   propTS_SS_packed, true));
	    TIME(corrB3[k]->B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0, 3, true));
            TIME(corrB4[k]->B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 4, true));
	    TIME(corrB5[k]->B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0, 5, true));
            TIME(corrB6[k]->B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 6, true));

	    stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
	    stochastic_propagator_packed.load();
	    stochastic_propagator_packed.apply_gamma5();



            TIME(reductionsV2.V4( stochastic_propagator_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));//checked

	    TIME(corrW5[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 5, true));

	    TIME(corrW7[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 7, true));

            TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));//checked

            TIME(corrW6[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()], reductionsV2, 0, 6, true));
            TIME(corrW8[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()], reductionsV2, 0, 8, true));

            TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));//checked

            TIME(corrW13[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 13, true));
            TIME(corrW15[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 15, true));

            TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propUPpacked_to_sink, propTS_SS_packed_to_sink, false));//checked
            TIME(corrW14[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 14, true));
            TIME(corrW16[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 16, true));

	  }//loop over stochastic samples

        }//loop over source sink separations

	for (int k=0; k<tSinks.size();++k){

          corrW5.pop_back();
          corrW6.pop_back();
          corrW7.pop_back();
          corrW8.pop_back();

          corrB3.pop_back();
          corrB4.pop_back();
          corrB5.pop_back();
          corrB6.pop_back();

          corrW13.pop_back();
          corrW14.pop_back();
          corrW15.pop_back();
          corrW16.pop_back();

        }

        // ensuring mu positive
        if(mu<0) {
          mu*=-1.;
          solver.UpdateSolver();
        }

	for (int i_source_parallel=0; i_source_parallel < HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){

          PLEGMA_Propagator<float> propTS;

          //we first implemenet UU pizero UP
          if(mu<0) {
            mu*=-1.;
            solver.UpdateSolver();
          }

          //pi plus at the source
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            //Performing the smearing
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN_SS_packed, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vector2.mulMomentumPhases(momentum_i2,1);
              vectorAuxD.absorb(vector2, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
            }

            vectorAuxD2.absorbTimeslice(vectorAuxD, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], false);

            //Perform multiplication with glist_insertion[0]
            vectorAuxD2.apply_gamma_scatt(glist_insertion[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_insertion[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1);
            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS.absorb(vectorAuxF, isc/3, isc%3);

          }

          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        }


	PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus1(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus2(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus3(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus4(source, list_mpi2ptot);

	corrTproton_neutronpiplus1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq11");

        corrTproton_neutronpiplus2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq12");

        corrTproton_neutronpiplus3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq13");

        corrTproton_neutronpiplus4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq14");

	TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed, propUP_SS_packed, propDN_SS_packed));
        TIME(corrTproton_neutronpiplus1.convertTreductiontoDiagram( reductionsT1, false, true, false ));

        TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed, propDN_SS_packed, propUP_SS_packed));
        TIME(corrTproton_neutronpiplus2.convertTreductiontoDiagram( reductionsT2, false, false, true ));

	TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propDN_SS_packed));
        TIME(corrTproton_neutronpiplus3.convertTreductiontoDiagram( reductionsT1, false, false, false ));

        TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
        TIME(corrTproton_neutronpiplus4.convertTreductiontoDiagram( reductionsT1, false, false, true));

	std::vector<PLEGMA_ScattCorrelator<float>*> corrB9,corrB10,corrB11,corrB12;
        std::vector<PLEGMA_ScattCorrelator<float>*> corrW17,corrW18,corrW19,corrW20;
        std::vector<PLEGMA_ScattCorrelator<float>*> corrW21,corrW22,corrW23,corrW24;


        for (int k=0; k< tSinks.size(); ++k){
          try
          {
            corrB9.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB10.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB11.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB12.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

            corrW17.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW18.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW19.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW20.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

            corrW21.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW22.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW23.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW24.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

          }
          catch(std::bad_alloc&){
            PLEGMA_printf("Memory allocation fails to store factors");
            exit(1);
          }

	  ssource=(char *)malloc(sizeof(char)*100);
          sprintf(ssource,"B9_deltat_%d",tSinks[k]);
          corrB3[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B10_deltat_%d",tSinks[k]);
          corrB4[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B11_deltat_%d",tSinks[k]);
          corrB5[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B12_deltat_%d",tSinks[k]);
          corrB6[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W17_deltat_%d",tSinks[k]);
          corrW5[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W18_deltat_%d",tSinks[k]);
          corrW6[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W19_deltat_%d",tSinks[k]);
          corrW7[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W20_deltat_%d",tSinks[k]);
          corrW8[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W21_deltat_%d",tSinks[k]);
          corrW13[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W22_deltat_%d",tSinks[k]);
          corrW14[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W23_deltat_%d",tSinks[k]);
          corrW15[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W24_deltat_%d",tSinks[k]);
          corrW16[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          free(ssource);

        }



	for (int k=0; k<tSinks.size();++k){
	  PLEGMA_Propagator<float> propTS_SS_packed_to_sink;
          {
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propTS_SS_packed_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, i_source_parallel==0 ? true : false);
            }
          }

	  PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;
          {
            for (int j=0; j<HGC_totalL[3]/max_source_sink_separations;++j){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
              propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            }
          }

          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            PLEGMA_Vector<float> stochastic_propagator_packed;
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              PLEGMA_Vector<float> temporary;
              temporary.copy(*stochastic_propags_DN[lookuptable[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);

              stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
            }

            stochastic_propagator_packed.apply_gamma5();

	    TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion,   propTS_SS_packed, true ));

            TIME( corrB9[k]->B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0,  9, true));
            TIME(corrB10[k]->B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 10, true));
            TIME(corrB11[k]->B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0, 11, true));
            TIME(corrB12[k]->B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 12, true));

            stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
            stochastic_propagator_packed.load();
            stochastic_propagator_packed.apply_gamma5();


            TIME(reductionsV2.V4( stochastic_propagator_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));
            TIME(corrW17[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 17, true));
            TIME(corrW19[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 19, true));

            TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));
            TIME(corrW18[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 18, true));
            TIME(corrW20[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 20, true));

            TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propUPpacked_to_sink, propTS_SS_packed_to_sink, false));
	    TIME(corrW22[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 22, true));
            TIME(corrW24[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 24, true));

            TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));

	    TIME(corrW21[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 21, true));
            TIME(corrW23[k]->W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 23, true));

	  } //loop over stochastic samples

	} //loop over source sink separations

	for (int k=0; k<tSinks.size();++k){

          corrW21.pop_back();
          corrW22.pop_back();
          corrW23.pop_back();
          corrW24.pop_back();

          corrB9.pop_back();
          corrB10.pop_back();
	  corrB11.pop_back();
          corrB12.pop_back();

	  corrW17.pop_back();
          corrW18.pop_back();
          corrW19.pop_back();
          corrW20.pop_back();

        }


        for (int i_source_parallel=0; i_source_parallel < HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){

          PLEGMA_Propagator<float> propTS;

          //we first implemenet DD pizero DN
          if(mu>0) {
            mu*=-1.;
            solver.UpdateSolver();
          }

          //pi plus at the source
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            //Performing the smearing
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN_SS_packed, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vector2.mulMomentumPhases(momentum_i2,1);
              vectorAuxD.absorb(vector2, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
            }

            vectorAuxD2.absorbTimeslice(vectorAuxD, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], false);

            //Perform multiplication with glist_insertion[0]
            vectorAuxD2.apply_gamma_scatt(glist_insertion[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,-1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_insertion[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,-1);
            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS.absorb(vectorAuxF, isc/3, isc%3);

          }

          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        }

        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero5(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero6(source, list_mpi2ptot);

        corrTproton_protonpizero5.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq25");

        corrTproton_protonpizero6.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq26");

        TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propUP_SS_packed));
        TIME(corrTproton_protonpizero5.convertTreductiontoDiagram( reductionsT1, false, true, false ));

        TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propDN_SS_packed));
        TIME(corrTproton_protonpizero6.convertTreductiontoDiagram( reductionsT2, false, false, true ));

	std::vector<PLEGMA_ScattCorrelator<float>*> corrB7,corrB8;
        std::vector<PLEGMA_ScattCorrelator<float>*> corrW9,corrW10,corrW11,corrW12;

	for (int k=0; k< tSinks.size(); ++k){
          try
          {
            corrB7.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrB8.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

            corrW9.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW10.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW11.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));
            corrW12.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList));

          }
          catch(std::bad_alloc&){
            PLEGMA_printf("Memory allocation fails to store factors");
            exit(1);
          }

	  ssource=(char *)malloc(sizeof(char)*100);
          sprintf(ssource,"B7_deltat_%d",tSinks[k]);
          corrB3[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B8_deltat_%d",tSinks[k]);
          corrB4[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W9_deltat_%d",tSinks[k]);
          corrW13[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W10_deltat_%d",tSinks[k]);
          corrW14[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W11_deltat_%d",tSinks[k]);
          corrW15[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W12_deltat_%d",tSinks[k]);
          corrW16[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          free(ssource);


          corrB7[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", "B9");
          corrB8[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", "B10");


          corrW9[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", "W17");
          corrW10[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", "W18");
          corrW11[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", "W19");
          corrW12[k]->initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", "W20");

        }

        for (int k=0; k<tSinks.size();++k){
          PLEGMA_Propagator<float> propTS_SS_packed_to_sink;
          {
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propTS_SS_packed_to_sink.pack_propagator_as_sink(propTS_SS_packed,  global_fixSinkTime, tsinkMtsource, i_source_parallel==0 ? true : false);
            }
          }

          PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;
          {
            for (int j=0; j<HGC_totalL[3]/max_source_sink_separations;++j){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
              propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            }
          }

          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            PLEGMA_Vector<float> stochastic_propagator_packed;
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              PLEGMA_Vector<float> temporary;
              temporary.copy(*stochastic_propags_UP[lookuptable[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);

              stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
            }

            stochastic_propagator_packed.apply_gamma5();

	    TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion,   propTS_SS_packed, true));

            TIME(corrB7[k]->B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k], 0, 7, true));
            TIME(corrB8[k]->B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k], 0, 8, true));

            stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
            stochastic_propagator_packed.load();
            stochastic_propagator_packed.apply_gamma5();

	    TIME(reductionsV2.V2( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));

            TIME( corrW9[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0,  9, true));
            TIME(corrW11[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 11, true));

            TIME(reductionsV2.V4( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));

            TIME(corrW10[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 10, true));
            TIME(corrW12[k]->W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 12, true));

	  } //loop over stochastic samples

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput(*corrB7[k], outfilename, "4pt", n_stochastic_samples));
          TIME(produceOutput(*corrB8[k], outfilename, "4pt", n_stochastic_samples));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput(*corrW9[k], outfilename,"4pt", n_stochastic_samples));//because of V4
          TIME(produceOutput(*corrW10[k], outfilename,"4pt", n_stochastic_samples));//because of V4
          TIME(produceOutput(*corrW11[k], outfilename,"4pt", n_stochastic_samples));
          TIME(produceOutput(*corrW12[k], outfilename,"4pt", n_stochastic_samples));


	} //loop over tSinks

	for (int k=0; k<tSinks.size();++k){

          corrW9.pop_back();
          corrW10.pop_back();
          corrW11.pop_back();
          corrW12.pop_back();

          corrB7.pop_back();
          corrB8.pop_back();

        }


      }//loop over mpi2

      for(int i_samples=0; i_samples< n_stochastic_samples; ++i_samples) {

	for (int k=0; k<tSinks.size();++k){

          reductions_UU_V2_GAMMAF1D_U.pop_back();
          reductions_UU_V4_GAMMAF1U_D.pop_back();
          reductions_UU_V3_GAMMAF2U.pop_back();

          reductions_DD_V2_GAMMAF1U_U.pop_back();
          reductions_DD_V3_GAMMAF2D.pop_back();

	}

      }

    }//loop over source position

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
    }
    for (int i=0; i<countindex;++i){
      stochastic_propags_UP.pop_back();
      stochastic_propags_DN.pop_back();
    }

  }//loop in finalize
  finalize();
  return 0;
}
