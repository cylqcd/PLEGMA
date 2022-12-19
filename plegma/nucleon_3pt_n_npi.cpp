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

void produceOutput_3pt( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_stochastic_samples,
		    int source_sink_separation,
		    int max_source_sink_separation){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions_3pt( true, source_sink_separation, max_source_sink_separation ));
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

void produceOutput_3pt( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
		    int source_sink_separation,
                    int max_source_sink_separation
                  ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions_3pt( true, source_sink_separation, max_source_sink_separation ));
  TIME(source.writeHDF5( outputFilename ));
}

  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::vector<double> mu_s;
  std::vector<double> mu_c;
  int rand_seed1=1234;
  int rand_seed2=5678;
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


  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("maxSourceSinkSeparations", "Maximal source sink separations", verbosity, max_source_sink_separations);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);

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
    std::vector<int> lookuptable_UP;
    std::vector<int> lookuptable_DN;

    
    for (int i=0; i<HGC_totalL[3]; ++i){
      lookuptable_UP.push_back(-1);
      lookuptable_DN.push_back(-1);
    }


    PLEGMA_Vector<double> vectorSource_stochastic;
    vectorSource_stochastic.randInit(rand_seed1);

    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);


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

    std::vector<PLEGMA_Vector<float>*> stochastic_propags_UP_SL;

    std::vector<PLEGMA_Vector<float>*> stochastic_propags_DN_SL;

    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_zero_mom_SS;

    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_zero_mom_SS;

    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_fini_mom_SL;

    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_fini_mom_SL;


    for(int i=0; i< 4; ++i) {

      stochastic_oet_prop_u_zero_mom_SS.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_d_zero_mom_SS.push_back(new PLEGMA_Vector<float>(HOST));

      stochastic_oet_prop_u_fini_mom_SL.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_d_fini_mom_SL.push_back(new PLEGMA_Vector<float>(HOST));


    }




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

      PLEGMA_printf("Save the stochastic source for sample %d\n",i);
      std::string nstoch=std::to_string(i);
      vectorSource_stochastic.writeHDF5("stochastic_source"+nstoch);

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
          if (lookuptable_UP[(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]==-1){
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
              //TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

              vectorAuxD1.unload();
	      PLEGMA_printf("Save the stochastic source for sample %d\n",i);
              std::string nstoch=std::to_string(i);
              vectorAuxD1.writeHDF5("stochastic_propagators_UP"+nstoch+"_t"+std::to_string(timeSlice));

	      //writeHDF5(outfilename) 
	      stochastic_propags_UP_SL.push_back(new PLEGMA_Vector<float>(HOST));
              stochastic_propags_UP_SL[countindex]->copy(vectorAuxD1, HOST);
	      countindex++;
              //(isource*tSinks.size()*parallel_sources+k*parallel_sources+l)*n_stochastic_samples+i]->copy(vectorInOut, HOST);
	      vectorAuxD1.load();

	    }
	    lookuptable_UP[timeSlice]=(countindex-1)/n_stochastic_samples;
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
          if (lookuptable_DN[(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]==-1){
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
              //TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

              vectorAuxD1.unload();
	      PLEGMA_printf("Save the stochastic source for sample %d\n",i);
              std::string nstoch=std::to_string(i);
              vectorAuxD1.writeHDF5("stochastic_propagators_DN"+nstoch+"_t"+std::to_string(timeSlice));

              stochastic_propags_DN_SL.push_back(new PLEGMA_Vector<float>(HOST));
	      stochastic_propags_DN_SL[countindex]->copy(vectorAuxD1, HOST);
              countindex++;

              //stochastic_propags_DN_SL[(isource*tSinks.size()*parallel_sources+k*parallel_sources+l)*n_stochastic_samples+i]->copy(vectorInOut, HOST);
              vectorAuxD1.load();

            }
            //lookuptable[(source[3]+tSinks[k]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]=isource*tSinks.size()*parallel_sources+k*parallel_sources+l;
            lookuptable_DN[timeSlice]=(countindex-1)/n_stochastic_samples;
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

      site source = sourcePositions[isource];

      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",isource, source[0], source[1], source[2], source[3]);
      //updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

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
				     double tmp=vector1.norm();
				     PLEGMA_printf("Norm of source location %d is %e\n",source_location[3],tmp);
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));

				     vectorInOut.absorb(vector2,source_location[DIM_T]);
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

      


      for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){
	PLEGMA_Propagator<float> propUP_SS;
        PLEGMA_Propagator<float> propDN_SS;

	PLEGMA_Propagator<float> propUP_SL;
        PLEGMA_Propagator<float> propDN_SL;

        site& source_local = sourcePositions[isource];
	source_local[3]=(sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

	site source_local_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});
        source_local_reduction[3]=(sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];


  	PLEGMA_ScattCorrelator<float> corrNP(source_local, list_mpf1_twopt);
    	TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"));

        PLEGMA_ScattCorrelator<float> corrN0(source_local, list_mpf1_twopt);
        TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"));

        
        // If twop_filename exists we hold the computation of the light props
        PLEGMA_printf("TESTING %d\n",source_local[DIM_T]);
	TIME(computePropagator(propUP_SS, propUP_SL, mu_ud, LIGHT, nsmearGauss, source_local, false));
	TIME(computePropagator(propDN_SS, propDN_SL, -mu_ud, LIGHT, nsmearGauss, source_local, false));

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

	outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_N";
        TIME( corrN0.apply_phase());
        TIME( corrN0.apply_sign("N"));
        TIME( corrN0.applyBoundaryConditions( true ));
        TIME( corrN0.writeHDF5(outfilename));

        TIME( corrNP.apply_phase() );
        TIME( corrNP.apply_sign("N") );
        TIME( corrNP.applyBoundaryConditions( true ) );
        TIME( corrNP.writeHDF5(outfilename) );

	{
	  PLEGMA_printf("Save propagator for the up  and dn quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propUP_SS,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeHDF5("propUPSS_"+spin+"_c"+col+"_t_"+std::to_string(source_local[3]));
          }
	}


        /*for (int ii=0;ii<12;++ii){
          PLEGMA_Vector<float> temporary1,temporary2;
	  temporary1.absorb(propUP_SS,ii/3,ii%3);
	  temporary2.absorb(propUP_SS_packed,ii/3,ii%3);
	  if (ii==0){
            temporary1.unload();
	    temporary1.writeHDF5("propUPtest_source_j"+std::to_string(i_source_parallel));
	    temporary1.load();
	  }
	  if (ii==0){
            temporary2.unload();
	    temporary2.writeHDF5("propUPtest_before_j"+std::to_string(i_source_parallel));
	    temporary2.load();
	  }
	  temporary2.pack_propagator_from_source_to_sink(temporary1, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
	  if (ii==0){
            temporary2.unload();
	    temporary2.writeHDF5("propUPtest_after_j"+std::to_string(i_source_parallel));
	    temporary2.load();
	  }
	  propUP_SS_packed.absorb(temporary2,ii/3,ii%3);
	}*/

	//int sink_local=(sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	//PLEGMA_printf("sink local %d another right %d\n", sink_local, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
        propUP_SS_packed.pack_propagator_from_source_to_sink(propUP_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        propDN_SS_packed.pack_propagator_from_source_to_sink(propDN_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
//          stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (source[3]+(j+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, j == 0 ? true : false);

        propUP_SL_packed.pack_propagator_from_source_to_sink(propUP_SL, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        propDN_SL_packed.pack_propagator_from_source_to_sink(propDN_SL, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

      } //parallel source position
      {
        PLEGMA_printf("Save propagator for the up  and dn quark\n");
        PLEGMA_Vector<float> vectorAuxPrint(BOTH);
        for(int isc = 0 ; isc < 12 ; isc++){
          std::string spin=std::to_string(isc/3);
          std::string col=std::to_string(isc%3);

          vectorAuxPrint.absorb(propUP_SL_packed,isc/3,isc%3);
          vectorAuxPrint.unload();
          vectorAuxPrint.writeHDF5("propUPSL_packed_s"+spin+"_c"+col);
        }
        for(int isc = 0 ; isc < 12 ; isc++){
          std::string spin=std::to_string(isc/3);
          std::string col=std::to_string(isc%3);

          vectorAuxPrint.absorb(propUP_SS_packed,isc/3,isc%3);
          vectorAuxPrint.unload();
          vectorAuxPrint.writeHDF5("propUPSS_packed_s"+spin+"_c"+col);
        }
	for(int isc = 0 ; isc < 12 ; isc++){
          std::string spin=std::to_string(isc/3);
          std::string col=std::to_string(isc%3);

          vectorAuxPrint.absorb(propDN_SL_packed,isc/3,isc%3);
          vectorAuxPrint.unload();
          vectorAuxPrint.writeHDF5("propDNSL_packed_s"+spin+"_c"+col);
        }
	for(int isc = 0 ; isc < 12 ; isc++){
          std::string spin=std::to_string(isc/3);
          std::string col=std::to_string(isc%3);

          vectorAuxPrint.absorb(propDN_SS_packed,isc/3,isc%3);
          vectorAuxPrint.unload();
          vectorAuxPrint.writeHDF5("propDNSS_packed_s"+spin+"_c"+col);
        }
      }


      //We implement the UD part first
      //The neutron piplus at the source
	
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
	
          PLEGMA_Vector<float> stochastic_source_packed;
	  for (int i_source_parallel=0; i_source_parallel<parallel_sources;i_source_parallel++){
            stochastic_source_packed.pack_propagator_as_sink(stochastic_source_packed, (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], tSinks[k], i_source_parallel == 0 ? true : false);
	  }

          stochastic_source_packed.apply_gamma5();

          //For U(xf1,xf2)
          //B3,B5,B9,B11
	  PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;
          int tsinkMtsource = tSinks[k];
          if(tsinkMtsource >= HGC_totalL[3])
            PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
          for (int j=0; j<parallel_sources;++j){
            int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
          }
          
          TIME(reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k]->V2( stochastic_source_packed,     glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, false));


          //B4,B6
          //B10,B12
          TIME(reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k]->V4( stochastic_source_packed,     glist_sink_nucleon, propUPpacked_to_sink, propDNpacked_to_sink, false));

          //W5,W6,W7,W8
          //W9,W10,W11,W12,W17,W18,W19,W20
	  PLEGMA_Vector<float> stochastic_propagator_packed;
          for (int j=0; j<parallel_sources;++j){
            PLEGMA_Vector<float> temporary;

	    //PLEGMA_printf("Look up table %d TIMESlice %d j %d\n", lookuptable_DN[(source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]],
	    //		    (source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3],j);
	    //fflush(stdout);
            temporary.copy(*stochastic_propags_DN_SL[lookuptable_DN[(source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	    temporary.load();
            stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (source[3]+(j+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, j == 0 ? true : false);
          }

          stochastic_propagator_packed.apply_gamma5();

	  
          TIME(reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k]->V3( stochastic_propagator_packed, glist_insertion,   propUP_SL_packed, true));

          //For D(xf1,xf2)

          stochastic_propagator_packed.apply_gamma5();


	  for (int j=0; j<parallel_sources;++j){
            PLEGMA_Vector<float> temporary;
            temporary.copy(*stochastic_propags_UP_SL[lookuptable_UP[(source[3]+tSinks[k]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	    temporary.load();
	    temporary.writeHDF5("source_for_packing_j"+std::to_string(j)+"_stoch"+std::to_string(i_sample));
            temporary.load();
	  
            stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary,  (source[3]+(j+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, j == 0 ? true : false);
          }

	  stochastic_propagator_packed.unload();
	  stochastic_propagator_packed.writeHDF5("stochastic_up_packed_SL_ns"+std::to_string(i_sample));
	  stochastic_propagator_packed.load();

          stochastic_propagator_packed.apply_gamma5();


          //W13,W14,W15,W16
          //W21,W22,W23,W24
          TIME(reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k]->V3( stochastic_propagator_packed, glist_insertion,   propDN_SL_packed, true));

          //B7,B8
          //B1,B2
          TIME(reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k]->V2( stochastic_source_packed, glist_sink_nucleon, propUPpacked_to_sink, propUPpacked_to_sink, false));

        } //end of for source sink separations
      } //end of for stochastic samples
     
      vectorStoc_source_oet.stochastic_Z(nroots);
      PLEGMA_printf("DONE stochastic factors\n");


#if 1

      {

         //Doing for +mu for the UP propagator spin dilution oet
         if(mu<0) {
           mu*=-1.;
           solver.UpdateSolver();
         }
         for (int l=0; l<parallel_sources;++l){
	   site& source_local = sourcePositions[isource];
           source_local[3]=(sourcePositions[isource][3]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
           //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];


           PLEGMA_Vector<double> vectortmp1;
           PLEGMA_Vector<double> vectortmp2;
           PLEGMA_Vector<double> vectorSave_diluted;


           {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorStoc_source_oet, source_local[3]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectortmp1.absorb(vector2, source_local[3]);

           }

           //Dilution
           vectortmp2.dilutespin(vectortmp1,0);

           //Save the smeared,transformed and diluted source for non-zero momentum oet.
           vectorSave_diluted.copy(vectortmp2);

           for (int spinindex=0; spinindex<4; ++spinindex){
             //Transforming to physical base for the UP quark
             vectortmp2.rotateToPhysicalBasis(vectorSave_diluted,+1);
             //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_reduction_momentum"+std::to_string(spinindex));
             //Doing the zero momentum stochastic propagator with spin dilution
             //Doing the inversion
             TIME(solver.solve(vectortmp2, vectortmp2));
             //Rotate back immediately to the physical basis
             vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
             //Gaussian smearing of the propagator
             TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
	     PLEGMA_Vector<float> temporary1,temporary2;
	     temporary1.copy(*stochastic_oet_prop_u_zero_mom_SS[spinindex],HOST);
	     temporary1.load();
             temporary2.copy(vectortmp2);
	     temporary1.pack_propagator_from_source_to_sink(temporary2, (source[3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, l == 0 ? true : false);
	     temporary1.unload();
             stochastic_oet_prop_u_zero_mom_SS[spinindex]->copy(temporary1,HOST);
	     temporary1.load();

             //temporary1.writeLIME(outfile_V+confnumber+"propagator_up"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));

             vectortmp1.diluteSpinDisplace(vectorSave_diluted,(spinindex+1)%4,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           } //end of loop in spin indices

	 } //end of loop on parallel sources

	 PLEGMA_printf("DONE OET zero mom up\n");

           //Doing for -mu for the DN propagator spin dilution oet
         if(mu>0) {
           mu*=-1.;
           solver.UpdateSolver();
	 }

	 for (int l=0; l<parallel_sources;++l){

           site& source_local = sourcePositions[isource];
           source_local[3]=(sourcePositions[isource][3]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
           //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

           PLEGMA_Vector<double> vectortmp1;
           PLEGMA_Vector<double> vectortmp2;
           PLEGMA_Vector<double> vectorSave_diluted;


           {  // Smearing the source

            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.absorb(vectorStoc_source_oet, source_local[3]);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectortmp1.absorb(vector2, source_local[3]);

           }

           //Dilution
           vectortmp2.dilutespin(vectortmp1,0);

           //Save the smeared,transformed and diluted source for non-zero momentum oet.
           vectorSave_diluted.copy(vectortmp2);


           for (int spinindex=0; spinindex<4; ++spinindex){
             //Transforming to physical base for the DN quark
             vectortmp2.rotateToPhysicalBasis(vectorSave_diluted,-1);
             //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_reduction_momentum"+std::to_string(spinindex));
             //Doing the zero momentum stochastic propagator with spin dilution
             //Doing the inversion
             TIME(solver.solve(vectortmp2, vectortmp2));
             //Rotate back immediately to the physical basis
             vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);
             //Gaussian smearing of the propagator
             TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
	     PLEGMA_Vector<float> temporary1,temporary2;
             temporary1.copy(*stochastic_oet_prop_d_zero_mom_SS[spinindex],HOST);
             temporary1.load();
             temporary2.copy(vectortmp2);
             temporary1.pack_propagator_from_source_to_sink(temporary2, (source[3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, l == 0 ? true : false);
	     temporary1.unload();
             stochastic_oet_prop_d_zero_mom_SS[spinindex]->copy(temporary1,HOST);
             //vectortmp2.writeLIME(outfile_V+confnumber+"propagator_dn"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
	     temporary1.load();
             vectortmp1.diluteSpinDisplace(vectorSave_diluted,(spinindex+1)%4,spinindex);
             vectorSave_diluted.copy(vectortmp1);
	   }//end of loop on spin indices
         } //end of loop on parallel sources
      }//end of do_stochastic_oet
      
      PLEGMA_printf("DONE OET zero mom dn\n");

#endif

      std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV3_diluted_U_DN;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV3_diluted_D_UP;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV4_diluted_STOCHU_DN_UP;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV2_diluted_STOCHU_DN_UP;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV2_diluted_STOCHD_UP_UP;
 
      for (int spinindex=0;spinindex<4;++spinindex){
	try 
	{
          reductionsV3_diluted_U_DN.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));
          reductionsV3_diluted_D_UP.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));

	}
	catch(std::bad_alloc&){
          PLEGMA_printf("Memory allocation fails to store V3 factors oet");
          exit(1);
        }

        for (int k=0; k<tSinks.size();++k){
          try
          {
            reductionsV4_diluted_STOCHU_DN_UP.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
	    reductionsV2_diluted_STOCHU_DN_UP.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
	    reductionsV2_diluted_STOCHD_UP_UP.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
	  }
	  catch(std::bad_alloc&){
            PLEGMA_printf("Memory allocation fails to store V24 factors oet");
            exit(1);
          }
	}
      }
      for (int k=0; k<tSinks.size();++k){
        PLEGMA_Propagator<float> propUPpacked_to_sink;
        PLEGMA_Propagator<float> propDNpacked_to_sink;
        int tsinkMtsource = tSinks[k];
        if(tsinkMtsource >= HGC_totalL[3])
          PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
        for (int j=0; j<parallel_sources;++j){
          int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
          propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
          propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
        }


        for (int spinindex=0;spinindex<4;++spinindex){
          PLEGMA_Vector<float> st_oet_u_zero;
          PLEGMA_Vector<float> st_oet_d_zero;

          st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom_SS[spinindex],HOST);
          st_oet_u_zero.load();
          st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom_SS[spinindex],HOST);
          st_oet_d_zero.load();

	  PLEGMA_Vector<float> st_oet_u_zeropacked_to_sink;
          PLEGMA_Vector<float> st_oet_d_zeropacked_to_sink;

          for (int j=0; j<parallel_sources;++j){
            int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            st_oet_u_zeropacked_to_sink.pack_propagator_as_sink(st_oet_u_zero,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            st_oet_d_zeropacked_to_sink.pack_propagator_as_sink(st_oet_d_zero,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
          }
 


          TIME(reductionsV2_diluted_STOCHU_DN_UP[4*k+spinindex]->V2( st_oet_u_zeropacked_to_sink, glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, false));
          TIME(reductionsV2_diluted_STOCHD_UP_UP[4*k+spinindex]->V2( st_oet_d_zeropacked_to_sink, glist_sink_nucleon, propUPpacked_to_sink, propUPpacked_to_sink, false));
          TIME(reductionsV4_diluted_STOCHU_DN_UP[4*k+spinindex]->V4( st_oet_u_zeropacked_to_sink, glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, false));

        }
      }

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
		 
	PLEGMA_Propagator<float> propTS_SS_packed;
        PLEGMA_Propagator<float> propTS_SL_packed;


	for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

	  PLEGMA_Propagator<float> propTS_SS;
          PLEGMA_Propagator<float> propTS_SL;


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
            vectorAuxD2.apply_gamma_scatt(glist_source_meson[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_insertion[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert UP for sequential propagator UP  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1);
            //performing smearing
	    vectorAuxF.copy(vectorAuxD2);
            propTS_SL.absorb(vectorAuxF, isc/3, isc%3);
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          }

	  propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
          propTS_SL_packed.pack_propagator_from_source_to_sink(propTS_SL, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

	}



	PLEGMA_printf("source %d\n",source[3]);
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

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";


	TIME(produceOutput(corrTproton_protonpizero1, outfilename, "T"));
	TIME(produceOutput(corrTproton_protonpizero2, outfilename, "T"));
        TIME(produceOutput(corrTproton_protonpizero3, outfilename, "T"));
        TIME(produceOutput(corrTproton_protonpizero4, outfilename, "T"));


	//creating factors

	PLEGMA_Vector<float> stochastic_propagator_packed;
	for (int k=0; k<tSinks.size(); ++k){
       
          PLEGMA_ScattCorrelator<float> corrB3(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB4(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB5(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB6(source, filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW5(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW6(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW7(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW8(source, filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW13(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW14(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW15(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW16(source, filtered_sourcemomentumList);


          ssource=(char *)malloc(sizeof(char)*100);
          sprintf(ssource,"B3_deltat_%d",tSinks[k]);
          corrB3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B4_deltat_%d",tSinks[k]);
          corrB4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B5_deltat_%d",tSinks[k]);
          corrB5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B6_deltat_%d",tSinks[k]);
          corrB6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W5_deltat_%d",tSinks[k]);
          corrW5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W6_deltat_%d",tSinks[k]);
          corrW6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W7_deltat_%d",tSinks[k]);
          corrW7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W8_deltat_%d",tSinks[k]);
          corrW8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W13_deltat_%d",tSinks[k]);
          corrW13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W14_deltat_%d",tSinks[k]);
          corrW14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W15_deltat_%d",tSinks[k]);
          corrW15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W16_deltat_%d",tSinks[k]);
          corrW16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          free(ssource);
        
	  PLEGMA_Propagator<float> propTS_SS_packed_to_sink;
  
          for (int i_source_parallel=0; i_source_parallel< parallel_sources;++i_source_parallel){
            int tsinkMtsource = tSinks[k];
            if(tsinkMtsource >= HGC_totalL[3])
              PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
            int global_fixSinkTime = (tsinkMtsource + source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            propTS_SS_packed_to_sink.pack_propagator_as_sink(propTS_SS_packed,  global_fixSinkTime, tsinkMtsource, i_source_parallel==0 ? true : false);
          }

	  PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;
          {
            for (int j=0; j< parallel_sources;++j){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
              propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            }
          }

          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
	      PLEGMA_Vector<float> temporary;
	      temporary.copy(*stochastic_propags_DN_SL[lookuptable_DN[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	      temporary.load();

	      stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
	    }

            stochastic_propagator_packed.apply_gamma5();
	    
	    //V3
            TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion, propTS_SL_packed, true));

	    TIME(corrB3.B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0, 3, true));
            TIME(corrB4.B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 4, true));
	    TIME(corrB5.B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0, 5, true));
            TIME(corrB6.B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 6, true));

	    stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
	    stochastic_propagator_packed.load();
	    PLEGMA_Vector<float> stochastic_source_packed;
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;i_source_parallel++){
              stochastic_source_packed.pack_propagator_as_sink(stochastic_propagator_packed, (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], tSinks[k], i_source_parallel == 0 ? true : false);
            }

	    stochastic_source_packed.apply_gamma5();



            TIME(reductionsV2.V4( stochastic_source_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));//checked

	    TIME(corrW5.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 5, true));

	    TIME(corrW7.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 7, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));//checked

            TIME(corrW6.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()], reductionsV2, 0, 6, true));
            TIME(corrW8.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()], reductionsV2, 0, 8, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));//checked

            TIME(corrW13.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 13, true));
            TIME(corrW15.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 15, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propUPpacked_to_sink, propTS_SS_packed_to_sink, false));//checked
            TIME(corrW14.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 14, true));
            TIME(corrW16.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 16, true));

	  }//loop over stochastic samples

	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput_3pt(corrB3, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrB4, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrB5, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrB6, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput_3pt(corrW5, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));
          TIME(produceOutput_3pt(corrW6, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));
          TIME(produceOutput_3pt(corrW7, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));
          TIME(produceOutput_3pt(corrW8, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));

          TIME(produceOutput_3pt(corrW13, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));
          TIME(produceOutput_3pt(corrW14, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));
          TIME(produceOutput_3pt(corrW15, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));
          TIME(produceOutput_3pt(corrW16, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations));


        }//loop over source sink separations


        // ensuring mu positive
        if(mu<0) {
          mu*=-1.;
          solver.UpdateSolver();
        }

	for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

          PLEGMA_Propagator<float> propTS_SS;
          PLEGMA_Propagator<float> propTS_SL;


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
            vectorAuxD2.apply_gamma_scatt(glist_source_meson[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_insertion[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1);
	    vectorAuxF.copy(vectorAuxD2);
            propTS_SL.absorb(vectorAuxF, isc/3, isc%3);

            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          }

          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
          propTS_SL_packed.pack_propagator_from_source_to_sink(propTS_SL, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

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

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

	TIME(produceOutput(corrTproton_neutronpiplus1, outfilename, "T"));
	TIME(produceOutput(corrTproton_neutronpiplus2, outfilename, "T"));
        TIME(produceOutput(corrTproton_neutronpiplus3, outfilename, "T"));
        TIME(produceOutput(corrTproton_neutronpiplus4, outfilename, "T"));





        for (int k=0; k< tSinks.size(); ++k){
         
          PLEGMA_ScattCorrelator<float> corrB9(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB10(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB11(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB12(source, filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW17(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW18(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW19(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW20(source, filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrW21(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW22(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW23(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW24(source, filtered_sourcemomentumList);


	  ssource=(char *)malloc(sizeof(char)*100);
          sprintf(ssource,"B9_deltat_%d",tSinks[k]);
          corrB9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B10_deltat_%d",tSinks[k]);
          corrB10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B11_deltat_%d",tSinks[k]);
          corrB11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B12_deltat_%d",tSinks[k]);
          corrB12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W17_deltat_%d",tSinks[k]);
          corrW17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W18_deltat_%d",tSinks[k]);
          corrW18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W19_deltat_%d",tSinks[k]);
          corrW19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W20_deltat_%d",tSinks[k]);
          corrW20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W21_deltat_%d",tSinks[k]);
          corrW21.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W22_deltat_%d",tSinks[k]);
          corrW22.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W23_deltat_%d",tSinks[k]);
          corrW23.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W24_deltat_%d",tSinks[k]);
          corrW24.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          free(ssource);

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
          for (int j=0; j<parallel_sources;++j){
            int tsinkMtsource = tSinks[k];
            if(tsinkMtsource >= HGC_totalL[3])
              PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
            int global_fixSinkTime = (tsinkMtsource + source[3]+j*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
            propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  global_fixSinkTime, tsinkMtsource, j==0 ? true : false);
          }

          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            PLEGMA_Vector<float> stochastic_propagator_packed;
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
              PLEGMA_Vector<float> temporary;
              temporary.copy(*stochastic_propags_DN_SL[lookuptable_DN[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	      temporary.load(); 

              stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
            }

            stochastic_propagator_packed.apply_gamma5();

	    TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion,   propTS_SL_packed, true ));

            TIME( corrB9.B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0,  9, true));
            TIME(corrB10.B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 10, true));
            TIME(corrB11.B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], 0, 11, true));
            TIME(corrB12.B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], 0, 12, true));

            stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
            stochastic_propagator_packed.load();

	    PLEGMA_Vector<float> stochastic_source_packed;
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;i_source_parallel++){
              stochastic_source_packed.pack_propagator_as_sink(stochastic_propagator_packed, (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], tSinks[k], i_source_parallel == 0 ? true : false);
            }

            stochastic_source_packed.apply_gamma5();


            TIME(reductionsV2.V4( stochastic_source_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));
            TIME(corrW17.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 17, true));
            TIME(corrW19.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 19, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, false));
            TIME(corrW18.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 18, true));
            TIME(corrW20.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 20, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propUPpacked_to_sink, propTS_SS_packed_to_sink, false));
	    TIME(corrW22.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 22, true));
            TIME(corrW24.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 24, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));

	    TIME(corrW21.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 21, true));
            TIME(corrW23.W_diagrams( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, 0, 23, true));

	  } //loop over stochastic samples

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput_3pt(corrB9, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations  ));
          TIME(produceOutput_3pt(corrB10, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrB11, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrB12, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput_3pt(corrW17, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW18, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW19, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW20, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));

          TIME(produceOutput_3pt(corrW21, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW22, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW23, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW24, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));




	} //loop over source sink separations


        for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

          PLEGMA_Propagator<float> propTS_SS;
          PLEGMA_Propagator<float> propTS_SL;


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
            vectorAuxD2.apply_gamma_scatt(glist_source_meson[0]);
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
            vectorAuxF.copy(vectorAuxD);
            propTS_SL.absorb(vectorAuxF, isc/3, isc%3);
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          }

          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
          propTS_SL_packed.pack_propagator_from_source_to_sink(propTS_SL, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

        }

        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero5(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero6(source, list_mpi2ptot);

        corrTproton_protonpizero5.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq25");

        corrTproton_protonpizero6.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq26");

        TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propUP_SS_packed));
        TIME(corrTproton_protonpizero5.convertTreductiontoDiagram( reductionsT1, false, true, false ));

        TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propDN_SS_packed));
        TIME(corrTproton_protonpizero6.convertTreductiontoDiagram( reductionsT2, false, false, true ));

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

	TIME(produceOutput(corrTproton_protonpizero5, outfilename, "T"));
        TIME(produceOutput(corrTproton_protonpizero6, outfilename, "T"));

	for (int k=0; k< tSinks.size(); ++k){

          PLEGMA_ScattCorrelator<float> corrB7(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB8(source, filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW9(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW10(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW11(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW12(source, filtered_sourcemomentumList);

	  ssource=(char *)malloc(sizeof(char)*100);
          sprintf(ssource,"B7_deltat_%d",tSinks[k]);
          corrB7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"B8_deltat_%d",tSinks[k]);
          corrB8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"W9_deltat_%d",tSinks[k]);
          corrW9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W10_deltat_%d",tSinks[k]);
          corrW10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W11_deltat_%d",tSinks[k]);
          corrW11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"W12_deltat_%d",tSinks[k]);
          corrW12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          free(ssource);

          PLEGMA_Propagator<float> propTS_SS_packed_to_sink;
          {
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int global_fixSinkTime = (tsinkMtsource + source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
              propTS_SS_packed_to_sink.pack_propagator_as_sink(propTS_SS_packed,  global_fixSinkTime, tsinkMtsource, i_source_parallel==0 ? true : false);
            }
          }

          PLEGMA_Propagator<float> propUPpacked_to_sink;
          
          for (int i_source_parallel=0; i_source_parallel< parallel_sources;++i_source_parallel){
            int tsinkMtsource = tSinks[k];
            if(tsinkMtsource >= HGC_totalL[3])
              PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
            int global_fixSinkTime = (tsinkMtsource + source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  global_fixSinkTime, tsinkMtsource, i_source_parallel==0 ? true : false);
          }
       

          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            PLEGMA_Vector<float> stochastic_propagator_packed;
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              PLEGMA_Vector<float> temporary;
              temporary.copy(*stochastic_propags_UP_SL[lookuptable_UP[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	      temporary.load();

              stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (sourcePositions[isource][3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
            }

            stochastic_propagator_packed.apply_gamma5();

	    TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion,   propTS_SS_packed, true));

            TIME(corrB7.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k], 0, 7, true));
            TIME(corrB8.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k], 0, 8, true));

            stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
            stochastic_propagator_packed.load();

	    PLEGMA_Vector<float> stochastic_source_packed;
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;i_source_parallel++){
              stochastic_source_packed.pack_propagator_as_sink(stochastic_propagator_packed, (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], tSinks[k], i_source_parallel == 0 ? true : false);
            }

            stochastic_source_packed.apply_gamma5();

	    TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));

            TIME( corrW9.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0,  9, true));
            TIME(corrW11.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 11, true));

            TIME(reductionsV2.V4( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, false));

            TIME(corrW10.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 10, true));
            TIME(corrW12.W_diagrams( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, 0, 12, true));

	  } //loop over stochastic samples

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput_3pt(corrB7, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrB8, outfilename, "4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput_3pt(corrW9 , outfilename,"4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));//because of V4
          TIME(produceOutput_3pt(corrW10, outfilename,"4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));//because of V4
          TIME(produceOutput_3pt(corrW11, outfilename,"4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrW12, outfilename,"4pt", n_stochastic_samples, tSinks[k], max_source_sink_separations ));


	} //loop over tSinks


	{ //Finite momentum oet
          PLEGMA_Vector<double> vectortmp1;
          PLEGMA_Vector<double> vectortmp2;
          PLEGMA_Vector<double> vectorSource_finite_mom;

          //Doing for +mu for the UP propagator spin dilution oet
          if(mu<0) {
            mu*=-1.;
            solver.UpdateSolver();
          }


	  for (int l=0; l<parallel_sources;++l){
            site& source_local = sourcePositions[isource];
            source_local[3]=(sourcePositions[isource][3]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

            //Multiplying by the appropriate momentum phase

            vectortmp1.dilutespin(vectorStoc_source_oet,0);
            vectorSource_finite_mom.copy(vectortmp1);

            {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorSource_finite_mom, source_local[3]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorSource_finite_mom.absorb(vector2, source_local[3]);

            }
            std::vector<int> tmp_4Dmom= momentum_i2 ;
            tmp_4Dmom.push_back(0);
            vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);



            for (int spinindex=0; spinindex<4; ++spinindex){

              //Transforming to physical base for the UP quark
              vectortmp2.rotateToPhysicalBasis(vectorSource_finite_mom,+1);

              //Doing the inversion
              TIME(solver.solve(vectortmp2, vectortmp2));

              //Rotate back immediately to the physical basis
              vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);

	      PLEGMA_Vector<float> temporary1,temporary2;
              temporary1.copy(*stochastic_oet_prop_u_fini_mom_SL[spinindex],HOST);
              temporary1.load();
              temporary2.copy(vectortmp2);
              temporary1.pack_propagator_from_source_to_sink(temporary2,(source[3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, l == 0 ? true : false);
              temporary1.unload();
              stochastic_oet_prop_u_fini_mom_SL[spinindex]->copy(temporary1,HOST);
              temporary1.load();



              vectortmp1.diluteSpinDisplace(vectorSource_finite_mom,(spinindex+1)%4,spinindex);
              vectorSource_finite_mom.copy(vectortmp1);

            } //end of spin dilution 

          } // end of parallel sources

          //Doing for +mu for the DN propagator spin dilution oet
          if(mu>0) {
            mu*=-1.;
            solver.UpdateSolver();
          }


          for (int l=0; l<parallel_sources;++l){
            site& source_local = sourcePositions[isource];
            source_local[3]=(sourcePositions[isource][3]+l*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

            //Multiplying by the appropriate momentum phase

            vectortmp1.dilutespin(vectorStoc_source_oet,0);
            vectorSource_finite_mom.copy(vectortmp1);

            {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorSource_finite_mom, source_local[3]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorSource_finite_mom.absorb(vector2, source_local[3]);

            }

            std::vector<int> tmp_4Dmom= momentum_i2 ;
            tmp_4Dmom.push_back(0);
            vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);

            for (int spinindex=0; spinindex<4; ++spinindex){

              //Transforming to physical base for the UP quark
              vectortmp2.rotateToPhysicalBasis(vectorSource_finite_mom,-1);

              //Doing the inversion
              TIME(solver.solve(vectortmp2, vectortmp2));

              //Rotate back immediately to the physical basis
              vectortmp2.rotateToPhysicalBasis(vectortmp1,-1);

              PLEGMA_Vector<float> temporary1,temporary2;
              temporary1.copy(*stochastic_oet_prop_d_fini_mom_SL[spinindex],HOST);
	      temporary1.load();
              temporary2.copy(vectortmp2);
              temporary1.pack_propagator_from_source_to_sink(temporary2,(source[3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3],max_source_sink_separations, l == 0 ? true : false);
	      temporary1.unload();
              stochastic_oet_prop_d_fini_mom_SL[spinindex]->copy(temporary1,HOST);
	      temporary1.load();

              vectortmp1.diluteSpinDisplace(vectorSource_finite_mom,(spinindex+1)%4,spinindex);
              vectorSource_finite_mom.copy(vectortmp1);

            } //end of spin dilution

          } // end of parallel sources

        }//end of finite momentum oet


        for (int i=0; i< 4; ++i){
          PLEGMA_Vector<float> st_oet_d_fini;
          st_oet_d_fini.copy(*stochastic_oet_prop_d_fini_mom_SL[i],HOST);
          st_oet_d_fini.load();

          st_oet_d_fini.apply_gamma5();

          TIME(reductionsV3_diluted_D_UP[i]->V3( st_oet_d_fini, glist_insertion, propUP_SL_packed, true));

        }

	for (int i=0; i< 4; ++i){
          PLEGMA_Vector<float> st_oet_u_fini;
          st_oet_u_fini.copy(*stochastic_oet_prop_u_fini_mom_SL[i],HOST);
	  st_oet_u_fini.load();

	  st_oet_u_fini.apply_gamma5();

          TIME(reductionsV3_diluted_U_DN[i]->V3( st_oet_u_fini, glist_insertion, propDN_SL_packed, true));
	}



	for (int k=0; k<tSinks.size();++k){

          PLEGMA_ScattCorrelator<float> corrZ5(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ6(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ7(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ8(source, filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrZ9(source,  filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ10(source, filtered_sourcemomentumList);

	  PLEGMA_ScattCorrelator<float> corrZ11(source, filtered_sourcemomentumList);
    	  PLEGMA_ScattCorrelator<float> corrZ12(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ13(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ14(source, filtered_sourcemomentumList);

	  ssource=(char *)malloc(sizeof(char)*100);
          sprintf(ssource,"Z5_deltat_%d",tSinks[k]);
          corrZ5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"Z6_deltat_%d",tSinks[k]);
          corrZ6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"Z7_deltat_%d",tSinks[k]);
          corrZ7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"Z8_deltat_%d",tSinks[k]);
          corrZ8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"Z9_deltat_%d",tSinks[k]);
          corrZ9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"Z10_deltat_%d",tSinks[k]);
          corrZ10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

	  sprintf(ssource,"Z11_deltat_%d",tSinks[k]);
          corrZ11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"Z12_deltat_%d",tSinks[k]);
          corrZ12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);

          sprintf(ssource,"Z13_deltat_%d",tSinks[k]);
          corrZ13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
          sprintf(ssource,"Z14_deltat_%d",tSinks[k]);
          corrZ14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_insertion, "12", ssource);
	  free(ssource);

	  TIME(corrZ5.Z_diagrams( reductionsV3_diluted_D_UP, reductionsV2_diluted_STOCHU_DN_UP, 4*k, 5 ));
          TIME(corrZ7.Z_diagrams( reductionsV3_diluted_D_UP, reductionsV2_diluted_STOCHU_DN_UP, 4*k, 7 ));

	  TIME(corrZ6.Z_diagrams( reductionsV3_diluted_D_UP, reductionsV4_diluted_STOCHU_DN_UP, 4*k, 6 ));
          TIME(corrZ8.Z_diagrams( reductionsV3_diluted_D_UP, reductionsV4_diluted_STOCHU_DN_UP, 4*k, 8 ));

          TIME(corrZ9.Z_diagrams(  reductionsV3_diluted_U_DN, reductionsV2_diluted_STOCHD_UP_UP,4*k, 9 ));
          TIME(corrZ10.Z_diagrams( reductionsV3_diluted_U_DN, reductionsV2_diluted_STOCHD_UP_UP,4*k, 10));


	  TIME(corrZ11.Z_diagrams( reductionsV3_diluted_U_DN, reductionsV2_diluted_STOCHU_DN_UP,4*k, 11 ));
          TIME(corrZ13.Z_diagrams( reductionsV3_diluted_U_DN, reductionsV2_diluted_STOCHU_DN_UP,4*k, 13 ));

	  TIME(corrZ12.Z_diagrams( reductionsV3_diluted_U_DN, reductionsV4_diluted_STOCHU_DN_UP,4*k, 12 ));
          TIME(corrZ14.Z_diagrams( reductionsV3_diluted_U_DN, reductionsV4_diluted_STOCHU_DN_UP,4*k, 14 ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

	  TIME(produceOutput_3pt(corrZ5, outfilename, "4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ6, outfilename, "4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ7, outfilename, "4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ8, outfilename, "4pt", tSinks[k], max_source_sink_separations ));

          TIME(produceOutput_3pt(corrZ9, outfilename, "4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ10, outfilename,"4pt", tSinks[k], max_source_sink_separations ));

          TIME(produceOutput_3pt(corrZ11, outfilename,"4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ12, outfilename,"4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ13, outfilename,"4pt", tSinks[k], max_source_sink_separations ));
          TIME(produceOutput_3pt(corrZ14, outfilename,"4pt", tSinks[k], max_source_sink_separations ));


	}//end for loop source sink separations

      }//loop over mpi2

      for (int spinindex=0;spinindex<4;++spinindex){
        reductionsV3_diluted_U_DN.pop_back();
        reductionsV3_diluted_D_UP.pop_back();

        for (int k=0; k<tSinks.size();++k){
          reductionsV4_diluted_STOCHU_DN_UP.pop_back();
          reductionsV2_diluted_STOCHU_DN_UP.pop_back();
	  reductionsV2_diluted_STOCHD_UP_UP.pop_back();
        }
      }


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
      stochastic_propags_UP_SL.pop_back();
      stochastic_propags_DN_SL.pop_back();
    }

    for(int i=0; i< 4; ++i) {
      stochastic_oet_prop_d_zero_mom_SS.pop_back();
      stochastic_oet_prop_u_zero_mom_SS.pop_back();

      stochastic_oet_prop_d_fini_mom_SL.pop_back();
      stochastic_oet_prop_u_fini_mom_SL.pop_back();

    }


  }//loop in finalize
  finalize();
  return 0;
}
