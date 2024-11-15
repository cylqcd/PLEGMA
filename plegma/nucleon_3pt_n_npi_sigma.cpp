#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <util_quda.h>
std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

#define COMPUTEBACKWARD true

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

void produceOutput_2pt_packed( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_stochastic_samples,
		    int n_coherent_source,
		    int *attract_look_up_table){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions( true, n_coherent_source, attract_look_up_table ));
  TIME(source.normalize_nstoch(n_stochastic_samples));
  TIME(source.writeHDF5( outputFilename ));

}


void produceOutput_3pt( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_stochastic_samples,
		    int n_coherent_source,
                    int *attract_look_up_table,
		    int source_sink_separation){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions_3pt( true, n_coherent_source, attract_look_up_table, source_sink_separation ));
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

void produceOutput_2pt_packed( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
		    int n_coherent_source,
                    int *attract_look_up_table
                  ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions( true, n_coherent_source, attract_look_up_table ));
  TIME(source.writeHDF5( outputFilename ));
}


void produceOutput_3pt( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
		    int n_coherent_source,
                    int *attract_look_up_table,
		    int source_sink_separation
                  ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions_3pt( true,n_coherent_source, attract_look_up_table, source_sink_separation ));
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

  std::vector<GAMMAS_SCATT> glist_source_meson={ID};
  std::vector<GAMMAS_SCATT> glist_sink_meson={ID};

  std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4,S_12,S_13,S_23,S_41,S_42,S_43};//,S12,S13,S23,S41,S42,S43};

  int n_stochastic_samples;
  int max_source_sink_separations;
  int dotwopoint;
  int readStochSamples;
  //setVerbosity(QUDA_DEBUG_VERBOSE);

  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("maxSourceSinkSeparations", "Maximal source sink separations", verbosity, max_source_sink_separations);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);
  HGC_options->set("dotwopoint", "Doing also the twopoint functions", verbosity,dotwopoint);
  HGC_options->set("readStochSamples", "Flag for switching read/building stochastic propagators", verbosity, readStochSamples);



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
    momList sourcemomentumList_threept(4,pathListMomenta_threept,{1,2,3});
    PLEGMA_printf("N momenta in sourcemomentumList: %d\n",sourcemomentumList_threept.size());
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

    std::vector<std::vector<int>> mpf2_twopt = sourcemomentumList_twopt.uniq_p(2);
    momList list_mpf2_twopt(1,{mpf2_twopt,},{0,});


    int parallel_sources=HGC_totalL[3]/max_source_sink_separations;
    std::vector<int> lookuptable_UP;
    std::vector<int> lookuptable_DN;

    
    for (int i=0; i<HGC_totalL[3]; ++i){
      lookuptable_UP.push_back(-1);
      lookuptable_DN.push_back(-1);
    }

    int *attract_lookup_table;
    attract_lookup_table=(int *)malloc(sizeof(int)*HGC_totalL[3]);
#if defined (COMPUTEBACKWARD)
    int *attract_lookup_table_backward;
    attract_lookup_table_backward=(int *)malloc(sizeof(int)*HGC_totalL[3]);
#endif



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

    std::vector<PLEGMA_Vector<float>*> stochastic_propagator_2pt_SS;

    std::vector<PLEGMA_Vector<float>*> stochastic_propags_UP_SL;

    std::vector<PLEGMA_Vector<float>*> stochastic_propags_DN_SL;
#if defined (COMPUTEBACKWARD)

    PLEGMA_Vector<float> stochastic_oet_prop_u_zero_mom_SS_backward;

#endif

    PLEGMA_Vector<float> stochastic_oet_prop_u_zero_mom_SS;

#if defined (COMPUTEBACKWARD)

    PLEGMA_Vector<float> stochastic_oet_prop_d_zero_mom_SS_backward;

#endif

    PLEGMA_Vector<float> stochastic_oet_prop_d_zero_mom_SS;


    PLEGMA_Vector<float> stochastic_oet_prop_u_fini_mom_SL;

    PLEGMA_Vector<float> stochastic_oet_prop_d_fini_mom_SL;

    PLEGMA_Vector<float> stochastic_oet_prop_u_fini_mom_SS;

    PLEGMA_Vector<float> stochastic_oet_prop_d_fini_mom_SS;

#if defined (COMPUTEBACKWARD)
        
    PLEGMA_Vector<float> stochastic_oet_prop_u_fini_mom_SS_backward;

    PLEGMA_Vector<float> stochastic_oet_prop_d_fini_mom_SS_backward;

#endif



    /******************************************************
     *
     *Step 1: Producing the stochastic sources
     *
     ******************************************************/
#if 1 
    for (int i=0; i<n_stochastic_samples;++i){
   
      stochastic_sources.push_back(new PLEGMA_Vector<float>(HOST));
      if (dotwopoint==1){
        stochastic_propagator_2pt_SS.push_back(new PLEGMA_Vector<float>(HOST));
      }
      if (readStochSamples==0){
        
	vectorSource_stochastic.stochastic_Z(nroots);
        vectorSource_stochastic.unload();
        vectorSource_stochastic.writeLIME("globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
#if 0
	PLEGMA_Vector<float> vectorRead(BOTH);
	vectorRead.readFile("stochastic_source.0000.00000_plegma_conventions.lime",LIME_FORMAT);
	vectorRead.load();
	vectorRead.rotate_uk_ch_g5g4();
        vectorRead.apply_gamma(G2);
	vectorRead.unload();
        stochastic_sources[i]->copy(vectorRead,HOST);
#endif
        stochastic_sources[i]->copy(vectorSource_stochastic,HOST);
	vectorSource_stochastic.load();
      }
      else{
        std::string inputfilename="globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read stochastic source from: %s\n",inputfilename.c_str());
        PLEGMA_Vector<double> vectorRead(BOTH);
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        stochastic_sources[i]->copy(vectorRead,HOST);
        inputfilename="globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
	PLEGMA_Vector<float> vectorFloat(BOTH);
        PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
        vectorFloat.readFile(inputfilename,LIME_FORMAT);
        stochastic_propagator_2pt_SS[i]->copy(vectorFloat,HOST);
      }
    }


    /******************************************************
     *
     *Step 2: Producing the stochastic propagators
     *
     ******************************************************/

    if ((dotwopoint==1) && (readStochSamples==0)){
      for (int i=0; i<n_stochastic_samples;++i){
        PLEGMA_Vector<double> vectorInOut, vectorAuxD1,vectorAuxD2;

        vectorAuxD1.copy(*stochastic_sources[i], HOST);
        vectorAuxD1.load();

        //Step(3) Smearing all the time slice
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(4) We rotate the source to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //In vectorAuxD2 we store the results for the inversion
        vectorAuxD2.scale(0.0);

        PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
        for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
          //Step(5) pick out a particular timeslice from the source
	  //
	  //PLEGMA_Vector3D<double> vec3D;
	  //vec3D.absorb(vectorAuxD1, timeidx);
	  //vectorInOut.absorb(vec3D, timeidx);
	  //vectorInOut.copy(vectorAuxD1);
          vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
          //Step(6) Solve
          TIME(solver.solve(vectorInOut, vectorInOut));
          //Step(7) absorbing the particular timeslice to a 4d vector
	  //vec3D.absorb(vectorInOut, timeidx);
	  //vectorAuxD2.absorb(vec3D, timeidx, false);
          vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
        }

        //Step(6) We rotate back the propagator to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //Step(7) Smearing all the time slice in the propagator
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(8) Save the propagator to the disk
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorAuxD2);
          vectorAuxF.unload();
          vectorAuxF.writeLIME("globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber);
        }

        //Step(9) Save the propagator to the host memory
        vectorAuxD2.unload();
        stochastic_propagator_2pt_SS[i]->copy(vectorAuxD2,HOST);
        vectorAuxD2.load();

      }
    }

#endif

#if 1
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
	      //We do not perform smearing for stochastic propagator
	      //because it always ends at the insertion
              //TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

              vectorAuxD1.unload();
	      //PLEGMA_printf("Save the stochastic source for sample %d\n",i);
              //std::string nstoch=std::to_string(i);
              //vectorAuxD1.writeHDF5("stochastic_propagators_UP"+nstoch+"_t"+std::to_string(timeSlice));

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

	      vectorAuxD1.unload();
              stochastic_propags_DN_SL.push_back(new PLEGMA_Vector<float>(HOST));
	      stochastic_propags_DN_SL[countindex]->copy(vectorAuxD1, HOST);
              countindex++;

              //stochastic_propags_DN_SL[(isource*tSinks.size()*parallel_sources+k*parallel_sources+l)*n_stochastic_samples+i]->copy(vectorInOut, HOST);
              vectorAuxD1.load();

            }
            lookuptable_DN[timeSlice]=(countindex-1)/n_stochastic_samples;
          }
        }
      }
    }

#endif


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

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear,  site &source_location, bool finalize) {
                                 PLEGMA_Gauge3D<double> smearedGauge3D;
                                 smearedGauge3D.absorb(smearedGauge, source_location[DIM_T]);
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

#if defined (COMPUTEBACKWARD)

      PLEGMA_Propagator<float> propUP_SS_packed_backward;
      PLEGMA_Propagator<float> propDN_SS_packed_backward;

#endif


      PLEGMA_Propagator<float> propUP_SS_packed;
      PLEGMA_Propagator<float> propDN_SS_packed;

      PLEGMA_Propagator<float> propUP_SL_packed;
      PLEGMA_Propagator<float> propDN_SL_packed;

      for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){
	for (int sink=0; sink<max_source_sink_separations;++sink){
	  attract_lookup_table[(sourcePositions[isource][DIM_T]+max_source_sink_separations*i_source_parallel+sink)%HGC_totalL[3]]=(sourcePositions[isource][DIM_T]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	}
      }

#if defined (COMPUTEBACKWARD)

      for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){
        for (int sink=0; sink<max_source_sink_separations;++sink){
          attract_lookup_table_backward[(sourcePositions[isource][DIM_T]+max_source_sink_separations*i_source_parallel+sink)%HGC_totalL[3]]=(sourcePositions[isource][DIM_T]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
        }
      }

#endif

      for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

	PLEGMA_Propagator<float> propUP_SS;
        PLEGMA_Propagator<float> propDN_SS;

	PLEGMA_Propagator<float> propUP_SL;
        PLEGMA_Propagator<float> propDN_SL;

        //site source_local = sourcePositions[isource];
	//source_local[3]=(sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
        site source_local;
	source_local[0] = sourcePositions[isource][0];
	source_local[1] = sourcePositions[isource][1];
	source_local[2] = sourcePositions[isource][2];
	source_local[3] = (sourcePositions[isource][DIM_T]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	site source_localPtSinkMtSource;
	source_localPtSinkMtSource[0] = sourcePositions[isource][0];
        source_localPtSinkMtSource[1] = sourcePositions[isource][1];
        source_localPtSinkMtSource[2] = sourcePositions[isource][2];
        source_localPtSinkMtSource[3] = (sourcePositions[isource][DIM_T]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

        site source_localTotalMsink;
        source_localTotalMsink[0] = sourcePositions[isource][0];
        source_localTotalMsink[1] = sourcePositions[isource][1];
        source_localTotalMsink[2] = sourcePositions[isource][2];
        source_localTotalMsink[3] = ((sourcePositions[isource][DIM_T]+(i_source_parallel)*max_source_sink_separations)+HGC_totalL[3])%HGC_totalL[3];


	site source_local_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});
        source_local_reduction[3]=(sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];


  	PLEGMA_ScattCorrelator<float> corrNP(source_local, list_mpf1_twopt);
    	TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"));

        PLEGMA_ScattCorrelator<float> corrN0(source_local, list_mpf1_twopt);
        TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"));

        
        // If twop_filename exists we hold the computation of the light props
	TIME(computePropagator(propUP_SS, propUP_SL, mu_ud, LIGHT, nsmearGauss, source_local, false));
	TIME(computePropagator(propDN_SS, propDN_SL, -mu_ud, LIGHT, nsmearGauss, source_local, false));

	if (dotwopoint==1){

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
            TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN_SS, propUP_SS, propDN_SS));
            TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ));
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
	}
	
        propUP_SS_packed.pack_propagator_from_source_to_sink(propUP_SS, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        propDN_SS_packed.pack_propagator_from_source_to_sink(propDN_SS, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)
        propUP_SS_packed_backward.pack_propagator_from_source_to_sink(propUP_SS, source_localTotalMsink[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        propDN_SS_packed_backward.pack_propagator_from_source_to_sink(propDN_SS, source_localTotalMsink[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
#endif

        propUP_SL_packed.pack_propagator_from_source_to_sink(propUP_SL, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        propDN_SL_packed.pack_propagator_from_source_to_sink(propDN_SL, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

      } //parallel source position

    /******************************************************
     *
     * Step 4: Computing the factors from the point to all 
     *         and 
     *         stochastic pieces
     *
     ******************************************************/


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
      

      //Here the prefix DD means that reduction is based phi*g5 and xi*g5
      //We replace D(x_f2,x_f1) with xi(x_f2)*gamma_5* phi^dagger(x_f1) *gamma_5
      //phi goes to V3 reduction and xi goes to V2 reduction
      
      //For the proton pizero x neutron piplus


      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U_2pt;//implemented

#if defined (COMPUTEBACKWARD)
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U_2pt_backward;//implemented
#endif

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D;//implemented

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D_2pt;//implemented

#if defined (COMPUTEBACKWARD)
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D_2pt_backward;//implemented
#endif


      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U_2pt;//implemented
#if defined (COMPUTEBACKWARD)
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U_2pt_backward;//implemented
#endif
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D_2pt;//implemented

#if defined (COMPUTEBACKWARD)
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D_2pt_backward;//implemented
#endif

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U_2pt;
#if defined (COMPUTEBACKWARD)
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U_2pt_backward;
#endif
#if defined (COMPUTEBACKWARD)
#endif





      for(int i=0; i< n_stochastic_samples; ++i) {
	if (dotwopoint==1){
          reductions_DD_V2_GAMMAF1U_U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));

          reductions_UU_V2_GAMMAF1D_U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));
          reductions_UU_V4_GAMMAF1U_D_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));


	  reductions_UU_V3_GAMMAF2U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));
          reductions_DD_V3_GAMMAF2D_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));
#if defined (COMPUTEBACKWARD)
          reductions_DD_V2_GAMMAF1U_U_2pt_backward.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));

          reductions_UU_V2_GAMMAF1D_U_2pt_backward.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));
          reductions_UU_V4_GAMMAF1U_D_2pt_backward.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));


          reductions_UU_V3_GAMMAF2U_2pt_backward.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));
          reductions_DD_V3_GAMMAF2D_2pt_backward.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));
#endif

        }


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
          int tsinkMtsource = tSinks[k];
          if(tsinkMtsource >= HGC_totalL[3])
            PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);

          PLEGMA_Propagator<float> propUPpacked_to_sink;
          PLEGMA_Propagator<float> propDNpacked_to_sink;

          for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
            site sink_local;
            sink_local[0] = sourcePositions[isource][0];
            sink_local[1] = sourcePositions[isource][1];
            sink_local[2] = sourcePositions[isource][2];
            sink_local[3] = (sourcePositions[isource][DIM_T]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  sink_local[3], tsinkMtsource, i_source_parallel==0 ? true : false);
            propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  sink_local[3], tsinkMtsource, i_source_parallel==0 ? true : false);
          }
	
          PLEGMA_Vector<float> stochastic_source_packed;
	  for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
	    site sink_local;
            sink_local[0] = sourcePositions[isource][0];
            sink_local[1] = sourcePositions[isource][1];
            sink_local[2] = sourcePositions[isource][2];
            sink_local[3] = (sourcePositions[isource][DIM_T]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

            stochastic_source_packed.pack_propagator_as_sink(stochastic_source, sink_local[3], tsinkMtsource, i_source_parallel == 0 ? true : false);
	  }

          stochastic_source_packed.apply_gamma5();

          //For U(xf1,xf2)
          //B3,B5,B9,B11
          
          TIME(reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k]->V2( stochastic_source_packed,     glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, true));

          //B4,B6
          //B10,B12
          TIME(reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k]->V4( stochastic_source_packed,     glist_sink_nucleon, propUPpacked_to_sink, propDNpacked_to_sink, true));

	  if (k==0 && (dotwopoint==1)){
             PLEGMA_Vector<float> stoch_piece;
	     stoch_piece.unload();
	     stoch_piece.copy(*stochastic_sources[i_sample],HOST);
	     stoch_piece.load();

	     TIME(reductions_UU_V3_GAMMAF2U_2pt[i_sample]->V3( stoch_piece, glist_sink_meson,   propUP_SS_packed, true));
#if defined (COMPUTEBACKWARD)
             TIME(reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample]->V3( stoch_piece, glist_sink_meson,   propUP_SS_packed_backward, true));
#endif
	     stoch_piece.apply_gamma5();
             TIME(reductions_DD_V2_GAMMAF1U_U_2pt[i_sample]->V2( stoch_piece,     glist_sink_nucleon, propUP_SS_packed, propUP_SS_packed, true));


#if defined (COMPUTEBACKWARD)
             TIME(reductions_DD_V2_GAMMAF1U_U_2pt_backward[i_sample]->V2( stoch_piece,     glist_sink_nucleon, propUP_SS_packed_backward, propUP_SS_packed_backward, true));


#endif

             stoch_piece.unload();
             stoch_piece.copy(*stochastic_propagator_2pt_SS[i_sample],HOST);
             stoch_piece.load();

	     TIME(reductions_UU_V2_GAMMAF1D_U_2pt[i_sample]->V2( stoch_piece,  glist_sink_nucleon, propDN_SS_packed, propUP_SS_packed, true));

             TIME(reductions_UU_V4_GAMMAF1U_D_2pt[i_sample]->V4( stoch_piece,  glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, true));
#if defined (COMPUTEBACKWARD)
             TIME(reductions_UU_V2_GAMMAF1D_U_2pt_backward[i_sample]->V2( stoch_piece,  glist_sink_nucleon, propDN_SS_packed_backward, propUP_SS_packed_backward, true));

             TIME(reductions_UU_V4_GAMMAF1U_D_2pt_backward[i_sample]->V4( stoch_piece,  glist_sink_nucleon, propUP_SS_packed_backward, propDN_SS_packed_backward, true));
#endif

             stoch_piece.apply_gamma5();

             TIME(reductions_DD_V3_GAMMAF2D_2pt[i_sample]->V3( stoch_piece, glist_sink_meson,   propDN_SS_packed, true));

#if defined (COMPUTEBACKWARD)
             TIME(reductions_DD_V3_GAMMAF2D_2pt_backward[i_sample]->V3( stoch_piece, glist_sink_meson,   propDN_SS_packed_backward, true));

#endif


	  }


          //W5,W6,W7,W8
          //W9,W10,W11,W12,W17,W18,W19,W20
	  PLEGMA_Vector<float> stochastic_propagator_packed;
          for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
            PLEGMA_Vector<float> temporary;
            site maxinsertion_local;
            maxinsertion_local[0] = sourcePositions[isource][0];
            maxinsertion_local[1] = sourcePositions[isource][1];
            maxinsertion_local[2] = sourcePositions[isource][2];
            maxinsertion_local[3] = (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	    site sink_local;
            sink_local[0] = sourcePositions[isource][0];
            sink_local[1] = sourcePositions[isource][1];
            sink_local[2] = sourcePositions[isource][2];
            sink_local[3] = (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

            temporary.copy(*stochastic_propags_DN_SL[lookuptable_DN[sink_local[3]]*n_stochastic_samples+i_sample],HOST);
	    temporary.load();
            stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, maxinsertion_local[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
          }

          stochastic_propagator_packed.apply_gamma5();

	  
          TIME(reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k]->V3( stochastic_propagator_packed, glist_insertion,   propUP_SL_packed, true));

          //For D(xf1,xf2)

          stochastic_propagator_packed.apply_gamma5();


	  for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
            PLEGMA_Vector<float> temporary;
	    site maxinsertion_local;
            maxinsertion_local[0] = sourcePositions[isource][0];
            maxinsertion_local[1] = sourcePositions[isource][1];
            maxinsertion_local[2] = sourcePositions[isource][2];
            maxinsertion_local[3] = (sourcePositions[isource][DIM_T]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            site sink_local;
            sink_local[0] = sourcePositions[isource][0];
            sink_local[1] = sourcePositions[isource][1];
            sink_local[2] = sourcePositions[isource][2];
            sink_local[3] = (sourcePositions[isource][DIM_T]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

	    temporary.unload();
            temporary.copy(*stochastic_propags_UP_SL[lookuptable_UP[sink_local[3]]*n_stochastic_samples+i_sample],HOST);
	    temporary.load();
	  
            stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary,  maxinsertion_local[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
          }


          stochastic_propagator_packed.apply_gamma5();


          //W13,W14,W15,W16
          //W21,W22,W23,W24
          TIME(reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k]->V3( stochastic_propagator_packed, glist_insertion,   propDN_SL_packed, true));

          //B7,B8
          //B1,B2
          TIME(reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k]->V2( stochastic_source_packed, glist_sink_nucleon, propUPpacked_to_sink, propUPpacked_to_sink, true));

        } //end of for source sink separations
      } //end of for stochastic samples

//      auto &momentum_i2 =  {0,0,0};//mpi2_twopt[0];
      std::vector<int> momentum_i2= {1,0,0};
      //List of momenta corresponding to a fix value of p_i2

      vectorStoc_source_oet.stochastic_Z(nroots);
#if 0

/*      {
        PLEGMA_Vector<float> tmm(BOTH);
	tmm.copy(*stochastic_sources[0],HOST);
	tmm.load();
	vectorStoc_source_oet.copy(tmm);
      }*/

      //PLEGMA_printf("DONE stochastic factors\n");

#endif
#if 1
    /******************************************************
     *
     * Step 5: Computing OET propagators for UP DN
     *          
     *
     ******************************************************/

      {

         //Doing for +mu for the UP propagator spin dilution oet
         if(mu<0) {
           mu*=-1.;
           solver.UpdateSolver();
         }
         for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){

	   site source_local = sourcePositions[isource];
           source_local[3]=(source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	   site maxinsertion_local;
           maxinsertion_local[0] = sourcePositions[isource][0];
           maxinsertion_local[1] = sourcePositions[isource][1];
           maxinsertion_local[2] = sourcePositions[isource][2];
           maxinsertion_local[3] = (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

           site source_localTotalMsink;
           source_localTotalMsink[0] = sourcePositions[isource][0];
           source_localTotalMsink[1] = sourcePositions[isource][1];
           source_localTotalMsink[2] = sourcePositions[isource][2];
           source_localTotalMsink[3] = ((sourcePositions[isource][DIM_T]+(i_source_parallel)*max_source_sink_separations)+HGC_totalL[3])%HGC_totalL[3];


           //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];


           PLEGMA_Vector<double> vectortmp1;
           PLEGMA_Vector<double> vectortmp2;
           PLEGMA_Vector<double> vectorSave_diluted;


           {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorStoc_source_oet, source_local[DIM_T]);

	      vector1.apply_gamma_scatt( glist_source_meson[0]);
	      vector1.apply_gamma5();

              PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, source_local[DIM_T]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectortmp1.absorb(vector2, source_local[DIM_T]);

           }

           vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);

           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));

	   PLEGMA_Vector<float> vectorAuxF;
	   vectorAuxF.copy(vectortmp2);
	   stochastic_oet_prop_u_zero_mom_SS.pack_propagator_from_source_to_sink(vectorAuxF, maxinsertion_local[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)

	   stochastic_oet_prop_u_zero_mom_SS_backward.pack_propagator_from_source_to_sink(vectorAuxF, source_localTotalMsink[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#endif


         } //end of loop on parallel sources


	 //PLEGMA_printf("DONE OET zero mom up\n");

         //Doing for -mu for the DN propagator spin dilution oet
         if(mu>0) {
           mu*=-1.;
           solver.UpdateSolver();
	 }

	 for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){

           site source_local = sourcePositions[isource];
           source_local[3]=(source[DIM_T]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	   site maxinsertion_local;

           maxinsertion_local[0] = sourcePositions[isource][0];
           maxinsertion_local[1] = sourcePositions[isource][1];
           maxinsertion_local[2] = sourcePositions[isource][2];
           maxinsertion_local[3] = (source[DIM_T]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

	   site source_localTotalMsink;
           source_localTotalMsink[0] = sourcePositions[isource][0];
           source_localTotalMsink[1] = sourcePositions[isource][1];
           source_localTotalMsink[2] = sourcePositions[isource][2];
           source_localTotalMsink[3] = ((sourcePositions[isource][DIM_T]+(i_source_parallel)*max_source_sink_separations)+HGC_totalL[3])%HGC_totalL[3];

           //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

           PLEGMA_Vector<double> vectortmp1;
           PLEGMA_Vector<double> vectortmp2;
           PLEGMA_Vector<double> vectorSave_diluted;


           {  // Smearing the source

            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.absorb(vectorStoc_source_oet, source_local[3]);

	    vector1.apply_gamma_scatt( glist_source_meson[0]);
            vector1.apply_gamma5();

            PLEGMA_Gauge3D<double> smearedGauge3D;
            smearedGauge3D.absorb(smearedGauge, source_local[DIM_T]);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectortmp1.absorb(vector2, source_local[3]);
           }

	   vectortmp2.rotateToPhysicalBasis(vectortmp1,-1);
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
	   PLEGMA_Vector<float> vectorAuxF;
	   vectorAuxF.copy(vectortmp2);
	   stochastic_oet_prop_d_zero_mom_SS.pack_propagator_from_source_to_sink(vectorAuxF, maxinsertion_local[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)
           stochastic_oet_prop_d_zero_mom_SS_backward.pack_propagator_from_source_to_sink(vectorAuxF, source_localTotalMsink[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
#endif

         } //end of loop on parallel sources
      }//end of do_stochastic_oet

      
      //PLEGMA_printf("DONE OET zero mom dn\n");

#endif

#if 1
    /******************************************************
     *
     * Step 4: Computing the zero momentum oet factors
     *         i.e. V2 and V4 reductions
     *         
     *
     ******************************************************/

     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_U_DN(source_reduction, list_mpc);
     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_U_DN_2pt(source_reduction, list_mpf2_twopt);
#if defined (COMPUTEBACKWARD)
     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_U_DN_2pt_backward(source_reduction, list_mpf2_twopt);
#endif

     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_D_UP(source_reduction, list_mpc);
     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_D_UP_2pt(source_reduction, list_mpf2_twopt);

     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_U_UP_2pt(source_reduction, list_mpf2_twopt);


#if defined (COMPUTEBACKWARD)
     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_D_UP_2pt_backward(source_reduction, list_mpf2_twopt);

     PLEGMA_ScattCorrelator<float> reductionsV3_diluted_U_UP_2pt_backward(source_reduction, list_mpf2_twopt);
#endif


     std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV4_diluted_STOCHU_DN_UP;
     std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV2_diluted_STOCHU_DN_UP;
     std::vector<PLEGMA_ScattCorrelator<float>*> reductionsV2_diluted_STOCHD_UP_UP;

     PLEGMA_ScattCorrelator<float> reductionsV4_diluted_STOCHU_DN_UP_2pt(source_reduction, list_mpf1_twopt);
     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHU_DN_UP_2pt(source_reduction, list_mpf1_twopt);

     PLEGMA_ScattCorrelator<float> reductionsV4_diluted_STOCHD_UP_DN_2pt(source_reduction, list_mpf1_twopt);
     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHD_UP_DN_2pt(source_reduction, list_mpf1_twopt);

     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHD_UP_UP_2pt(source_reduction, list_mpf1_twopt);
     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHU_DN_DN_2pt(source_reduction, list_mpf1_twopt);

#if defined (COMPUTEBACKWARD)

     PLEGMA_ScattCorrelator<float> reductionsV4_diluted_STOCHU_DN_UP_2pt_backward(source_reduction, list_mpf1_twopt);
     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHU_DN_UP_2pt_backward(source_reduction, list_mpf1_twopt);

     PLEGMA_ScattCorrelator<float> reductionsV4_diluted_STOCHD_UP_DN_2pt_backward(source_reduction, list_mpf1_twopt);
     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHD_UP_DN_2pt_backward(source_reduction, list_mpf1_twopt);

     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHD_UP_UP_2pt_backward(source_reduction, list_mpf1_twopt);
     PLEGMA_ScattCorrelator<float> reductionsV2_diluted_STOCHU_DN_DN_2pt_backward(source_reduction, list_mpf1_twopt);
#endif

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
     for (int k=0; k<tSinks.size();++k){
       PLEGMA_Propagator<float> propUPpacked_to_sink;
       PLEGMA_Propagator<float> propDNpacked_to_sink;
       int tsinkMtsource = tSinks[k];
       if(tsinkMtsource >= HGC_totalL[3])
         PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
       for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
         site sink_local;
         sink_local[0] = sourcePositions[isource][0];
         sink_local[1] = sourcePositions[isource][1];
         sink_local[2] = sourcePositions[isource][2];
         sink_local[3] = (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
         propUPpacked_to_sink.pack_propagator_as_sink(propUP_SS_packed,  sink_local[3], tsinkMtsource, i_source_parallel==0 ? true : false);
         propDNpacked_to_sink.pack_propagator_as_sink(propDN_SS_packed,  sink_local[3], tsinkMtsource, i_source_parallel==0 ? true : false);
       }


       PLEGMA_Vector<float> st_oet_u_zeropacked_to_sink;
       PLEGMA_Vector<float> st_oet_d_zeropacked_to_sink;

       for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
          site sink_local;
          sink_local[0] = sourcePositions[isource][0];
          sink_local[1] = sourcePositions[isource][1];
          sink_local[2] = sourcePositions[isource][2];
          sink_local[3] = (source[DIM_T]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

          st_oet_u_zeropacked_to_sink.pack_propagator_as_sink(stochastic_oet_prop_u_zero_mom_SS,  sink_local[3], tsinkMtsource, i_source_parallel==0 ? true : false);
          st_oet_d_zeropacked_to_sink.pack_propagator_as_sink(stochastic_oet_prop_d_zero_mom_SS,  sink_local[3], tsinkMtsource, i_source_parallel==0 ? true : false);
       }
 

        TIME(reductionsV2_diluted_STOCHU_DN_UP[k]->V2( st_oet_u_zeropacked_to_sink, glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, true));
        TIME(reductionsV2_diluted_STOCHD_UP_UP[k]->V2( st_oet_d_zeropacked_to_sink, glist_sink_nucleon, propUPpacked_to_sink, propUPpacked_to_sink, true));
        TIME(reductionsV4_diluted_STOCHU_DN_UP[k]->V4( st_oet_u_zeropacked_to_sink, glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, true));

	if (k==0 && (dotwopoint==1)){

	  TIME(reductionsV2_diluted_STOCHU_DN_DN_2pt.V2( stochastic_oet_prop_u_zero_mom_SS, glist_sink_nucleon, propDN_SS_packed, propDN_SS_packed, true));
	  TIME(reductionsV2_diluted_STOCHU_DN_UP_2pt.V2( stochastic_oet_prop_u_zero_mom_SS, glist_sink_nucleon, propDN_SS_packed, propUP_SS_packed, true));

          TIME(reductionsV4_diluted_STOCHD_UP_DN_2pt.V4( stochastic_oet_prop_d_zero_mom_SS, glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, true));
          TIME(reductionsV2_diluted_STOCHD_UP_DN_2pt.V2( stochastic_oet_prop_d_zero_mom_SS, glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, true));

          TIME(reductionsV2_diluted_STOCHD_UP_UP_2pt.V2( stochastic_oet_prop_d_zero_mom_SS, glist_sink_nucleon, propUP_SS_packed, propUP_SS_packed, true));
          TIME(reductionsV4_diluted_STOCHU_DN_UP_2pt.V4( stochastic_oet_prop_u_zero_mom_SS, glist_sink_nucleon, propDN_SS_packed, propUP_SS_packed, true));

#if defined (COMPUTEBACKWARD)
	  TIME(reductionsV2_diluted_STOCHU_DN_DN_2pt_backward.V2( stochastic_oet_prop_u_zero_mom_SS_backward, glist_sink_nucleon, propDN_SS_packed_backward, propDN_SS_packed_backward, true));
          TIME(reductionsV2_diluted_STOCHU_DN_UP_2pt_backward.V2( stochastic_oet_prop_u_zero_mom_SS_backward, glist_sink_nucleon, propDN_SS_packed_backward, propUP_SS_packed_backward, true));

          TIME(reductionsV4_diluted_STOCHD_UP_DN_2pt_backward.V4( stochastic_oet_prop_d_zero_mom_SS_backward, glist_sink_nucleon, propUP_SS_packed_backward, propDN_SS_packed_backward, true));
          TIME(reductionsV2_diluted_STOCHD_UP_DN_2pt_backward.V2( stochastic_oet_prop_d_zero_mom_SS_backward, glist_sink_nucleon, propUP_SS_packed_backward, propDN_SS_packed_backward, true));

          TIME(reductionsV2_diluted_STOCHD_UP_UP_2pt_backward.V2( stochastic_oet_prop_d_zero_mom_SS_backward, glist_sink_nucleon, propUP_SS_packed_backward, propUP_SS_packed_backward, true));
          TIME(reductionsV4_diluted_STOCHU_DN_UP_2pt_backward.V4( stochastic_oet_prop_u_zero_mom_SS_backward, glist_sink_nucleon, propDN_SS_packed_backward, propUP_SS_packed_backward, true));

#endif
        }
        
      }

#endif

      //P diagram
      PLEGMA_ScattCorrelator<float> corrP0UP(source, list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrP0DN(source, list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrPPUP(source, list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrPPDN(source, list_mpi2_twopt);

      corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
      corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
      corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
      corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");

#if defined (COMPUTEBACKWARD)

      PLEGMA_ScattCorrelator<float> corrP0UP_backward(source, list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrP0DN_backward(source, list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrPPUP_backward(source, list_mpi2_twopt);
      PLEGMA_ScattCorrelator<float> corrPPDN_backward(source, list_mpi2_twopt);

      corrP0UP_backward.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
      corrP0DN_backward.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
      corrPPUP_backward.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
      corrPPDN_backward.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");

#endif




    /******************************************************
     *
     * Step 4: Computing the sequential for the UU part
     *         proton pizero up
     *        
     *
     ******************************************************/

      //We first have a loop over all unique the source meson momentum p_i2
      for (int i_mpi2=0; i_mpi2<mpi2_threept.size(); ++i_mpi2){

        auto &momentum_i2 =  mpi2_threept[i_mpi2];
        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList_threept.extract(momentum_i2, 0);
        
	momList filtered_sourcemomentumList_2pt = sourcemomentumList_twopt.extract(momentum_i2, 0);

	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList_2pt.uniq_p(3);

        std::vector<std::vector<int>> mpi2_filt  ;
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	//PLEGMA_printf("mptot_filt.size() %d\n",mptot_filt.size());
        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

#if 1

        PLEGMA_ScattCorrelator<float> reductionsV2(source_reduction, list_mpf1_threept);
        PLEGMA_ScattCorrelator<float> reductionsV2_2pt(source_reduction, list_mpf1_twopt);
		
        PLEGMA_ScattCorrelator<float> corrM(     source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrD1if12(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrD1if34(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrD1if56(source, filtered_sourcemomentumList_2pt);

#if defined (COMPUTEBACKWARD)
        PLEGMA_ScattCorrelator<float> corrM_backward(     source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrD1if12_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrD1if34_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrD1if56_backward(source, filtered_sourcemomentumList_2pt);
#endif
	PLEGMA_ScattCorrelator<float> reductionsT1(source_reduction, mptot_filt);
        PLEGMA_ScattCorrelator<float> reductionsT2(source_reduction, mptot_filt);

        PLEGMA_ScattCorrelator<float> reductionsV3(source_reduction, list_mpc);
        PLEGMA_ScattCorrelator<float> reductionsV3_2pt(source_reduction, list_mpf2_twopt);
		 
	PLEGMA_Propagator<float> propTS_SS_packed;
#if defined (COMPUTEBACKWARD)
        PLEGMA_Propagator<float> propTS_SS_packed_backward;
#endif
        PLEGMA_Propagator<float> propTS_SL_packed;
#if 1
	//First we do the UP - UP case sequential inversion for the proton pizero
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
              vector1.absorb( vectorAuxD, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
              PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);

              vector1.mulMomentumPhases(momentum_i2,1);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorAuxD.absorb(vector2, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
	    }
            
	    vectorAuxD2.absorbTimeslice(vectorAuxD, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], false);

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

	  propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)

          propTS_SS_packed_backward.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
#endif
          propTS_SL_packed.pack_propagator_from_source_to_sink(propTS_SL, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

	}


	PLEGMA_ScattCorrelator<float> corrTproton_protonpizero1(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero2(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero3(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero4(source, list_mpi2ptot);
#if defined (COMPUTEBACKWARD)
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero1_backward(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero2_backward(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero3_backward(source, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrTproton_protonpizero4_backward(source, list_mpi2ptot);
#endif

	//std::vector<> mm=list_mpi2ptot.to_string();

        corrTproton_protonpizero1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq21");

        corrTproton_protonpizero2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq22");

        corrTproton_protonpizero3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq23");

        corrTproton_protonpizero4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq24");

#if defined (COMPUTEBACKWARD)
	corrTproton_protonpizero1_backward.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq21");

        corrTproton_protonpizero2_backward.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq22");

        corrTproton_protonpizero3_backward.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq23");

        corrTproton_protonpizero4_backward.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq24");
#endif

	if (dotwopoint==1){


          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed, propDN_SS_packed, propUP_SS_packed));
          TIME(corrTproton_protonpizero1.convertTreductiontoDiagram( reductionsT1, 0, false, true, true ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed, propDN_SS_packed, propUP_SS_packed));
          TIME(corrTproton_protonpizero2.convertTreductiontoDiagram( reductionsT2, 0, false, true, true ));

#if defined (COMPUTEBACKWARD)

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed_backward, propDN_SS_packed_backward, propUP_SS_packed_backward));
          TIME(corrTproton_protonpizero1_backward.convertTreductiontoDiagram( reductionsT1, 0, false, true, true ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS_SS_packed_backward, propDN_SS_packed_backward, propUP_SS_packed_backward));
          TIME(corrTproton_protonpizero2_backward.convertTreductiontoDiagram( reductionsT2, 0, false, true, true ));

#endif
          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
          TIME(corrTproton_protonpizero3.convertTreductiontoDiagram( reductionsT1, 0, false, true, true ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
          TIME(corrTproton_protonpizero4.convertTreductiontoDiagram( reductionsT2, 0, false, true, true));

#if defined (COMPUTEBACKWARD)

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed_backward, propDN_SS_packed_backward, propTS_SS_packed_backward));
          TIME(corrTproton_protonpizero3_backward.convertTreductiontoDiagram( reductionsT1, 0, false, true, true ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed_backward, propDN_SS_packed_backward, propTS_SS_packed_backward));
          TIME(corrTproton_protonpizero4_backward.convertTreductiontoDiagram( reductionsT2, 0, false, true, true));

#endif

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

	  TIME(produceOutput_2pt_packed(corrTproton_protonpizero1, outfilename, "T", parallel_sources, attract_lookup_table));
	  TIME(produceOutput_2pt_packed(corrTproton_protonpizero2, outfilename, "T", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero3, outfilename, "T", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero4, outfilename, "T", parallel_sources, attract_lookup_table));

#if defined (COMPUTEBACKWARD)
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T_backward";

          TIME(produceOutput_2pt_packed(corrTproton_protonpizero1_backward, outfilename, "T", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero2_backward, outfilename, "T", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero3_backward, outfilename, "T", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero4_backward, outfilename, "T", parallel_sources, attract_lookup_table_backward));

	}
#endif


	//creating factors

	PLEGMA_Vector<float> stochastic_propagator_packed;

        PLEGMA_ScattCorrelator<float> corrB3_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB4_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB5_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB6_2pt(source, filtered_sourcemomentumList_2pt);


        PLEGMA_ScattCorrelator<float> corrW5_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW6_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW7_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW8_2pt(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrW13_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW14_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW15_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW16_2pt(source, filtered_sourcemomentumList_2pt);


#if defined (COMPUTEBACKWARD)

        PLEGMA_ScattCorrelator<float> corrB3_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB4_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB5_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB6_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrW5_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW6_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW7_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW8_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrW13_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW14_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW15_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW16_2pt_backward(source, filtered_sourcemomentumList_2pt);


#endif

        corrB3_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B3");
        corrB4_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B4");
        corrB5_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B5");
        corrB6_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B6");

        corrW5_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W5");
        corrW6_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W6");
        corrW7_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W7");
        corrW8_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W8");

        corrW13_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W13");
        corrW14_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W14");
        corrW15_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W15");
        corrW16_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W16");


#if defined (COMPUTEBACKWARD)

        corrB3_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B3");
        corrB4_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B4");
        corrB5_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B5");
        corrB6_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B6");

        corrW5_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W5");
        corrW6_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W6");
        corrW7_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W7");
        corrW8_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W8");

        corrW13_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W13");
        corrW14_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W14");
        corrW15_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W15");
        corrW16_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W16");

#endif
#endif


	if (dotwopoint==1){

	  for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
 
            PLEGMA_Vector<float> stochastic_piece;
            stochastic_piece.unload();
	    stochastic_piece.copy(*stochastic_sources[i_sample], HOST);
	    stochastic_piece.load();

            //V3
            TIME(reductionsV3_2pt.V3( stochastic_piece, glist_sink_meson, propTS_SS_packed, true));

	    TIME(corrB3_2pt.Recombination(reductionsV3_2pt,*reductions_UU_V2_GAMMAF1D_U_2pt[i_sample],false,0,false,0,true,false,false,false,true));
            TIME(corrB4_2pt.Recombination(reductionsV3_2pt,*reductions_UU_V4_GAMMAF1U_D_2pt[i_sample],false,2,false,0,true,true,false,false,true));
            TIME(corrB5_2pt.Recombination(reductionsV3_2pt, *reductions_UU_V2_GAMMAF1D_U_2pt[i_sample],true,1,false,0,true,false,false,false,true));
            TIME(corrB6_2pt.Recombination(reductionsV3_2pt, *reductions_UU_V4_GAMMAF1U_D_2pt[i_sample],true,0,false,0,false,true,false,false,true));
#if defined (COMPUTEBACKWARD)

            TIME(reductionsV3_2pt.V3( stochastic_piece, glist_sink_meson, propTS_SS_packed_backward, true));

            TIME(corrB3_2pt_backward.Recombination(reductionsV3_2pt,*reductions_UU_V2_GAMMAF1D_U_2pt_backward[i_sample],false,0,false,0,true,false,false,false,true));
            TIME(corrB4_2pt_backward.Recombination(reductionsV3_2pt,*reductions_UU_V4_GAMMAF1U_D_2pt_backward[i_sample],false,2,false,0,true,true,false,false,true));           
            TIME(corrB5_2pt_backward.Recombination(reductionsV3_2pt,*reductions_UU_V2_GAMMAF1D_U_2pt_backward[i_sample],true,1,false,0,true,false,false,false,true));
            TIME(corrB6_2pt_backward.Recombination(reductionsV3_2pt,*reductions_UU_V4_GAMMAF1U_D_2pt_backward[i_sample],true,0,false,0,false,true,false,false,true));
            
#endif


	    stochastic_piece.apply_gamma5();
	    TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propTS_SS_packed, propUP_SS_packed, true));//checked
            TIME(corrW13_2pt.Recombination(*reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, false, 0, false, 0, false, true, false, false, true));

	    TIME(corrW15_2pt.Recombination( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, false, 2, true, 0, false, true, false, false, true));
	    

#if defined (COMPUTEBACKWARD)

            TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propTS_SS_packed_backward, propUP_SS_packed_backward, true));//checked

            TIME(corrW13_2pt_backward.Recombination( *reductions_DD_V3_GAMMAF2D_2pt_backward[i_sample], reductionsV2_2pt, false, 0, false, 0, false, true, false, false, true));
	    
	    TIME(corrW15_2pt_backward.Recombination( *reductions_DD_V3_GAMMAF2D_2pt_backward[i_sample], reductionsV2_2pt,  false, 2, true, 0, false, true, false, false, true));

#endif


            TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, true));//checked

            TIME(corrW14_2pt.Recombination( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, false, 2, true, 0, false, true, false, false, true));

            TIME(corrW16_2pt.Recombination( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, false, 0, false, 0, false, true, false, false, true));


#if defined (COMPUTEBACKWARD)

            TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propUP_SS_packed_backward, propTS_SS_packed_backward, true));//checked

	    TIME(corrW14_2pt_backward.Recombination( *reductions_DD_V3_GAMMAF2D_2pt_backward[i_sample], reductionsV2_2pt, false, 2, true, 0, false, true, false, false, true));

  	    TIME(corrW16_2pt_backward.Recombination( *reductions_DD_V3_GAMMAF2D_2pt_backward[i_sample], reductionsV2_2pt,  false, 0, false, 0, false, true, false, false, true));

#endif


	    stochastic_piece.unload();
            stochastic_piece.copy(*stochastic_propagator_2pt_SS[i_sample], HOST);
	    stochastic_piece.load();
	    TIME(reductionsV2_2pt.V4( stochastic_piece, glist_sink_nucleon, propDN_SS_packed, propTS_SS_packed, true));//checked

            TIME(corrW5_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, true, 0, false, 0,true, false, false, false, true));

            TIME(corrW7_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, false, 1, false, 0, true, false, false, false, true));

#if defined (COMPUTEBACKWARD)
	                
	    TIME(reductionsV2_2pt.V4( stochastic_piece, glist_sink_nucleon, propDN_SS_packed_backward, propTS_SS_packed_backward, true));//checked

	    TIME(corrW5_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, true, 0, false, 0,true, false, false, false, true));

            TIME(corrW7_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, false, 1, false, 0, true, false, false, false, true));

#endif

	    TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propDN_SS_packed, propTS_SS_packed, true));//checked

            TIME(corrW6_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, true, 1, false, 0, true, false, false, false, true));

            TIME(corrW8_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, false, 0, false, 0, true, false, false, false, true));

#if defined (COMPUTEBACKWARD)
            TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propDN_SS_packed_backward, propTS_SS_packed_backward, true));//checked

            TIME(corrW6_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, true, 1, false, 0, true, false, false, false, true));

            TIME(corrW8_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, false, 0, false, 0, true, false, false, false, true));

#endif

	  } //loop over sample

	 
	}//loop over doing twopoint
 
	for (int k=0; k<tSinks.size(); ++k){
          int tsinkMtsource = tSinks[k];
          if(tsinkMtsource >= HGC_totalL[3])
            PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);            
       
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

            TIME(corrB3.Recombination(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], false,0,false,0,true,false,false,true,true));
 
            TIME(corrB4.Recombination(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], false,2,false,0,true,true,false, true,true));

	    TIME(corrB5.Recombination(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k], true,1,false,0,true,false,false,true,true));
 
            TIME(corrB6.Recombination(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k], true,0,false,0,false,true,false,true,true));

	    stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
	    stochastic_propagator_packed.load();
	    PLEGMA_Vector<float> stochastic_source_packed;
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;i_source_parallel++){
              stochastic_source_packed.pack_propagator_as_sink(stochastic_propagator_packed, (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], tSinks[k], i_source_parallel == 0 ? true : false);
            }

	    stochastic_source_packed.apply_gamma5();



            TIME(reductionsV2.V4( stochastic_source_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, true));//checked

	    TIME(corrW5.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, true, 0, false, 0,true, false, false, true, true));

	    TIME(corrW7.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, false, 1, false, 0, true, false, false, true, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propDNpacked_to_sink, propTS_SS_packed_to_sink, true));//checked

	    TIME(corrW6.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()], reductionsV2,  true, 1, false, 0, true, false, false, true, true));

	    TIME(corrW8.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()], reductionsV2, false, 0, false, 0, true, false, false, true, true));


            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, true));//checked


	    TIME(corrW13.Recombination( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, false, 0, false, 0, false, true, false, true, true));

	    TIME(corrW15.Recombination( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, false, 2, true, 0, false, true, false, true, true));

            TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propUPpacked_to_sink, propTS_SS_packed_to_sink, true));//checked

	    TIME(corrW14.Recombination( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2,  false, 2, true, 0, false, true, false, true, true));


	    TIME(corrW16.Recombination( *reductions_DD_V3_GAMMAF2D[i_sample*tSinks.size()+k], reductionsV2, false, 0, false, 0, false, true, false, true, true));

	  }//loop over stochastic samples

	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput_3pt(corrB3, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrB4, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrB5, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrB6, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput_3pt(corrW5, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));
          TIME(produceOutput_3pt(corrW6, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));
          TIME(produceOutput_3pt(corrW7, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));
          TIME(produceOutput_3pt(corrW8, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));

          TIME(produceOutput_3pt(corrW13, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));
          TIME(produceOutput_3pt(corrW14, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));
          TIME(produceOutput_3pt(corrW15, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));
          TIME(produceOutput_3pt(corrW16, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k]));

	  if (k==0 && (dotwopoint==1)){
	    outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B_2pt";
            TIME(produceOutput_2pt_packed(corrB3_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrB4_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrB5_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrB6_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));

#if defined (COMPUTEBACKWARD)

	    outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B_2pt_backward";
            TIME(produceOutput_2pt_packed(corrB3_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrB4_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrB5_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrB6_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));

#endif


            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt";
            TIME(produceOutput_2pt_packed(corrW5_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW6_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW7_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW8_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));

            TIME(produceOutput_2pt_packed(corrW13_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW14_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW15_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW16_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));


#if defined (COMPUTEBACKWARD)
	    outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt_backward";
            TIME(produceOutput_2pt_packed(corrW5_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW6_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW7_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW8_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));

            TIME(produceOutput_2pt_packed(corrW13_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW14_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW15_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW16_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));

#endif
	  }


        }//loop over source sink separations

#endif
//Here we perform the last sequential the DN DN case
#if 1
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
              vector1.absorb( vectorAuxD, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
              PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);

              vector1.mulMomentumPhases(momentum_i2,1);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorAuxD.absorb(vector2, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
            }

            vectorAuxD2.absorbTimeslice(vectorAuxD, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], false);

            //Perform multiplication with glist_insertion[0]
            vectorAuxD2.apply_gamma_scatt(glist_source_meson[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,-1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_insertion[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert DN for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,-1);
            //performing smearing
            vectorAuxF.copy(vectorAuxD2);
            propTS_SL.absorb(vectorAuxF, isc/3, isc%3);
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          }

          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)
          propTS_SS_packed_backward.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
#endif
          propTS_SL_packed.pack_propagator_from_source_to_sink(propTS_SL, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

        }

	if (dotwopoint==1){

          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero5(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero6(source, list_mpi2ptot);

          corrTproton_protonpizero5.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq25");

          corrTproton_protonpizero6.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq26");

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propUP_SS_packed));
          TIME(corrTproton_protonpizero5.convertTreductiontoDiagram( reductionsT1, 0, false, true, true ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, propUP_SS_packed));
          TIME(corrTproton_protonpizero6.convertTreductiontoDiagram( reductionsT2, 0, false, true, true ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

	  TIME(produceOutput_2pt_packed(corrTproton_protonpizero5, outfilename, "T", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero6, outfilename, "T", parallel_sources, attract_lookup_table));

#if defined (COMPUTEBACKWARD)

	  PLEGMA_ScattCorrelator<float> corrTproton_protonpizero5_backward(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero6_backward(source, list_mpi2ptot);

          corrTproton_protonpizero5_backward.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq25");

          corrTproton_protonpizero6_backward.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq26");

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed_backward, propTS_SS_packed_backward, propUP_SS_packed_backward));
          TIME(corrTproton_protonpizero5_backward.convertTreductiontoDiagram( reductionsT1, 0, false, true, true ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP_SS_packed_backward, propTS_SS_packed_backward, propUP_SS_packed_backward));
          TIME(corrTproton_protonpizero6_backward.convertTreductiontoDiagram( reductionsT2, 0, false, true, true ));


          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T_backward";

          TIME(produceOutput_2pt_packed(corrTproton_protonpizero5_backward, outfilename, "T", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrTproton_protonpizero6_backward, outfilename, "T", parallel_sources, attract_lookup_table_backward));
#endif
	}

	PLEGMA_ScattCorrelator<float> corrB7_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB8_2pt(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrW9_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW10_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW11_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW12_2pt(source, filtered_sourcemomentumList_2pt);



	corrB7_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B7");
        corrB8_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B8");

        corrW9_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W9");
        corrW10_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W10");
        corrW11_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W11");
        corrW12_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W12");

#if defined (COMPUTEBACKWARD)
	PLEGMA_ScattCorrelator<float> corrB7_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrB8_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrW9_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW10_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW11_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW12_2pt_backward(source, filtered_sourcemomentumList_2pt);


        corrB7_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B7");
        corrB8_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B8");

        corrW9_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W9");
        corrW10_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W10");
        corrW11_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W11");
        corrW12_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W12");

#endif



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

	    if (k==0 && (dotwopoint==1)){
	      stochastic_propagator_packed.unload();
	      stochastic_propagator_packed.copy(*stochastic_propagator_2pt_SS[i_sample],HOST);
	      stochastic_propagator_packed.load();

              TIME(reductionsV2_2pt.V2( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed, propUP_SS_packed, true));
              TIME( corrW9_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt,  false, 0, false, 0, true, false, false, false, true));

	      TIME(corrW11_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, true, 1, false, 0, true, false, false, false, true));

              TIME(reductionsV2_2pt.V4( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed, propUP_SS_packed, true));

              TIME(corrW10_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt,false, 1, false, 0, true, false, false, false, true ));

              TIME(corrW12_2pt.Recombination( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, true, 0, false, 0, true, false, false, false, true));

#if defined (COMPUTEBACKWARD)
	      TIME(reductionsV2_2pt.V2( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed_backward, propUP_SS_packed_backward, true));

              TIME( corrW9_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, false, 0, false, 0, true, false, false, false, true));

	      TIME(corrW11_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt,true, 1, false, 0, true, false, false, false, true));
              
              TIME(reductionsV2_2pt.V4( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed_backward, propUP_SS_packed_backward, true));

              TIME(corrW10_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, false, 1, false, 0, true, false, false, false, true ));

	      TIME(corrW12_2pt_backward.Recombination( *reductions_UU_V3_GAMMAF2U_2pt_backward[i_sample], reductionsV2_2pt, true, 0, false, 0, true, false, false, false, true));

#endif

	      stochastic_propagator_packed.apply_gamma5();

              TIME(reductionsV3_2pt.V3( stochastic_propagator_packed, glist_sink_meson,   propTS_SS_packed, true));

	      TIME(corrB7_2pt.Recombination(reductionsV3_2pt, *reductions_DD_V2_GAMMAF1U_U_2pt[i_sample], false, 2, true, 0, false, true, false, false, true));


	      TIME(corrB8_2pt.Recombination(reductionsV3_2pt, *reductions_DD_V2_GAMMAF1U_U_2pt[i_sample], false, 0, false, 0, false, true, false, false, true));


#if defined (COMPUTEBACKWARD)

	      TIME(reductionsV3_2pt.V3( stochastic_propagator_packed, glist_sink_meson,   propTS_SS_packed_backward, true));

	      TIME(corrB7_2pt_backward.Recombination(reductionsV3_2pt, *reductions_DD_V2_GAMMAF1U_U_2pt_backward[i_sample], false, 2, true, 0, false, true, false, false, true));


	      TIME(corrB8_2pt_backward.Recombination(reductionsV3_2pt, *reductions_DD_V2_GAMMAF1U_U_2pt_backward[i_sample], false, 0, false, 0, false, true, false, false, true));


#endif

	    }
            for (int i_source_parallel=0; i_source_parallel<HGC_totalL[3]/max_source_sink_separations;++i_source_parallel){
              PLEGMA_Vector<float> temporary;
              temporary.copy(*stochastic_propags_UP_SL[lookuptable_UP[(source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]]*n_stochastic_samples+i_sample],HOST);
	      temporary.load();

              stochastic_propagator_packed.pack_propagator_from_source_to_sink(temporary, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
            }

            stochastic_propagator_packed.apply_gamma5();

	    TIME(reductionsV3.V3( stochastic_propagator_packed, glist_insertion,   propTS_SL_packed, true));

            TIME(corrB7.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k], 0, 7, true, false, true));
            TIME(corrB8.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i_sample*tSinks.size()+k], 0, 8, true, false, true));

            stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
            stochastic_propagator_packed.load();

	    PLEGMA_Vector<float> stochastic_source_packed;
            for (int i_source_parallel=0; i_source_parallel<parallel_sources;i_source_parallel++){

              stochastic_source_packed.pack_propagator_as_sink(stochastic_propagator_packed, (source[3]+tSinks[k]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], tSinks[k], i_source_parallel == 0 ? true : false);
            }

            stochastic_source_packed.apply_gamma5();

	    TIME(reductionsV2.V2( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, true));

            TIME( corrW9.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2,false, 0, false, 0, true, false, false, true, true));

            TIME(corrW11.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, true, 1, false, 0, true, false, false, true, true));

            TIME(reductionsV2.V4( stochastic_source_packed, glist_sink_nucleon, propTS_SS_packed_to_sink, propUPpacked_to_sink, true));

	    
            TIME(corrW10.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, false, 1, false, 0, true, false, false, true, true ));

            TIME(corrW12.Recombination( *reductions_UU_V3_GAMMAF2U[i_sample*tSinks.size()+k], reductionsV2, true, 0, false, 0, true, false, false, true, true));

	  } //loop over stochastic samples

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput_3pt(corrB7, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrB8, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput_3pt(corrW9 , outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));//because of V4
          TIME(produceOutput_3pt(corrW10, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));//because of V4
          TIME(produceOutput_3pt(corrW11, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrW12, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table, tSinks[k] ));

	  if (k==0 && (dotwopoint==1)){
            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B_2pt";
            TIME(produceOutput_2pt_packed(corrB7_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrB8_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));

            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt";
            TIME(produceOutput_2pt_packed(corrW9_2pt , outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));//because of V4
            TIME(produceOutput_2pt_packed(corrW10_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW11_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW12_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));


#if defined (COMPUTEBACKWARD)

	    outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B_2pt_backward";
            TIME(produceOutput_2pt_packed(corrB7_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrB8_2pt_backward, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));

            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt_backward";
            TIME(produceOutput_2pt_packed(corrW9_2pt_backward , outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));//because of V4
            TIME(produceOutput_2pt_packed(corrW10_2pt_backward, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW11_2pt_backward, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));
            TIME(produceOutput_2pt_packed(corrW12_2pt_backward, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table_backward));

#endif

	  }


	} //loop over tSinks
#endif

	{ //Finite momentum oet
          PLEGMA_Vector<double> vectortmp1;
          PLEGMA_Vector<double> vectortmp2;
          PLEGMA_Vector<double> vectorSource_finite_mom;

          //Doing for +mu for the UP propagator spin dilution oet
          if(mu<0) {
            mu*=-1.;
            solver.UpdateSolver();
          }


	  for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
            site source_local = sourcePositions[isource];
            source_local[3]=(source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            //int sink_local=(sourcePositions[isource][3]+(l+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

            //Multiplying by the appropriate momentum phase

            vectorSource_finite_mom.copy(vectorStoc_source_oet);

            std::vector<int> tmp_4Dmom= momentum_i2 ;
            tmp_4Dmom.push_back(0);
            vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);
	    vectortmp1.zero_device();

            {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorSource_finite_mom, source_local[3]);
	      PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, source_local[DIM_T]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectortmp1.absorb(vector2, source_local[3],true);

            }


            //Transforming to physical base for the UP quark
            vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);

            //Doing the inversion
            TIME(solver.solve(vectortmp2, vectortmp2));

            //Rotate back immediately to the physical basis
            vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
            PLEGMA_Vector<float> vectorAuxF;

	    vectorAuxF.copy(vectortmp1);
	    stochastic_oet_prop_u_fini_mom_SL.pack_propagator_from_source_to_sink(vectorAuxF,(source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

	    TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss ));

            vectorAuxF.copy(vectortmp2);
            stochastic_oet_prop_u_fini_mom_SS.pack_propagator_from_source_to_sink(vectorAuxF,(source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)

	    stochastic_oet_prop_u_fini_mom_SS_backward.pack_propagator_from_source_to_sink(vectorAuxF,(source[3]+(i_source_parallel)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
#endif

	  }


          //Doing for +mu for the DN propagator spin dilution oet
          if(mu>0) {
            mu*=-1.;
            solver.UpdateSolver();
          }


          for (int i_source_parallel=0; i_source_parallel<parallel_sources;++i_source_parallel){
            site source_local = sourcePositions[isource];
            source_local[3]=(source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
            vectortmp1.zero_device();
            vectorSource_finite_mom.copy(vectorStoc_source_oet);

            std::vector<int> tmp_4Dmom= momentum_i2 ;
            tmp_4Dmom.push_back(0);
            vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);

            {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorSource_finite_mom, source_local[3]);
              PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, source_local[DIM_T]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectortmp1.absorb(vector2, source_local[3]);

            }

            //Transforming to physical base for the UP quark
            vectortmp2.rotateToPhysicalBasis(vectortmp1,-1);

            //Doing the inversion
            TIME(solver.solve(vectortmp2, vectortmp2));

            //Rotate back immediately to the physical basis
            vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxF.copy(vectortmp1);
	    stochastic_oet_prop_d_fini_mom_SL.pack_propagator_from_source_to_sink(vectorAuxF,(source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3],max_source_sink_separations, i_source_parallel == 0 ? true : false);

	    TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss ));

            vectorAuxF.copy(vectortmp2);
            stochastic_oet_prop_d_fini_mom_SS.pack_propagator_from_source_to_sink(vectorAuxF,(source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

#if defined (COMPUTEBACKWARD)
            stochastic_oet_prop_d_fini_mom_SS_backward.pack_propagator_from_source_to_sink(vectorAuxF,(source[3]+(i_source_parallel)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
#endif


          } // end of parallel sources

        }//end of finite momentum oet
#if 1
        { 
          PLEGMA_Vector<float> st_oet_d_fini;
	  st_oet_d_fini.copy(stochastic_oet_prop_d_fini_mom_SL);

          st_oet_d_fini.apply_gamma5();

          TIME(reductionsV3_diluted_D_UP.V3( st_oet_d_fini, glist_insertion, propUP_SL_packed, true));

	  if (dotwopoint==1){
            PLEGMA_Vector<float> st_oet_u_fini;

	    st_oet_u_fini.copy(stochastic_oet_prop_u_fini_mom_SS);
            st_oet_u_fini.apply_gamma5();

	    TIME(reductionsV3_diluted_U_UP_2pt.V3( st_oet_u_fini, glist_sink_meson, propUP_SS_packed, true));

            PLEGMA_Vector<float> st_oet_d_fini;

            st_oet_d_fini.copy(stochastic_oet_prop_d_fini_mom_SS);
            st_oet_d_fini.apply_gamma5();

            TIME(reductionsV3_diluted_D_UP_2pt.V3( st_oet_d_fini, glist_sink_meson, propUP_SS_packed, true));

#if defined (COMPUTEBACKWARD)

	    st_oet_u_fini.copy(stochastic_oet_prop_u_fini_mom_SS_backward);
            st_oet_u_fini.apply_gamma5();

            TIME(reductionsV3_diluted_U_UP_2pt_backward.V3( st_oet_u_fini, glist_sink_meson, propUP_SS_packed_backward, true));

            st_oet_d_fini.copy(stochastic_oet_prop_d_fini_mom_SS_backward);
            st_oet_d_fini.apply_gamma5();

            TIME(reductionsV3_diluted_D_UP_2pt_backward.V3( st_oet_d_fini, glist_sink_meson, propUP_SS_packed_backward, true));
#endif


	  }


        }
#endif
#if 1
	//for (int i=0; i< 4; ++i){
	{
          PLEGMA_Vector<float> st_oet_u_fini;
          st_oet_u_fini.copy(stochastic_oet_prop_u_fini_mom_SL);

	  st_oet_u_fini.apply_gamma5();

          TIME(reductionsV3_diluted_U_DN.V3( st_oet_u_fini, glist_insertion, propDN_SL_packed, true));
	}
#if 1 
	if (dotwopoint==1){
          PLEGMA_Vector<float> st_oet_u_fini;
          st_oet_u_fini.copy(stochastic_oet_prop_u_fini_mom_SS);

          st_oet_u_fini.apply_gamma5();

          TIME(reductionsV3_diluted_U_DN_2pt.V3( st_oet_u_fini, glist_sink_meson, propDN_SS_packed, true));

#if defined (COMPUTEBACKWARD)

	  st_oet_u_fini.copy(stochastic_oet_prop_u_fini_mom_SS_backward);

          st_oet_u_fini.apply_gamma5();

          TIME(reductionsV3_diluted_U_DN_2pt_backward.V3( st_oet_u_fini, glist_sink_meson, propDN_SS_packed_backward, true));
#endif
        }
#endif


	PLEGMA_ScattCorrelator<float> corrZ5_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ6_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ7_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ8_2pt(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrZ9_2pt(source,  filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ10_2pt(source, filtered_sourcemomentumList_2pt);



	corrZ5_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z5");
        corrZ6_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z6");

        corrZ7_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z7");
        corrZ8_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z8");

        corrZ9_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z9");
        corrZ10_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z10");

#if defined (COMPUTEBACKWARD)

        PLEGMA_ScattCorrelator<float> corrZ5_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ6_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ7_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ8_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrZ9_2pt_backward(source,  filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ10_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrZ11_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ12_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ13_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ14_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrZ15_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ16_2pt_backward(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrZ17_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ18_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ19_2pt_backward(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrZ20_2pt_backward(source, filtered_sourcemomentumList_2pt);


        corrZ5_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z5");
        corrZ6_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z6");

        corrZ7_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z7");
        corrZ8_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z8");

        corrZ9_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z9");
        corrZ10_2pt_backward.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z10");

#endif

      if (dotwopoint==1){

#if defined (COMPUTEBACKWARD)
          TIME(corrZ5_2pt_backward.Recombination( reductionsV3_diluted_D_UP_2pt_backward, reductionsV2_diluted_STOCHU_DN_UP_2pt_backward, false, 0, false, 0, true, false, false, false, true ));


	  TIME(corrZ7_2pt_backward.Recombination( reductionsV3_diluted_D_UP_2pt_backward, reductionsV2_diluted_STOCHU_DN_UP_2pt_backward, true, 1, false, 0, true, false, false, false, true ));


          TIME(corrZ6_2pt_backward.Recombination( reductionsV3_diluted_D_UP_2pt_backward, reductionsV4_diluted_STOCHU_DN_UP_2pt_backward, false, 1, false, 0, true, false, false, false, true ));


          TIME(corrZ8_2pt_backward.Recombination( reductionsV3_diluted_D_UP_2pt_backward, reductionsV4_diluted_STOCHU_DN_UP_2pt_backward,  true, 0, false, 0, true, false, false, false , true ));


          TIME( corrZ9_2pt_backward.Recombination( reductionsV3_diluted_U_DN_2pt_backward, reductionsV2_diluted_STOCHD_UP_UP_2pt_backward, false, 2, true, 0, false, true, false, false, true));


          TIME(corrZ10_2pt_backward.Recombination( reductionsV3_diluted_U_DN_2pt_backward, reductionsV2_diluted_STOCHD_UP_UP_2pt_backward,false, 0, false, 0, false, true, false, false, true ));

#endif


	  TIME(corrZ5_2pt.Recombination( reductionsV3_diluted_D_UP_2pt, reductionsV2_diluted_STOCHU_DN_UP_2pt, false, 0, false, 0, true, false, false, false, true ));

	  TIME(corrZ7_2pt.Recombination( reductionsV3_diluted_D_UP_2pt, reductionsV2_diluted_STOCHU_DN_UP_2pt, true, 1, false, 0, true, false, false, false, true ));

	  TIME(corrZ6_2pt.Recombination( reductionsV3_diluted_D_UP_2pt, reductionsV4_diluted_STOCHU_DN_UP_2pt, false, 1, false, 0, true, false, false, false, true ));

          TIME(corrZ8_2pt.Recombination( reductionsV3_diluted_D_UP_2pt, reductionsV4_diluted_STOCHU_DN_UP_2pt, true, 0, false, 0, true, false, false, false , true ));
	  
          TIME( corrZ9_2pt.Recombination( reductionsV3_diluted_U_DN_2pt, reductionsV2_diluted_STOCHD_UP_UP_2pt, false, 2, true, 0, false, true, false, false, true));

	  TIME(corrZ10_2pt.Recombination( reductionsV3_diluted_U_DN_2pt, reductionsV2_diluted_STOCHD_UP_UP_2pt,false, 0, false, 0, false, true, false, false, true ));


#endif

	  TIME(corrPPUP.P_diagrams( stochastic_oet_prop_u_zero_mom_SS, stochastic_oet_prop_u_fini_mom_SS, i_mpi2));
          TIME(corrPPDN.P_diagrams( stochastic_oet_prop_d_zero_mom_SS, stochastic_oet_prop_d_fini_mom_SS, i_mpi2));
          TIME(corrP0UP.P_diagrams( stochastic_oet_prop_d_zero_mom_SS, stochastic_oet_prop_u_fini_mom_SS, i_mpi2));
          TIME(corrP0DN.P_diagrams( stochastic_oet_prop_u_zero_mom_SS, stochastic_oet_prop_d_fini_mom_SS, i_mpi2));

#if defined (COMPUTEBACKWARD)

          TIME(corrPPUP_backward.P_diagrams( stochastic_oet_prop_u_zero_mom_SS_backward, stochastic_oet_prop_u_fini_mom_SS_backward, i_mpi2));
          TIME(corrPPDN_backward.P_diagrams( stochastic_oet_prop_d_zero_mom_SS_backward, stochastic_oet_prop_d_fini_mom_SS_backward, i_mpi2));
          TIME(corrP0UP_backward.P_diagrams( stochastic_oet_prop_d_zero_mom_SS_backward, stochastic_oet_prop_u_fini_mom_SS_backward, i_mpi2));
          TIME(corrP0DN_backward.P_diagrams( stochastic_oet_prop_u_zero_mom_SS_backward, stochastic_oet_prop_d_fini_mom_SS_backward, i_mpi2));

#endif

#if 1

	  outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z_2pt";


          TIME(produceOutput_2pt_packed(corrZ5_2pt, outfilename, "4pt", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrZ6_2pt, outfilename, "4pt", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrZ7_2pt, outfilename, "4pt", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrZ8_2pt, outfilename, "4pt", parallel_sources, attract_lookup_table));

          TIME(produceOutput_2pt_packed(corrZ9_2pt, outfilename, "4pt", parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrZ10_2pt, outfilename,"4pt", parallel_sources, attract_lookup_table));


#if defined (COMPUTEBACKWARD)

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z_2pt_backward";

	  TIME(produceOutput_2pt_packed(corrZ5_2pt_backward, outfilename, "4pt", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrZ6_2pt_backward, outfilename, "4pt", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrZ7_2pt_backward, outfilename, "4pt", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrZ8_2pt_backward, outfilename, "4pt", parallel_sources, attract_lookup_table_backward));

          TIME(produceOutput_2pt_packed(corrZ9_2pt_backward, outfilename, "4pt", parallel_sources, attract_lookup_table_backward));
          TIME(produceOutput_2pt_packed(corrZ10_2pt_backward, outfilename,"4pt", parallel_sources, attract_lookup_table_backward));

#endif

        }
#endif

#if 1
	for (int k=0; k<tSinks.size();++k){

          PLEGMA_ScattCorrelator<float> corrZ5(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ6(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ7(source, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ8(source, filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrZ9(source,  filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrZ10(source, filtered_sourcemomentumList);

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

	  free(ssource);

	  TIME(corrZ5.Recombination( reductionsV3_diluted_D_UP, *reductionsV2_diluted_STOCHU_DN_UP[k], false, 0, false, 0, true, false, false, true, true ));
          TIME(corrZ7.Recombination( reductionsV3_diluted_D_UP, *reductionsV2_diluted_STOCHU_DN_UP[k], true, 1, false, 0, true, false, false, true, true ));
	  TIME(corrZ6.Recombination( reductionsV3_diluted_D_UP, *reductionsV4_diluted_STOCHU_DN_UP[k], false, 1, false, 0, true, false, false, true, true ));
	  TIME(corrZ8.Recombination( reductionsV3_diluted_D_UP, *reductionsV4_diluted_STOCHU_DN_UP[k], true, 0, false, 0, true, false, false, true , true ));
          TIME(corrZ9.Recombination(  reductionsV3_diluted_U_DN, *reductionsV2_diluted_STOCHD_UP_UP[k],false, 2, true, 0, false, true, false, true, true));

	  TIME(corrZ10.Recombination( reductionsV3_diluted_U_DN, *reductionsV2_diluted_STOCHD_UP_UP[k],false, 0, false, 0, false, true, false, true, true ));


          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

	  TIME(produceOutput_3pt(corrZ5, outfilename, "4pt", parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrZ6, outfilename, "4pt", parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrZ7, outfilename, "4pt", parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrZ8, outfilename, "4pt", parallel_sources, attract_lookup_table, tSinks[k] ));

          TIME(produceOutput_3pt(corrZ9, outfilename, "4pt", parallel_sources, attract_lookup_table, tSinks[k] ));
          TIME(produceOutput_3pt(corrZ10, outfilename,"4pt", parallel_sources, attract_lookup_table, tSinks[k] ));


	}//end for loop source sink separations
#endif

      }//loop over mpi2

      if (dotwopoint==1){

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_2pt";
        TIME(corrP0UP.apply_sign("P"));
        TIME(corrP0UP.writeHDF5( outfilename ));
        TIME(corrP0DN.apply_sign("P"));
        TIME(corrP0DN.writeHDF5( outfilename ));
        TIME(corrPPUP.apply_sign("P"));
        TIME(corrPPUP.writeHDF5( outfilename ));
        TIME(corrPPDN.apply_sign("P"));
        TIME(corrPPDN.writeHDF5( outfilename ));

#if defined (COMPUTEBACKWARD)

	outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_2pt_backward";
        TIME(corrP0UP_backward.apply_sign("P"));
        TIME(corrP0UP_backward.writeHDF5( outfilename ));
        TIME(corrP0DN_backward.apply_sign("P"));
        TIME(corrP0DN_backward.writeHDF5( outfilename ));
        TIME(corrPPUP_backward.apply_sign("P"));
        TIME(corrPPUP_backward.writeHDF5( outfilename ));
        TIME(corrPPDN_backward.apply_sign("P"));
        TIME(corrPPDN_backward.writeHDF5( outfilename ));

#endif


      }


#if 1
      for (int k=0; k<tSinks.size();++k){
        reductionsV4_diluted_STOCHU_DN_UP.pop_back();
        reductionsV2_diluted_STOCHU_DN_UP.pop_back();
        reductionsV2_diluted_STOCHD_UP_UP.pop_back();
      }
      //}


      for(int i_samples=0; i_samples< n_stochastic_samples; ++i_samples) {
	if (dotwopoint==1){
          reductions_DD_V2_GAMMAF1U_U_2pt.pop_back();

          reductions_UU_V4_GAMMAF1U_D_2pt.pop_back();
	  reductions_UU_V2_GAMMAF1D_U_2pt.pop_back();

          reductions_UU_V3_GAMMAF2U_2pt.pop_back();
          reductions_DD_V3_GAMMAF2D_2pt.pop_back();

#if defined (COMPUTEBACKWARD)

	  reductions_DD_V2_GAMMAF1U_U_2pt_backward.pop_back();

          reductions_UU_V4_GAMMAF1U_D_2pt_backward.pop_back();
          reductions_UU_V2_GAMMAF1D_U_2pt_backward.pop_back();

          reductions_UU_V3_GAMMAF2U_2pt_backward.pop_back();
          reductions_DD_V3_GAMMAF2D_2pt_backward.pop_back();
#endif


	}


	for (int k=0; k<tSinks.size();++k){

          reductions_UU_V2_GAMMAF1D_U.pop_back();
          reductions_UU_V4_GAMMAF1U_D.pop_back();
          reductions_UU_V3_GAMMAF2U.pop_back();

          reductions_DD_V2_GAMMAF1U_U.pop_back();
          reductions_DD_V3_GAMMAF2D.pop_back();

	}

      }
#endif

    }//loop over source position

    free(attract_lookup_table);
    free(attract_lookup_table_backward);
#if 1
    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      if (dotwopoint==1){
        stochastic_propagator_2pt_SS.pop_back();
      }
    }

    for (int i=0; i<countindex;++i){
      stochastic_propags_UP_SL.pop_back();
      stochastic_propags_DN_SL.pop_back();
    }
#endif
  }//loop in finalize
  finalize();
  return 0;
}
