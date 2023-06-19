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

  std::vector<GAMMAS_SCATT> glist_source_meson={G_5};
  std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};

  std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};//,S12,S13,S23,S41,S42,S43};

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

  /*
  FILE *fid;
  LimeWriter *limewriter = (LimeWriter*)NULL;
  //if(unloadFromDev) unload();
  if(comm_rank() == 0){
    fid=fopen("stochastic_source0.lime","a");
    if(fid==NULL) PLEGMA_error("Error opening file for writing: stochastic_source0.lime\n");
    else printf("Opening was fine\n");
    limewriter = limeCreateWriter(fid);
    if(limewriter==(LimeWriter*)NULL) PLEGMA_error("Could not create limeWriter");
    std::string xlf_message = getDateAndTime(); // More xlf-info can be added
    write_lime_header(limewriter,"xlf-info",xlf_message,1,1);
    std::ostringstream oss;
    oss << lime_version_header() << "<field>" << "PLEGMA_Vector" << "</field>\n" << "<precision>" << 32 << "</precision>\n";
    oss << "<dof>" << 12 << "</dof>\n";
    std::vector<std::string> xyzt = {"x","y","z","t"};
    for(int i = 0 ; i < N_DIMS; i++) oss << "<l" << xyzt[i] << ">" << HGC_totalL[i] << "</l" << xyzt[i] << ">\n";
    oss << "</ildgFormat>";
    write_lime_header(limewriter,"ildg-format",oss.str(),1,0);
  }
//  write_binary_to_lime(filename,fid,limewriter,h_elem,field_length);
  limeDestroyWriter(limewriter);

  exit(1);*/

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
    //PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
    //momList sourcemomentumList_threept(4,pathListMomenta_threept,{1,2,3});
    //PLEGMA_printf("N momenta in sourcemomentumList: %d\n",sourcemomentumList_threept.size());
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
    //if(sourcemomentumList_threept.empty())
    // PLEGMA_error("threept momentumList empty");

    std::vector<std::vector<int>> mpi2_twopt = sourcemomentumList_twopt.uniq_p(0);
    momList list_mpi2_twopt(1,{mpi2_twopt,},{0,});

    //std::vector<std::vector<int>> mpi2_threept = sourcemomentumList_threept.uniq_p(0);
    //momList list_mpi2_threept(1,{mpi2_threept,},{0,});

    //std::vector<std::vector<int>> mpf1_threept = sourcemomentumList_threept.uniq_p(1);
    //momList list_mpf1_threept(1,{mpf1_threept,},{0,});

    std::vector<std::vector<int>> mpf1_twopt = sourcemomentumList_twopt.uniq_p(1);
    momList list_mpf1_twopt(1,{mpf1_twopt,},{0,});

    //std::vector<std::vector<int>> mpc = sourcemomentumList_threept.uniq_p(3);
    //momList list_mpc(1,{mpc,},{0,});

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


    PLEGMA_Vector<double> vectorSource_stochastic;
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

    std::vector<PLEGMA_Vector<float>*> stochastic_propagator_2pt_SS;


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
    /*
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

              //Step(7) Smearing all the time slice in the propagator
              //TIME(vectorInOut.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

              //vectorAuxD1.unload();
	      //PLEGMA_printf("Save the stochastic source for sample %d\n",i);
              //std::string nstoch=std::to_string(i);
              //vectorAuxD1.writeHDF5("stochastic_propagators_DN"+nstoch+"_t"+std::to_string(timeSlice));

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
*/
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


#if 1
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

      PLEGMA_Propagator<float> propUP_SS_packed;
      PLEGMA_Propagator<float> propDN_SS_packed;

//      PLEGMA_Propagator<float> propUP_SL_packed;
//      PLEGMA_Propagator<float> propDN_SL_packed;

      for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){
	for (int sink=0; sink<max_source_sink_separations;++sink){
	  attract_lookup_table[(sourcePositions[isource][DIM_T]+max_source_sink_separations*i_source_parallel+sink)%HGC_totalL[3]]=(sourcePositions[isource][DIM_T]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];
	}
      }
      for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

	PLEGMA_Propagator<float> propUP_SS;
        PLEGMA_Propagator<float> propDN_SS;

	PLEGMA_Propagator<float> propUP_SL(NONE);
        PLEGMA_Propagator<float> propDN_SL(NONE);

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

	site source_local_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});
        source_local_reduction[3]=(sourcePositions[isource][3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3];

        
        // If twop_filename exists we hold the computation of the light props
	TIME(computePropagator(propUP_SS, propUP_SL,  mu_ud, LIGHT, nsmearGauss, source_local, false));
	TIME(computePropagator(propDN_SS, propDN_SL, -mu_ud, LIGHT, nsmearGauss, source_local, false));
	
        propUP_SS_packed.pack_propagator_from_source_to_sink(propUP_SS, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
        propDN_SS_packed.pack_propagator_from_source_to_sink(propDN_SS, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
//        propUP_SL_packed.pack_propagator_from_source_to_sink(propUP_SL, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);
//        propDN_SL_packed.pack_propagator_from_source_to_sink(propDN_SL, source_localPtSinkMtSource[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

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


//      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U_2pt;//implemented

//      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D_2pt;//implemented

//      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U_2pt;//implemented

//      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D_2pt;//implemented

//      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U_2pt;

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V4_GAMMAF1U_D_2pt;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_D_2pt;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2U_2pt;//implemented





      for(int i=0; i< n_stochastic_samples; ++i) {
	if (dotwopoint==1){
          reductions_DD_V2_GAMMAF1U_D_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));
          reductions_DD_V2_GAMMAF1U_U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));
          reductions_DD_V4_GAMMAF1U_D_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));

          reductions_UU_V2_GAMMAF1D_U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));
          reductions_UU_V4_GAMMAF1U_D_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_twopt));


	  reductions_UU_V3_GAMMAF2U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));
 	  reductions_DD_V3_GAMMAF2U_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));
          reductions_DD_V3_GAMMAF2D_2pt.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf2_twopt));

        }

/*
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
	*/
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
          
      //    TIME(reductions_UU_V2_GAMMAF1D_U[i_sample*tSinks.size()+k]->V2( stochastic_source_packed,     glist_sink_nucleon, propDNpacked_to_sink, propUPpacked_to_sink, true));

          //B4,B6
          //B10,B12
        //  TIME(reductions_UU_V4_GAMMAF1U_D[i_sample*tSinks.size()+k]->V4( stochastic_source_packed,     glist_sink_nucleon, propUPpacked_to_sink, propDNpacked_to_sink, true));

	  if (k==0 && (dotwopoint==1)){
             PLEGMA_Vector<float> stoch_piece;
	     stoch_piece.unload();
	     stoch_piece.copy(*stochastic_sources[i_sample],HOST);
	     stoch_piece.load();

	     TIME(reductions_UU_V3_GAMMAF2U_2pt[i_sample]->V3( stoch_piece, glist_sink_meson,   propUP_SS_packed, true));

	     stoch_piece.apply_gamma5();
             TIME(reductions_DD_V2_GAMMAF1U_U_2pt[i_sample]->V2( stoch_piece,     glist_sink_nucleon, propUP_SS_packed, propUP_SS_packed, true));

             TIME(reductions_DD_V2_GAMMAF1U_D_2pt[i_sample]->V2( stoch_piece,     glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, true));

             TIME(reductions_DD_V4_GAMMAF1U_D_2pt[i_sample]->V4( stoch_piece,     glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, true));


             stoch_piece.unload();
             stoch_piece.copy(*stochastic_propagator_2pt_SS[i_sample],HOST);
             stoch_piece.load();

	     TIME(reductions_UU_V2_GAMMAF1D_U_2pt[i_sample]->V2( stoch_piece,  glist_sink_nucleon, propDN_SS_packed, propUP_SS_packed, true));

             TIME(reductions_UU_V4_GAMMAF1U_D_2pt[i_sample]->V4( stoch_piece,  glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, true));


             stoch_piece.apply_gamma5();

             TIME(reductions_DD_V3_GAMMAF2D_2pt[i_sample]->V3( stoch_piece, glist_sink_meson,   propDN_SS_packed, true));

	     TIME(reductions_DD_V3_GAMMAF2U_2pt[i_sample]->V3( stoch_piece, glist_sink_meson,   propUP_SS_packed, true));


	  }
        } //end of for source sink separations
      } //end of for stochastic samples

//      auto &momentum_i2 =  {0,0,0};//mpi2_twopt[0];
      std::vector<int> momentum_i2= {0,0,0};
      //List of momenta corresponding to a fix value of p_i2
      momList filtered_sourcemomentumList_2pt_single = sourcemomentumList_twopt.extract(momentum_i2, 0);

      //proton pizero x proton pizero
      //udu ubaru (x_f) dbarubarubar
      PLEGMA_ScattCorrelator<float> corrD1ii1(source, filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii2(source, filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii3(source, filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii4(source, filtered_sourcemomentumList_2pt_single);

      //udu dbard (x_f) dbarubarubar
      PLEGMA_ScattCorrelator<float> corrD1ii9(source,  filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii10(source, filtered_sourcemomentumList_2pt_single);

      //dud dbaru (x_f) dbarubarubar
      PLEGMA_ScattCorrelator<float> corrD1ii13(source, filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii14(source, filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii15(source, filtered_sourcemomentumList_2pt_single);
      PLEGMA_ScattCorrelator<float> corrD1ii16(source, filtered_sourcemomentumList_2pt_single);


      corrD1ii1.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii1");
      corrD1ii2.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii2");
      corrD1ii3.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii3");
      corrD1ii4.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii4");

      corrD1ii9.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii9");
      corrD1ii10.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii10");

      corrD1ii13.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii13");
      corrD1ii14.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii14");
      corrD1ii15.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii15");
      corrD1ii16.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii16");



      for (int i=0; i<n_stochastic_samples; ++i){

        TIME(corrD1ii1.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_2pt[i], *reductions_UU_V2_GAMMAF1D_U_2pt[i], NULL, 0, 1, true));
        TIME(corrD1ii2.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_2pt[i], *reductions_UU_V4_GAMMAF1U_D_2pt[i], NULL, 0, 2, true));
        TIME(corrD1ii3.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_2pt[i], *reductions_UU_V2_GAMMAF1D_U_2pt[i], NULL, 0, 3, true));
        TIME(corrD1ii4.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_2pt[i], *reductions_UU_V4_GAMMAF1U_D_2pt[i], NULL, 0, 4, true));

        TIME(corrD1ii9.D1ii_diagrams( *reductions_DD_V3_GAMMAF2D_2pt[i], *reductions_DD_V2_GAMMAF1U_U_2pt[i],NULL, 0, 9, true));
        TIME(corrD1ii10.D1ii_diagrams(*reductions_DD_V3_GAMMAF2D_2pt[i], *reductions_DD_V2_GAMMAF1U_U_2pt[i],NULL, 0, 10, true));

        TIME(corrD1ii13.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_2pt[i], *reductions_DD_V4_GAMMAF1U_D_2pt[i],NULL, 0, 13, true));
        TIME(corrD1ii14.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_2pt[i], *reductions_DD_V4_GAMMAF1U_D_2pt[i],NULL, 0, 14, true));
        TIME(corrD1ii15.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_2pt[i], *reductions_DD_V2_GAMMAF1U_D_2pt[i],NULL, 0, 15, true));
        TIME(corrD1ii16.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_2pt[i], *reductions_DD_V2_GAMMAF1U_D_2pt[i],NULL, 0, 16, true));

      } //stochastic samples

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ii";

      TIME(produceOutput_2pt_packed(corrD1ii1, outfilename, "T", parallel_sources, attract_lookup_table));
      TIME(produceOutput_2pt_packed(corrD1ii2, outfilename, "T", parallel_sources, attract_lookup_table));
      TIME(produceOutput_2pt_packed(corrD1ii3, outfilename, "T", parallel_sources, attract_lookup_table));
      TIME(produceOutput_2pt_packed(corrD1ii4, outfilename, "T", parallel_sources, attract_lookup_table));

      TIME(produceOutput_2pt_packed(corrD1ii9,  outfilename, "T", parallel_sources, attract_lookup_table));
      TIME(produceOutput_2pt_packed(corrD1ii10, outfilename, "T", parallel_sources, attract_lookup_table));

      TIME(produceOutput_2pt_packed(corrD1ii13, outfilename, "T", parallel_sources, attract_lookup_table));//Because of V4
      TIME(produceOutput_2pt_packed(corrD1ii14, outfilename, "T", parallel_sources, attract_lookup_table));//Because of V4
      TIME(produceOutput_2pt_packed(corrD1ii15, outfilename, "T", parallel_sources, attract_lookup_table));
      TIME(produceOutput_2pt_packed(corrD1ii16, outfilename, "T", parallel_sources, attract_lookup_table));

#endif 


    /******************************************************
     *
     * Step 4: Computing the sequential for the UU part
     *         proton pizero up
     *        
     *
     ******************************************************/

      //We first have a loop over all unique the source meson momentum p_i2
      for (int i_mpi2=0; i_mpi2<mpi2_twopt.size(); ++i_mpi2){

        auto &momentum_i2 =  mpi2_twopt[i_mpi2];
        //List of momenta corresponding to a fix value of p_i2
        
	momList filtered_sourcemomentumList_2pt = sourcemomentumList_twopt.extract(momentum_i2, 0);

	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList_2pt.uniq_p(3);

        std::vector<std::vector<int>> mpi2_filt  ;
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	//PLEGMA_printf("mptot_filt.size() %d\n",mptot_filt.size());
        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

#if 1

        PLEGMA_ScattCorrelator<float> reductionsV2_2pt(source_reduction, list_mpf1_twopt);
		
        PLEGMA_ScattCorrelator<float> reductionsV3_2pt(source_reduction, list_mpf2_twopt);
		 
	PLEGMA_Propagator<float> propTS_SS_packed;
#if 1
	//First we do the UP - UP case sequential inversion for the proton pizero
	for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

	  PLEGMA_Propagator<float> propTS_SS;


          //we first implemenet UU
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
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          }

	  propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

	}

	//creating factors

	PLEGMA_Vector<float> stochastic_propagator_packed;

        PLEGMA_ScattCorrelator<float> corrW29_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW30_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW31_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW32_2pt(source, filtered_sourcemomentumList_2pt);


        corrW29_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W29");
        corrW30_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W30");
        corrW31_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W31");
        corrW32_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W32");




	  for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
 
            PLEGMA_Vector<float> stochastic_piece;
            stochastic_piece.unload();
	    stochastic_piece.copy(*stochastic_sources[i_sample], HOST);
	    stochastic_piece.load();

	    stochastic_piece.apply_gamma5();

	    TIME(reductionsV2_2pt.V2( stochastic_piece, glist_sink_nucleon, propTS_SS_packed, propDN_SS_packed, false));//checked

            TIME(corrW31_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 31, true, false));
            TIME(corrW32_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 32, true, false));

            TIME(reductionsV2_2pt.V4( stochastic_piece, glist_sink_nucleon, propTS_SS_packed, propDN_SS_packed, false));//checked

            TIME(corrW29_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 29, true, false));
            TIME(corrW30_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 30, true, false));

	  } //loop over sample

	  

 
	    outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt";

	    TIME(produceOutput_2pt_packed(corrW29_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW30_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW31_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW32_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));


#endif
	//We perform next U-D sequential to handle the piplus neutron case
        // ensuring mu positive
        if(mu<0) {
          mu*=-1.;
          solver.UpdateSolver();
        }
	propTS_SS_packed.zero_where(BOTH);
 

	for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

          PLEGMA_Propagator<float> propTS_SS;

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
              vector1.absorb( vectorAuxD, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
	      PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);

              vector1.mulMomentumPhases(momentum_i2,1);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorAuxD.absorb(vector2, (source[3]+i_source_parallel*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3]);
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

            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          } 
           
          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

        }


        PLEGMA_ScattCorrelator<float> corrW17_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW18_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW19_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW20_2pt(source, filtered_sourcemomentumList_2pt);

        PLEGMA_ScattCorrelator<float> corrW21_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW22_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW23_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW24_2pt(source, filtered_sourcemomentumList_2pt);


	corrW17_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W17");
        corrW18_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W18");
        corrW19_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W19");
        corrW20_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W20");

        corrW21_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W21");
        corrW22_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W22");
        corrW23_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W23");
        corrW24_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W24");


          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            PLEGMA_Vector<float> stochastic_propagator_packed;

//	      stochastic_propagator_packed.unload();
//            stochastic_propagator_packed.copy(*stochastic_sources[i_sample], HOST);
//	      stochastic_propagator_packed.load();
              stochastic_propagator_packed.unload();
              stochastic_propagator_packed.copy(*stochastic_propagator_2pt_SS[i_sample], HOST);
              stochastic_propagator_packed.load();

	      TIME(reductionsV2_2pt.V4( stochastic_propagator_packed, glist_sink_nucleon, propDN_SS_packed, propTS_SS_packed, true));

              TIME(corrW17_2pt.W_diagrams( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 17, true, false));
              TIME(corrW19_2pt.W_diagrams( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 19, true, false));

              TIME(reductionsV2_2pt.V2( stochastic_propagator_packed, glist_sink_nucleon, propDN_SS_packed, propTS_SS_packed, true));

              TIME(corrW18_2pt.W_diagrams( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 18, true, false));
              TIME(corrW20_2pt.W_diagrams( *reductions_UU_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 20, true, false));


              stochastic_propagator_packed.unload();
              stochastic_propagator_packed.copy(*stochastic_sources[i_sample], HOST);
              stochastic_propagator_packed.load();

	      stochastic_propagator_packed.apply_gamma5();

              TIME(reductionsV2_2pt.V2( stochastic_propagator_packed, glist_sink_nucleon, propTS_SS_packed, propUP_SS_packed, true));

              TIME(corrW21_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, 0, 21, true, false));
              TIME(corrW23_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, 0, 23, true, false));

              TIME(reductionsV2_2pt.V2( stochastic_propagator_packed, glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, true));

              TIME(corrW22_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, 0, 22, true, false));
              TIME(corrW24_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2D_2pt[i_sample], reductionsV2_2pt, 0, 24, true, false));


	  } //loop over stochastic samples

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt";

          TIME(produceOutput_2pt_packed(corrW17_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrW18_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrW19_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrW20_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));

          TIME(produceOutput_2pt_packed(corrW21_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrW22_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrW23_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
          TIME(produceOutput_2pt_packed(corrW24_2pt, outfilename, "4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));




//Here we perform the last sequential the DN DN case
//
        propTS_SS_packed.zero_where(BOTH);


#if 1
        for (int i_source_parallel=0; i_source_parallel < parallel_sources;++i_source_parallel){

          PLEGMA_Propagator<float> propTS_SS;


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
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS_SS.absorb(vectorAuxF, isc/3, isc%3);

          }

          propTS_SS_packed.pack_propagator_from_source_to_sink(propTS_SS, (source[3]+(i_source_parallel+1)*max_source_sink_separations+HGC_totalL[3])%HGC_totalL[3], max_source_sink_separations, i_source_parallel == 0 ? true : false);

        }



        PLEGMA_ScattCorrelator<float> corrW33_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW34_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW35_2pt(source, filtered_sourcemomentumList_2pt);
        PLEGMA_ScattCorrelator<float> corrW36_2pt(source, filtered_sourcemomentumList_2pt);


        corrW33_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W33");
        corrW34_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W34");
        corrW35_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W35");
        corrW36_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W36");





	for (int k=0; k< tSinks.size(); ++k){


          for (int i_sample=0; i_sample<n_stochastic_samples; ++i_sample){
            PLEGMA_Vector<float> stochastic_propagator_packed;

	    if (k==0 && (dotwopoint==1)){
              
	      stochastic_propagator_packed.unload();	      
  	      stochastic_propagator_packed.copy(*stochastic_sources[i_sample],HOST);
              stochastic_propagator_packed.load();

              stochastic_propagator_packed.apply_gamma5();

	      TIME(reductionsV2_2pt.V4( stochastic_propagator_packed, glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, true));

              TIME(corrW33_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 33, true, false));
              TIME(corrW34_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 34, true, false));

              TIME(reductionsV2_2pt.V2( stochastic_propagator_packed, glist_sink_nucleon, propUP_SS_packed, propTS_SS_packed, true));

              TIME(corrW35_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 35, true, false));
              TIME(corrW36_2pt.W_diagrams( *reductions_DD_V3_GAMMAF2U_2pt[i_sample], reductionsV2_2pt, 0, 36, true, false));


	    }
	  } //loop over stochastic samples
	  if ((dotwopoint==1)){
            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W_2pt";
	    TIME(produceOutput_2pt_packed(corrW33_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));//because of V4
            TIME(produceOutput_2pt_packed(corrW34_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW35_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));
            TIME(produceOutput_2pt_packed(corrW36_2pt, outfilename,"4pt", n_stochastic_samples, parallel_sources, attract_lookup_table));


	  }


	} //loop over tSinks
#endif
#endif

      }//loop over mpi2

      if (dotwopoint==1){

#if 1
/*	outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_N_2pt";
        TIME( corrN0_packed.apply_phase());
        TIME( corrN0_packed.apply_sign("N"));
        TIME( corrN0_packed.applyBoundaryConditions( true, parallel_sources, attract_lookup_table));
        TIME( corrN0_packed.writeHDF5(outfilename));

        TIME( corrNP_packed.apply_phase() );
        TIME( corrNP_packed.apply_sign("N") );
        TIME( corrNP_packed.applyBoundaryConditions( true, parallel_sources, attract_lookup_table));
        TIME( corrNP_packed.writeHDF5(outfilename) );
*/

#endif
/*
        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_2pt";
        TIME(corrP0UP.apply_sign("P"));
        TIME(corrP0UP.writeHDF5( outfilename ));
        TIME(corrP0DN.apply_sign("P"));
        TIME(corrP0DN.writeHDF5( outfilename ));
        TIME(corrPPUP.apply_sign("P"));
        TIME(corrPPUP.writeHDF5( outfilename ));
        TIME(corrPPDN.apply_sign("P"));
        TIME(corrPPDN.writeHDF5( outfilename ));
*/
      }


#if 1
/*      for (int k=0; k<tSinks.size();++k){
        reductionsV4_diluted_STOCHU_DN_UP.pop_back();
        reductionsV2_diluted_STOCHU_DN_UP.pop_back();
        reductionsV2_diluted_STOCHD_UP_UP.pop_back();
      }*/
      //}


      for(int i_samples=0; i_samples< n_stochastic_samples; ++i_samples) {
	if (dotwopoint==1){
          reductions_DD_V2_GAMMAF1U_U_2pt.pop_back();
          reductions_DD_V2_GAMMAF1U_D_2pt.pop_back();
          reductions_DD_V4_GAMMAF1U_D_2pt.pop_back();

          reductions_UU_V4_GAMMAF1U_D_2pt.pop_back();
	  reductions_UU_V2_GAMMAF1D_U_2pt.pop_back();

          reductions_UU_V3_GAMMAF2U_2pt.pop_back();
          reductions_DD_V3_GAMMAF2U_2pt.pop_back();
          reductions_DD_V3_GAMMAF2D_2pt.pop_back();


	}

/*
	for (int k=0; k<tSinks.size();++k){

          reductions_UU_V2_GAMMAF1D_U.pop_back();
          reductions_UU_V4_GAMMAF1U_D.pop_back();
          reductions_UU_V3_GAMMAF2U.pop_back();

          reductions_DD_V2_GAMMAF1U_U.pop_back();
          reductions_DD_V3_GAMMAF2D.pop_back();

	}*/

      }
#endif

    }//loop over source position

    free(attract_lookup_table);
#if 1
    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      if (dotwopoint==1){
        stochastic_propagator_2pt_SS.pop_back();
      }
    }

#endif
  }//loop in finalize
  finalize();
  return 0;
}
