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
  bool timedilution;
  int rand_seed1=1234;
  int rand_seed2=5678;
  int confnumber_int;
  std::string outfile_V="";
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

  int n_stochastic_samples;
  int max_source_sink_separations;
  int dotwopoint;
  int readstochastic;
  //setVerbosity(QUDA_DEBUG_VERBOSE);

  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("maxSourceSinkSeparations", "Maximal source sink separations", verbosity, max_source_sink_separations);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);
  HGC_options->set("dotwopoint", "Doing also the twopoint functions", verbosity,dotwopoint);
  HGC_options->set("readStochSamples", "Flag for switching read/building stochastic propagators", verbosity, readstochastic);
  HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);



  //=========================================================================================================//
  initializePLEGMA();


  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
      
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
    
 
    updateOptions(LIGHT);
    TIME(QUDA_solver solver_a(mu,1));


    // Loading to QUDA and computing plaquette also there
    updateGaugeQuda(gauge, false);
    plaqQuda();

    updateOptions(LIGHT);
    TIME(QUDA_solver solver_p(mu,1));

    //Get the confnumber for latfile
    char *ssource;
    asprintf(&ssource,"%04d", confnumber_int);
    std::string confnumber= ssource;
    free(ssource);

    momList sourcemomentumList_twopt(3,pathListMomenta_twopt,{1,2,});
    //For the twopoint functions we have also three momentum
    //first is pi2
    //second is pf1
    //third is pf2
    //and in this case the total momentum is defined as the sum of pf1 and pf2
    //to get pi1 we have to subtract pi2 from the total momentum
                    
                    
    if(sourcemomentumList_twopt.empty())
     PLEGMA_error("twopt momentumList empty");


    std::string given_twop_filename = twop_filename;


    if(sourcemomentumList_twopt.empty())
     PLEGMA_error("twopt momentumList empty");

    std::vector<std::vector<int>> mpi2_twopt = sourcemomentumList_twopt.uniq_p(0);
    momList list_mpi2_twopt(1,{mpi2_twopt,},{0,});

    std::vector<std::vector<int>> mpf1_twopt = sourcemomentumList_twopt.uniq_p(1);
    momList list_mpf1_twopt(1,{mpf1_twopt,},{0,});

    std::vector<std::vector<int>> mpf2_twopt = sourcemomentumList_twopt.uniq_p(2);
    momList list_mpf2_twopt(1,{mpf2_twopt,},{0,});


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

    std::vector<PLEGMA_Vector<float>*> stochastic_propagator_ppa;
    std::vector<PLEGMA_Vector<float>*> stochastic_propagator_pma;

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_propagator_ppa.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_propagator_pma.push_back(new PLEGMA_Vector<float>(HOST));
    }



    PLEGMA_Vector<float> stochastic_oet_prop_zero_mom_packed;
    PLEGMA_Vector<float> stochastic_oet_prop_fini_mom_packed;

    if (readstochastic==0){

      PLEGMA_Vector<double> vectorAuxD1(BOTH);//For storing the source (rotated and smeared)
      PLEGMA_Vector<double> vectorAuxD2_ppa(BOTH);//For storing the propagotor for the time-slices
      PLEGMA_Vector<double> vectorAuxD2_pma(BOTH);//For storing the propagotor for the time-slices
      PLEGMA_Vector<double> vectorInOut_a; //temporary vector using in solve
      PLEGMA_Vector<double> vectorInOut_p; //temporary vector using in solve
      PLEGMA_Vector<double> vectorTmp; //temporary vector using in solve

      PLEGMA_Vector<double> vectorSource(BOTH);//For storing the source 
      vectorSource.randInit(rand_seed2);

      for (int i=0; i<n_stochastic_samples; ++i){
        //Step(1) Creating the time-diluted stochastic source
        vectorSource.stochastic_Z(nroots);

        //Step(2) Save it on the host memory
        {
           PLEGMA_Vector<float> vectorAuxF;
           vectorAuxF.copy(vectorSource);
           vectorAuxF.unload();
           vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
        }


        vectorSource.unload();
        stochastic_sources[i]->copy(vectorSource,HOST);
        vectorSource.load();
           
        vectorAuxD1.copy(vectorSource);

        //Step(3) Smearing all the time slice
        TIME(vectorTmp.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(4) We rotate the source to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorTmp,+1));

        //In vectorAuxD2 we store the results for the inversion

        vectorAuxD2_ppa.scale(0.0);
        vectorAuxD2_pma.scale(0.0);

        if (timedilution){
          PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
          for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
            //Step(5) pick out a particular timeslice from the source
            vectorInOut_a.absorbTimeslice(vectorAuxD1, timeidx);
            //Step(6) Solve
            updateGaugeQuda(gauge, true);
            solver_a.UpdateSolver();
            solver_a.solve(vectorInOut_a, vectorInOut_a);

            //Step(7) pick out a particular timeslice from the source
            vectorInOut_p.absorbTimeslice(vectorAuxD1, timeidx);
            //Step(8) Solve
            updateGaugeQuda(gauge, false);
            solver_p.UpdateSolver();
            solver_p.solve(vectorInOut_p, vectorInOut_p);

            vectorTmp.copy(vectorInOut_p);
            vectorTmp.add(vectorInOut_a,1); 

            //Step(7) absorbing the particular timeslice to a 4d vector
            vectorAuxD2_ppa.absorbTimeslice(vectorTmp, timeidx, false);

            vectorTmp.copy(vectorInOut_p);
            vectorTmp.add(vectorInOut_a,-1);           

            //Step(7) absorbing the particular timeslice to a 4d vector
            vectorAuxD2_pma.absorbTimeslice(vectorTmp, timeidx, false);

          }
        }
        else{
          PLEGMA_printf("#piN_scattering_length_12: No time dilution is used n stochastic propagators\n");
          vectorInOut_a.copy(vectorAuxD1);
          updateGaugeQuda(gauge, true);
          solver_a.UpdateSolver();
          solver_a.solve(vectorInOut_a, vectorInOut_a);

          vectorInOut_p.copy(vectorAuxD1);
          updateGaugeQuda(gauge, false);
          solver_p.UpdateSolver();
          solver_p.solve(vectorInOut_p, vectorInOut_p);

          vectorTmp.copy(vectorInOut_p);
          vectorTmp.add(vectorInOut_a,1);  

          vectorAuxD2_ppa.copy(vectorTmp);

          vectorTmp.copy(vectorInOut_p);
          vectorTmp.add(vectorInOut_a,-1);  

          vectorAuxD2_pma.copy(vectorTmp);

        }

        //Step(6) We rotate back the propagator to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2_ppa,+1));

        //Step(7) Smearing all the time slice in the propagator
        TIME(vectorAuxD2_ppa.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(8) Save the propagator to the disk
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorAuxD2_ppa);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch_ppa"+std::to_string(i)+"_"+confnumber);
        }

        //Step(9) Save the propagator to the host memory
        vectorAuxD2_ppa.unload();
        stochastic_propagator_ppa[i]->copy(vectorAuxD2_ppa,HOST);
        vectorAuxD2_ppa.load();

        //Step(10) We rotate back the propagator to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2_pma,+1));

        //Step(7) Smearing all the time slice in the propagator
        TIME(vectorAuxD2_pma.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(8) Save the propagator to the disk
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorAuxD2_pma);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch_pma"+std::to_string(i)+"_"+confnumber);
        }

        //Step(9) Save the propagator to the host memory
        vectorAuxD2_pma.unload();
        stochastic_propagator_pma[i]->copy(vectorAuxD2_pma,HOST);
        vectorAuxD2_pma.load();


      } //loop over the stochastic samples

    }
    else{
      for (int i=0; i<n_stochastic_samples; ++i){
        std::string inputfilename=outfile_V+"globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read stochastic source from: %s\n",inputfilename.c_str());
        PLEGMA_Vector<float> vectorRead(BOTH);
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        stochastic_sources[i]->copy(vectorRead,HOST);
        inputfilename=outfile_V+"globalTfulltimedilution_propagator_nstoch_ppa"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        stochastic_propagator_ppa[i]->copy(vectorRead,HOST);
        inputfilename=outfile_V+"globalTfulltimedilution_propagator_nstoch_pma"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        stochastic_propagator_pma[i]->copy(vectorRead,HOST);   
      }
    }
    for(int isource = startSource; isource < numSourcePositions; isource++){


      site source = sourcePositions[isource];

      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",isource, source[0], source[1], source[2], source[3]);

    /******************************************************
     *
     * Step 5: Computing OET propagators
     *          
     *
     ******************************************************/
      {
        //Doing for +mu for the UP propagator spin dilution oet

        site source_local = sourcePositions[isource];
           
        PLEGMA_Vector<double> vectortmp1;

        PLEGMA_Vector<double> vectorInOut_p;
        PLEGMA_Vector<double> vectorInOut_a;

        PLEGMA_Vector<double> vectorInOut_ppa;
        PLEGMA_Vector<double> vectorInOut_pma;

        {  // Smearing the source

          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.absorb(vectorStoc_source_oet, source_local[DIM_T]);
          PLEGMA_Gauge3D<double> smearedGauge3D;
          smearedGauge3D.absorb(smearedGauge, source_local[DIM_T]);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectortmp1.absorb(vector2, source_local[DIM_T]);

        }

        vectorInOut_a.copy(vectortmp1);
        updateGaugeQuda(gauge, true);
        solver_a.UpdateSolver();
        TIME(solver_a.solve(vectorInOut_a, vectorInOut_a));

        vectorInOut_p.copy(vectortmp1);
        updateGaugeQuda(gauge, false);
        solver_p.UpdateSolver();
        TIME(solver_p.solve(vectorInOut_p, vectorInOut_p));

        vectorInOut_ppa.copy(vectorInOut_p);
        vectorInOut_ppa.add(vectorInOut_a,1);

        vectorInOut_pma.copy(vectorInOut_p);
        vectorInOut_pma.add(vectorInOut_a,-1);

        vectortmp1.pack_propagator(vectorInOut_ppa, vectorInOut_pma, source_local[DIM_T],  HGC_totalL[DIM_T]/2);

      
        //Gaussian smearing of the propagator
        TIME(vectorInOut_a.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));

        stochastic_oet_prop_zero_mom_packed.copy(vectorInOut_a);   

      }//end of do_stochastic_oet
      //uu case
      { 

        //We first have a loop over all unique the source meson momentum p_i2     
        for (int i_mpi2=0; i_mpi2<mpi2_twopt.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2_twopt[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList_twopt.extract(momentum_i2, 0);

          std::string pi2x=std::to_string(momentum_i2[0]);
          std::string pi2y=std::to_string(momentum_i2[1]);
          std::string pi2z=std::to_string(momentum_i2[2]);

          PLEGMA_ScattCorrelator<float> corrPION(source, filtered_sourcemomentumList);

          corrPION.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");

          TIME(corrPION.P_diagrams( stochastic_oet_prop_zero_mom_packed, stochastic_oet_prop_fini_mom_packed, i_mpi2));
         }
      }
    }
    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      stochastic_propagator_ppa.pop_back();
      stochastic_propagator_pma.pop_back();
    }
  }//loop in finalize
  finalize();
  return 0;
}
