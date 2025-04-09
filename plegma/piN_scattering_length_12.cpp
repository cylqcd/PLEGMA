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


    PLEGMA_Vector<float> stochastic_oet_prop_zero_mom_packed;
    PLEGMA_Vector<float> stochastic_oet_prop_fini_mom_packed;

    PLEGMA_Vector<float> stochastic_oet_prop_zero_mom_ppa;
    PLEGMA_Vector<float> stochastic_oet_prop_fini_mom_ppa;

    PLEGMA_Vector<float> stochastic_oet_prop_zero_mom_pma;
    PLEGMA_Vector<float> stochastic_oet_prop_fini_mom_pma;


    for(int isource = startSource; isource < numSourcePositions; isource++){

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_V3_GAMMAF2_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_V3_GAMMAF2GAMMA5_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_V4_GAMMAF1_U_U;
      site source = sourcePositions[isource];

      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",isource, source[0], source[1], source[2], source[3]);

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]); 
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      PLEGMA_Propagator<float> prop_packed(BOTH); //To be saved for all the coherent sources.

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto solve1 = [&](plegma::PLEGMA_Vector<double> &out, 
		        plegma::PLEGMA_Vector3D<double> &in,
			site source){
	    PLEGMA_Vector<double> vectorInOut_a;
            PLEGMA_Vector<double> vectorInOut_p;
            PLEGMA_Vector<double> vectorInOut_ppa;
            PLEGMA_Vector<double> vectorInOut_pma;
	    PLEGMA_Vector<double> vectorAuxD;

	    { // Smearing the source
               PLEGMA_Vector3D<double> vector1;
               TIME(vector1.gaussianSmearing(in, smearedGauge3D, nsmearGauss, alphaGauss));
               vectorInOut_a.absorb(vector1,source[DIM_T]);
            }
	    vectorInOut_p.copy(vectorInOut_a);
            // Inverting
            PLEGMA_printf("Going to invert for antiperiodic case\n" );
            updateGaugeQuda(gauge, true);
            solver_a.UpdateSolver();
            TIME(solver_a.solve(vectorInOut_a, vectorInOut_a));
            PLEGMA_printf("Going to invert for periodic case \n");
            updateGaugeQuda(gauge, false);
            solver_p.UpdateSolver();
            TIME(solver_p.solve(vectorInOut_p, vectorInOut_p));

	    vectorInOut_ppa.copy(vectorInOut_p);
            vectorInOut_ppa.add(vectorInOut_a,1);

            vectorInOut_ppa.copy(vectorAuxD);

	    vectorInOut_pma.copy(vectorInOut_p);
            vectorInOut_pma.add(vectorInOut_a,-1);

            vectorAuxD.pack_propagator(vectorInOut_ppa, vectorInOut_pma, source[DIM_T],  HGC_totalL[DIM_T]/2);

            TIME(out.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));
	    out.copy(vectorAuxD);
      };
      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_packed
                                   ) {
	    for(int isc = 0 ; isc < 12 ; isc++){
              PLEGMA_Vector<double> vectorInOut;
	      PLEGMA_Vector<float>  vectorAuxF;
              PLEGMA_Vector3D<double> vector1;
              vector1.pointSource(source, isc/3, isc%3, DEVICE);
              solve1(vectorInOut, vector1, source);
	      vectorAuxF.copy(vectorInOut);
              prop_packed.absorb(vectorAuxF, isc/3, isc%3);
	    };

      };
      TIME(computePropagator(prop_packed));

		
#if 0
      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_packed,
                                   PLEGMA_Propagator<float>& prop_PPA,
                                   PLEGMA_Propagator<float>& prop_PMA,
                                   int nSmear) {
                                 
                                 for(int isc = 0 ; isc < 12 ; isc++){

                                   PLEGMA_Vector<double> vectorAuxD;
                                   PLEGMA_Vector<float> vectorAuxF;

                                   PLEGMA_Vector<double> vectorInOut_a;
                                   PLEGMA_Vector<double> vectorInOut_p;
 
                                   PLEGMA_Vector<double> vectorInOut_ppa;
                                   PLEGMA_Vector<double> vectorInOut_pma;

                                   { // Smearing the source
                                     PLEGMA_Vector3D<double> vector1, vector2;
                                     vector1.pointSource(source, isc/3, isc%3, DEVICE);
                                     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                                     vectorInOut_a.absorb(vector2,source[DIM_T]);
                                   }
                                   // Inverting
                                   PLEGMA_printf("Going to invert for component %d\n", isc);

                                   updateGaugeQuda(gauge, true);
                                   solver_a.UpdateSolver();
                                   TIME(solver_a.solve(vectorInOut_a, vectorInOut_a));

                                   { // Smearing the source
                                     PLEGMA_Vector3D<double> vector1, vector2;
                                     vector1.pointSource(source, isc/3, isc%3, DEVICE);
                                     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                                     vectorInOut_p.absorb(vector2,source[DIM_T]);
                                   }
                                   // Inverting
                                   PLEGMA_printf("Going to invert for component %d\n", isc);

                                   updateGaugeQuda(gauge, false);
                                   solver_p.UpdateSolver();
                                   TIME(solver_p.solve(vectorInOut_p, vectorInOut_p));

                                   vectorInOut_ppa.copy(vectorInOut_p);
                                   vectorInOut_ppa.add(vectorInOut_a,1);

                                   TIME(vectorAuxD.gaussianSmearing(vectorInOut_ppa, smearedGauge, nSmear, alphaGauss));
                                   vectorInOut_ppa.copy(vectorAuxD);
                                   vectorAuxF.copy(vectorAuxD);
                                   prop_PPA.absorb(vectorAuxF, isc/3, isc%3);


                                   vectorInOut_pma.copy(vectorInOut_p);
                                   vectorInOut_pma.add(vectorInOut_a,-1);

                                   TIME(vectorAuxD.gaussianSmearing(vectorInOut_pma, smearedGauge, nSmear, alphaGauss));
                                   vectorInOut_pma.copy(vectorAuxD);
                                   vectorAuxF.copy(vectorAuxD);
                                   prop_PMA.absorb(vectorAuxF, isc/3, isc%3);

                                   vectorAuxD.pack_propagator(vectorInOut_ppa, vectorInOut_pma, source[DIM_T],  HGC_totalL[DIM_T]/2);
                                   vectorAuxF.copy(vectorAuxD);
                                   prop_packed.absorb(vectorAuxF, isc/3, isc%3);
                                   
                                 }
                               };
      TIME(computePropagator(prop_packed, prop_PPA, prop_PMA, nsmearGauss));
#endif

     //N diagram
      {
        std::vector<std::vector<int>> mtot = sourcemomentumList_twopt.uniq_p(3);
        momList list_mtot(1,{mtot,},{0,});
        PLEGMA_ScattCorrelator<float> corrN(source,list_mtot);
        //PLEGMA_ScattCorrelator<float> corrN_PPA(source,list_mtot);
        //PLEGMA_ScattCorrelator<float> corrN_PMA(source,list_mtot);


        //initialize diagram
        corrN.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired,glist_source_nucleon, glist_sink_nucleon,"N");
        //corrN_PPA.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired,glist_source_nucleon, glist_sink_nucleon,"N");
        //corrN_PMA.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired,glist_source_nucleon, glist_sink_nucleon,"N");


        PLEGMA_ScattCorrelator<float> reductionsT1(source_reduction, mtot);
        PLEGMA_ScattCorrelator<float> reductionsT2(source_reduction, mtot);

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, prop_packed, prop_packed, prop_packed));
        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, prop_packed, prop_packed, prop_packed));

        //write N
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_N";

        TIME( corrN.N_diagrams( reductionsT1, reductionsT2 ) );
        TIME( corrN.apply_phase() );
        TIME( corrN.apply_sign("N"));
        TIME( corrN.writeHDF5(outfilename));

        //TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, prop_PPA, prop_PPA, prop_PPA));
        //TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, prop_PPA, prop_PPA, prop_PPA));

        //write N
#if 0
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_N_PPA";

        TIME( corrN_PPA.N_diagrams( reductionsT1, reductionsT2 ) );
        TIME( corrN_PPA.apply_phase() );
        TIME( corrN_PPA.apply_sign("N"));
        TIME( corrN_PPA.writeHDF5(outfilename));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, prop_PMA, prop_PMA, prop_PMA));
        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, prop_PMA, prop_PMA, prop_PMA));

        //write N
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_N_PMA";

        TIME( corrN_PMA.N_diagrams( reductionsT1, reductionsT2 ) );
        TIME( corrN_PMA.apply_phase() );
        TIME( corrN_PMA.apply_sign("N"));
        TIME( corrN_PMA.writeHDF5(outfilename));
#endif
      }

      vectorSource_stochastic.randInit(rand_seed1);

      for (int i=0; i<n_stochastic_samples;++i){

	reductions_V4_GAMMAF1_U_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, sourcemomentumList_twopt.uniq_p(1)));

        reductions_V3_GAMMAF2_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, sourcemomentumList_twopt.uniq_p(2)));
	reductions_V3_GAMMAF2GAMMA5_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, sourcemomentumList_twopt.uniq_p(2)));

        PLEGMA_Vector<double> vectorPropagator_stochastic;
	PLEGMA_Vector<double> vectorInOut;
        if (readstochastic==0){
	  vectorSource_stochastic.stochastic_Z(nroots);
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorSource_stochastic);
          vectorAuxF.unload();
          vectorAuxF.writeLIME("globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
          vectorPropagator_stochastic.scale(0.0);
          if (timedilution){
            PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
            for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
              PLEGMA_Vector3D<double> vectorIn;
	      vectorIn.absorb(vectorSource_stochastic,timeidx);
              site source_in=site({0,0,0,timeidx});
              solve1(vectorInOut, vectorIn, source_in);
	      vectorPropagator_stochastic.absorbTimeslice(vectorInOut, timeidx, false);
	    }
	  }





          //In vectorAuxD2 we store the results for the inversion

          //vectorAuxD2_ppa.scale(0.0);
          //vectorAuxD2_pma.scale(0.0);

#if 0
          vectorSource_stochastic.stochastic_Z(nroots);
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorSource_stochastic);
          vectorAuxF.unload();
          vectorAuxF.writeLIME("globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
          //In vectorAuxD2 we store the results for the inversion

          vectorAuxD2_ppa.scale(0.0);
          vectorAuxD2_pma.scale(0.0);

          if (timedilution){
            PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
            for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){

              //Step(5) pick out a particular timeslice from the source
              vectorInOut_a.absorbTimeslice(vectorSource_stochastic, timeidx);
              //Step(6) Solve
              updateGaugeQuda(gauge, true);
              solver_a.UpdateSolver();
              solver_a.solve(vectorInOut_a, vectorInOut_a);

              //Step(7) pick out a particular timeslice from the source
              vectorInOut_p.absorbTimeslice(vectorSource_stochastic, timeidx);1
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
            vectorInOut_a.copy(vectorSource_stochastic);
            updateGaugeQuda(gauge, true);
            solver_a.UpdateSolver();
            solver_a.solve(vectorInOut_a, vectorInOut_a);

            vectorInOut_p.copy(vectorSource_stochastic,);
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

          //Step(7) Smearing all the time slice in the propagator
          TIME(vectorAuxD1.gaussianSmearing(vectorAuxD2_ppa, smearedGauge, nsmearGauss, alphaGauss ));

          {
            PLEGMA_Vector<float> vectorAuxF;
            vectorAuxF.copy(vectorAuxD1);
            vectorAuxF.unload();
            vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch_ppa"+std::to_string(i)+"_"+confnumber);
          }

          stochastic_propagator_ppa.copy(vectorAuxD1);

          //Step(7) Smearing all the time slice in the propagator
          TIME(vectorAuxD1.gaussianSmearing(vectorAuxD2_pma, smearedGauge, nsmearGauss, alphaGauss ));

          {
            PLEGMA_Vector<float> vectorAuxF;
            vectorAuxF.copy(vectorAuxD1);
            vectorAuxF.unload();
            vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch_pma"+std::to_string(i)+"_"+confnumber);
          }

          stochastic_propagator_pma.copy(vectorAuxD1);

#endif

        }
        else{
          std::string inputfilename="globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
          PLEGMA_printf("Read stochastic source from: %s\n",inputfilename.c_str());
          PLEGMA_Vector<float> vectorRead(BOTH);
          vectorRead.readFile(inputfilename,LIME_FORMAT);
          vectorRead.load();
          vectorSource_stochastic.copy(vectorRead);
          inputfilename="globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
          PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
          vectorRead.readFile(inputfilename,LIME_FORMAT);
          vectorPropagator_stochastic.copy(vectorRead,HOST);
          vectorPropagator_stochastic.load();

        }

	{
	   PLEGMA_Vector<float> vectorAuxF;
	   vectorAuxF.copy(vectorPropagator_stochastic);
           TIME(reductions_V3_GAMMAF2_U[i]->V3(vectorAuxF, glist_sink_meson,   prop_packed, true));

	   vectorAuxF.apply_gamma5();

           TIME(reductions_V3_GAMMAF2GAMMA5_U[i]->V3(vectorAuxF, glist_sink_meson,   prop_packed, true));

	   vectorAuxF.copy(vectorSource_stochastic);
	   TIME(reductions_V4_GAMMAF1_U_U[i]->V4( vectorAuxF, glist_sink_meson, prop_packed, prop_packed, false));
	}
        
      }

      readstochastic=1;


      //We first have a loop over all unique the source meson momentum p_i2
      for (int i_mpi2=0; i_mpi2<mpi2_twopt.size(); ++i_mpi2){

        auto &momentum_i2 =  mpi2_twopt[i_mpi2];
        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList_twopt.extract(momentum_i2, 0);

        momList filtered_sourcemomentumList_2pt = sourcemomentumList_twopt.extract(momentum_i2, 0);

        std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList_2pt.uniq_p(3);

        std::vector<std::vector<int>> mpi2_filt  ;
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);
        //PLEGMA_printf("mptot_filt.size() %d\n",mptot_filt.size());
        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

        PLEGMA_Propagator<float> propTS;

        PLEGMA_ScattCorrelator<float> corrB13_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB14_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB15_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB16_2pt(source, filtered_sourcemomentumList);


	corrB13_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B13");
        corrB14_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B14");
        corrB15_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B15");
        corrB16_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B16");


        //pi plus at the source
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorAuxD;
          PLEGMA_Vector<float> vectorAuxF;
          //Performing the smearing
          PLEGMA_Vector3D<double> vector1;
          vectorAuxF.absorb(prop_packed, isc/3, isc%3);
          vectorAuxD.copy(vectorAuxF);
          vector1.absorb( vectorAuxD, source[3]);
          vector1.mulMomentumPhases(momentum_i2,1);
	  vector1.apply_gamma_scatt(glist_source_meson[0]);
	  solve1(vectorAuxD, vector1, source);
          vectorAuxF.copy(vectorAuxD);
          propTS.absorb(vectorAuxF, isc/3, isc%3);
	}

        vectorSource_stochastic.randInit(rand_seed1);

        for (int i=0; i<n_stochastic_samples;++i){

          vectorSource_stochastic.stochastic_Z(nroots);

          PLEGMA_ScattCorrelator<float> reductions_V3_GAMMAF2_SEQ(source_reduction, sourcemomentumList_twopt.uniq_p(2));//implemented

	  {
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxF.copy(vectorSource_stochastic);

	    TIME(reductions_V3_GAMMAF2_SEQ.V3(vectorAuxF, glist_sink_meson, propTS, true));


            TIME(corrB13_2pt.Recombination(reductions_V3_GAMMAF2_SEQ, 
				           *reductions_V4_GAMMAF1_U_U[i],
					   false,
					   1,
					   false,
					   0,
					   true,
					   false,
					   false,
					   false,
					   true));

	  }


          
	  PLEGMA_ScattCorrelator<float> reductions_V4_GAMMAF1_1(source_reduction, sourcemomentumList_twopt.uniq_p(1));
	  PLEGMA_ScattCorrelator<float> reductions_V4_GAMMAF1_2(source_reduction, sourcemomentumList_twopt.uniq_p(1));

	  {
            PLEGMA_Vector<float> vectorAuxF;
            vectorAuxF.copy(vectorSource_stochastic);

            TIME(reductions_V4_GAMMAF1_1.V4( vectorAuxF,  glist_sink_nucleon, prop_packed,  propTS, false));
            TIME(reductions_V4_GAMMAF1_2.V4( vectorAuxF,  glist_sink_nucleon, propTS,  prop_packed, false));

	  }



	
	}


      }


    /******************************************************
     *
     * Step 5: Computing OET propagators
     *          
     *
     ******************************************************/
      vectorStoc_source_oet.stochastic_Z(nroots);

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

        TIME(vectorInOut_a.gaussianSmearing(vectorInOut_ppa, smearedGauge, nsmearGauss, alphaGauss));
        vectorInOut_ppa.copy(vectorInOut_a);

        TIME(vectorInOut_a.gaussianSmearing(vectorInOut_pma, smearedGauge, nsmearGauss, alphaGauss));
        vectorInOut_pma.copy(vectorInOut_a);



        vectortmp1.pack_propagator(vectorInOut_ppa, vectorInOut_pma, source_local[DIM_T],  HGC_totalL[DIM_T]/2);

      
        //Gaussian smearing of the propagator
        //TIME(vectorInOut_a.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));

        stochastic_oet_prop_zero_mom_ppa.copy(vectorInOut_ppa);   
        stochastic_oet_prop_zero_mom_pma.copy(vectorInOut_pma);   

        stochastic_oet_prop_zero_mom_packed.copy(vectortmp1);   

      }//end of do_stochastic_oet
      { 

        PLEGMA_ScattCorrelator<float> corrPION(source, list_mpi2_twopt);

        PLEGMA_ScattCorrelator<float> corrPION_PPA(source, list_mpi2_twopt);
        PLEGMA_ScattCorrelator<float> corrPION_PMA(source, list_mpi2_twopt);


        corrPION.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");

        corrPION_PPA.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
        corrPION_PMA.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");


        //We first have a loop over all unique the source meson momentum p_i2     
        for (int i_mpi2=0; i_mpi2<mpi2_twopt.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2_twopt[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList_twopt.extract(momentum_i2, 0);

          std::string pi2x=std::to_string(momentum_i2[0]);
          std::string pi2y=std::to_string(momentum_i2[1]);
          std::string pi2z=std::to_string(momentum_i2[2]);

          //Multiplying by the appropriate momentum phase finite momentum OET
          {

            site source_local = sourcePositions[isource];

            PLEGMA_Vector<double> vectortmp1;

            PLEGMA_Vector<double> vectorInOut_p;
            PLEGMA_Vector<double> vectorInOut_a;

            PLEGMA_Vector<double> vectorInOut_ppa;
            PLEGMA_Vector<double> vectorInOut_pma;

            vectortmp1.copy(vectorStoc_source_oet);
  
            std::vector<int> tmp_4Dmom= momentum_i2 ;
            tmp_4Dmom.push_back(0);
            vectortmp1.mulMomentumPhases(tmp_4Dmom,-1);
          
            {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectortmp1, source_local[DIM_T]);
              PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, source_local[DIM_T]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectortmp1.absorb(vector2, source_local[DIM_T],true);

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
            
            TIME(vectortmp1.gaussianSmearing(vectorInOut_ppa, smearedGauge, nsmearGauss, alphaGauss));
            stochastic_oet_prop_fini_mom_ppa.copy(vectortmp1);

            TIME(vectortmp1.gaussianSmearing(vectorInOut_pma, smearedGauge, nsmearGauss, alphaGauss));
            stochastic_oet_prop_fini_mom_pma.copy(vectortmp1);

            vectortmp1.pack_propagator(vectorInOut_ppa, vectorInOut_pma, source_local[DIM_T],  HGC_totalL[DIM_T]/2);


            //Gaussian smearing of the propagator
            TIME(vectorInOut_a.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));

            stochastic_oet_prop_fini_mom_packed.copy(vectorInOut_a);

          }

          TIME(corrPION_PMA.P_diagrams( stochastic_oet_prop_zero_mom_pma, stochastic_oet_prop_fini_mom_pma, i_mpi2));

          TIME(corrPION_PPA.P_diagrams( stochastic_oet_prop_zero_mom_ppa, stochastic_oet_prop_fini_mom_ppa, i_mpi2));

          TIME(corrPION.P_diagrams( stochastic_oet_prop_zero_mom_packed, stochastic_oet_prop_fini_mom_packed, i_mpi2));

         }
         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_2pt";

         TIME(corrPION.apply_sign("P"));
         TIME(corrPION.writeHDF5( outfilename ));

         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_PPA_2pt";

         TIME(corrPION_PPA.apply_sign("P"));
         TIME(corrPION_PPA.writeHDF5( outfilename ));

         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_PMA_2pt";

         TIME(corrPION_PMA.apply_sign("P"));
         TIME(corrPION_PMA.writeHDF5( outfilename ));


      }
    }
  }//loop in finalize
  finalize();
  return 0;
}
