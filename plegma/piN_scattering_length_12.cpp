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
  TIME(source.normalize_nstoch(n_stochastic_samples));
  TIME(source.writeHDF5( outputFilename ));

}

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name
                  ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
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

/*  PART I: Computing point to all  propagators:
          a; nucleon correlation function
          b; factors using stochastic source + boundary conditions
          c; 
*/


    for(int isource = startSource; isource < numSourcePositions; isource++){

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_V4_B;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_V2_B;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_V3_W;

      site source = sourcePositions[isource];

      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",isource, source[0], source[1], source[2], source[3]);

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]); 
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      PLEGMA_Propagator<float> prop_packed(BOTH); //To be saved for all the coherent sources.


      auto solve1 = [&](plegma::PLEGMA_Vector<double> &ppa,
                        plegma::PLEGMA_Vector<double> &pma, 
		        plegma::PLEGMA_Vector3D<double> &in,
			site source){
	    PLEGMA_Vector<double> vectorInOut_a;
            PLEGMA_Vector<double> vectorInOut_p;
	    PLEGMA_Vector<double> vectorAuxD;

	    { // Smearing the source
               PLEGMA_Vector3D<double> vector1;
               PLEGMA_Gauge3D<double> smearedGauge3D;
               smearedGauge3D.absorb(smearedGauge, source[DIM_T]);
               TIME(vector1.gaussianSmearing(in, smearedGauge3D, nsmearGauss, alphaGauss));
               vectorInOut_a.absorb(vector1,source[DIM_T]);
            }
	    vectorInOut_p.copy(vectorInOut_a);
            // Inverting
            PLEGMA_printf("Going to invert for antiperiodic case\n" );
            updateGaugeQuda(gauge, true);
            TIME(solver_a.UpdateSolver());
            TIME(solver_a.solve(vectorInOut_a, vectorInOut_a));
            PLEGMA_printf("Going to invert for periodic case \n");
            updateGaugeQuda(gauge, false);
            TIME(solver_p.UpdateSolver());
            TIME(solver_p.solve(vectorInOut_p, vectorInOut_p));
            
	    ppa.copy(vectorInOut_p);
            ppa.add(vectorInOut_a,1);

	    pma.copy(vectorInOut_p);
            pma.add(vectorInOut_a,-1);
            
            vectorAuxD.copy(ppa);
            TIME(ppa.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));

            vectorAuxD.copy(pma);
            TIME(pma.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));

      };
      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_packed
                                   ) {
	    for(int isc = 0 ; isc < 12 ; isc++){
              PLEGMA_Vector<double> vectorInOut_ppa,vectorInOut_pma;
	      PLEGMA_Vector<float>  vectorAuxF;
              PLEGMA_Vector<double>  vectorAuxD;
              PLEGMA_Vector3D<double> vector1;
              vector1.pointSource(source, isc/3, isc%3, DEVICE);
              solve1(vectorInOut_ppa, vectorInOut_pma, vector1, source);
              vectorAuxD.pack_propagator(vectorInOut_ppa, 
                                         vectorInOut_pma,
                                         source[DIM_T],
                                         HGC_totalL[DIM_T]/2);
	      vectorAuxF.copy(vectorAuxD);
              prop_packed.absorb(vectorAuxF, isc/3, isc%3);
	    };

      };
      TIME(computePropagator(prop_packed));

		
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
      }

      vectorSource_stochastic.randInit(rand_seed1);

      for (int i=0; i<n_stochastic_samples;++i){

	reductions_V4_B.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, sourcemomentumList_twopt.uniq_p(1)));

        reductions_V2_B.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, sourcemomentumList_twopt.uniq_p(1)));

        reductions_V3_W.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, sourcemomentumList_twopt.uniq_p(2)));


        PLEGMA_Vector<double> vectorPropagator_stochastic;
        PLEGMA_Vector<double> vectorPropagator_stochastic_ppa;
        PLEGMA_Vector<double> vectorPropagator_stochastic_pma;
        if (readstochastic==0){
	  vectorSource_stochastic.stochastic_Z(nroots);
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorSource_stochastic);
          vectorAuxF.unload();
          vectorAuxF.writeLIME("globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
          vectorPropagator_stochastic.scale(0.0);
          vectorPropagator_stochastic_ppa.scale(0.0);
          vectorPropagator_stochastic_pma.scale(0.0);
            
          if (timedilution){
            PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
            for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
              PLEGMA_printf("Inversion for timeslice %d\n", timeidx);

              PLEGMA_Vector<double> vectorInOut_ppa, vectorInOut_pma;
              PLEGMA_Vector3D<double> vectorIn;

	      vectorIn.absorb(vectorSource_stochastic,timeidx);

              site source_in=site({0,0,0,timeidx});
              double norm;
              norm=vectorIn.norm();
              PLEGMA_printf("Norm2 %e\n", norm);


              solve1(vectorInOut_ppa, vectorInOut_pma, vectorIn, source_in);
              norm=vectorInOut_ppa.norm();
              PLEGMA_printf("PPA Norm2 %e\n", norm);
              norm=vectorInOut_pma.norm();
              PLEGMA_printf("PMA Norm2 %e\n", norm);

              
              PLEGMA_printf("Inversion done for timeslice %d\n", timeidx);


	      vectorPropagator_stochastic_ppa.absorbTimeslice(vectorInOut_ppa, timeidx, false);
	      vectorPropagator_stochastic_pma.absorbTimeslice(vectorInOut_pma, timeidx, false);

	    }
	  }
          vectorPropagator_stochastic.pack_propagator(vectorPropagator_stochastic_ppa,
                                                      vectorPropagator_stochastic_pma,
                                                      source[DIM_T],
                                                      HGC_totalL[DIM_T]/2);
          vectorAuxF.copy(vectorPropagator_stochastic);
          vectorAuxF.unload();
          vectorAuxF.writeLIME("globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber);
          vectorAuxF.load();
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


           TIME(reductions_V2_B[i]->V2(vectorAuxF, glist_sink_nucleon,   prop_packed, prop_packed, false));

           TIME(reductions_V4_B[i]->V4(vectorAuxF, glist_sink_nucleon,   prop_packed, prop_packed, false));

           vectorAuxF.apply_gamma5();
           TIME(reductions_V3_W[i]->V3(vectorAuxF, glist_sink_meson,   prop_packed, true));


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

        PLEGMA_ScattCorrelator<float> corrB1_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB2_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB3_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB4_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB5_2pt(source, filtered_sourcemomentumList);


        PLEGMA_ScattCorrelator<float> corrW1_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW2_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW3_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW4_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW5_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW6_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW7_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW8_2pt(source, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW9_2pt(source, filtered_sourcemomentumList);


	corrB1_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B1");

        corrB2_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B2");

        corrB3_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B3");

        corrB4_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B4");

        corrB5_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B5");

        corrW1_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W1");

        corrW2_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W2");

        corrW3_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W3");

        corrW4_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W4");

        corrW5_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W5");

        corrW6_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W6");

        corrW7_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W7");

        corrW8_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W8");

        corrW9_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W9");

        //pi plus at the source
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorAuxD;
          PLEGMA_Vector<double> vectorInOut_ppa, vectorInOut_pma;
          PLEGMA_Vector<float> vectorAuxF;
          //Performing the smearing
          PLEGMA_Vector3D<double> vector1;
          vectorAuxF.absorb(prop_packed, isc/3, isc%3);
          vectorAuxD.copy(vectorAuxF);
          vector1.absorb( vectorAuxD, source[3]);
          vector1.mulMomentumPhases(momentum_i2,1);
	  vector1.apply_gamma_scatt(glist_source_meson[0]);
	  solve1(vectorInOut_ppa, vectorInOut_pma, vector1, source);
          vectorAuxD.pack_propagator(vectorInOut_ppa, 
                                     vectorInOut_pma,
                                     source[DIM_T],
                                     HGC_totalL[DIM_T]/2);
                                     
          vectorAuxF.copy(vectorAuxD);
          propTS.absorb(vectorAuxF, isc/3, isc%3);
	}

        vectorSource_stochastic.randInit(rand_seed1);

        for (int i=0; i<n_stochastic_samples;++i){

          vectorSource_stochastic.stochastic_Z(nroots);

          PLEGMA_ScattCorrelator<float> reductions_V3_B(source_reduction, sourcemomentumList_twopt.uniq_p(2));//implemented

          PLEGMA_ScattCorrelator<float> reductions_V4_W(source_reduction, sourcemomentumList_twopt.uniq_p(1));//implemented

          PLEGMA_ScattCorrelator<float> reductions_V2_W1(source_reduction, sourcemomentumList_twopt.uniq_p(1));//implemented

          PLEGMA_ScattCorrelator<float> reductions_V2_W2(source_reduction, sourcemomentumList_twopt.uniq_p(1));//implemented

	  
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxF.copy(vectorSource_stochastic);

	  TIME(reductions_V3_B.V3(vectorAuxF, glist_sink_meson, propTS, true));

          vectorAuxF.apply_gamma5();

          TIME(reductions_V4_W.V4(vectorAuxF, glist_sink_nucleon, prop_packed, propTS, true));

          TIME(reductions_V2_W1.V2(vectorAuxF, glist_sink_nucleon, prop_packed, propTS, true));

          TIME(reductions_V2_W2.V2(vectorAuxF, glist_sink_nucleon, propTS, prop_packed, true));


          TIME(corrB1_2pt.Recombination(reductions_V3_B, 
	                               *reductions_V4_B[i],
			                false,
					1,
					false,
					0,
					true,//checked
					false,
					false,
					false,
					true));

          TIME(corrB2_2pt.Recombination(reductions_V3_B,
                                         *reductions_V4_B[i],
                                           true,
                                           0,
                                           false,
                                           0,
                                           true,//checked
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrB3_2pt.Recombination(reductions_V3_B,
                                         *reductions_V2_B[i],
                                           false,
                                           0,
                                           false,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrB4_2pt.Recombination(reductions_V3_B,
                                         *reductions_V2_B[i],
                                           true,
                                           1,
                                           false,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrB5_2pt.Recombination(reductions_V3_B,
                                         *reductions_V2_B[i],
                                           true,
                                           2,
                                           true,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW1_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V4_W,
                                           false,
                                           1,
                                           false,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW2_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V4_W,
                                           false,
                                           2,
                                           false,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW3_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V2_W1,
                                           false,
                                           0,
                                           false,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW4_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V2_W1,
                                           false,
                                           2, 
                                           true,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW5_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V4_W,
                                           true,
                                           0, 
                                           false,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW6_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V2_W1,
                                           true,
                                           1, 
                                           false,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW7_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V2_W2,
                                           false,
                                           2, 
                                           true,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW8_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V2_W2,
                                           true,
                                           1, 
                                           false,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           true));

          TIME(corrW9_2pt.Recombination(*reductions_V3_W[i],
                                          reductions_V2_W2,
                                           false,
                                           0, 
                                           false,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           true));

	  

	
	}

        TIME(produceOutput(corrB1_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrB2_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrB3_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrB4_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrB5_2pt, outfilename, "4pt", n_stochastic_samples));

        TIME(produceOutput(corrW1_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW2_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW3_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW4_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW5_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW6_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW7_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW8_2pt, outfilename, "4pt", n_stochastic_samples));
        TIME(produceOutput(corrW9_2pt, outfilename, "4pt", n_stochastic_samples));

        for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
          reductions_V4_B.pop_back();
          reductions_V2_B.pop_back();
          reductions_V3_W.pop_back();
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
        PLEGMA_Vector3D<double> vectorIn1;
        vectorIn1.absorb(vectorSource_stochastic,source_local[DIM_T]);

        PLEGMA_Vector<double> vectorAuxD;
        PLEGMA_Vector<double> vectorInOut_ppa, vectorInOut_pma;

        solve1(vectorInOut_ppa, vectorInOut_pma, vectorIn1, source_local);
        vectorAuxD.pack_propagator(vectorInOut_ppa,
                                   vectorInOut_pma,
                                   source_local[DIM_T],
                                   HGC_totalL[DIM_T]/2);


        stochastic_oet_prop_zero_mom_packed.copy(vectorAuxD);   

      }//end of do_stochastic_oet

      PLEGMA_ScattCorrelator<float> reductionsV2_Z(source_reduction, list_mpf1_twopt);
      PLEGMA_ScattCorrelator<float> reductionsV4_Z(source_reduction, list_mpf1_twopt);

      TIME(reductionsV2_Z.V2( stochastic_oet_prop_zero_mom_packed, glist_sink_nucleon, prop_packed, prop_packed, true));
      TIME(reductionsV4_Z.V4( stochastic_oet_prop_zero_mom_packed, glist_sink_nucleon, prop_packed, prop_packed, true));


      { 

        PLEGMA_ScattCorrelator<float> reductionsV3_Z(source_reduction, list_mpf2_twopt);


        PLEGMA_ScattCorrelator<float> corrPION(source, list_mpi2_twopt);



        PLEGMA_ScattCorrelator<float> corrPION_PPA(source, list_mpi2_twopt);
        PLEGMA_ScattCorrelator<float> corrPION_PMA(source, list_mpi2_twopt);


        corrPION.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");


        //We first have a loop over all unique the source meson momentum p_i2     
        for (int i_mpi2=0; i_mpi2<mpi2_twopt.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2_twopt[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList_2pt = sourcemomentumList_twopt.extract(momentum_i2, 0);


          PLEGMA_ScattCorrelator<float> corrZ1_2pt(source, filtered_sourcemomentumList_2pt);
          PLEGMA_ScattCorrelator<float> corrZ2_2pt(source, filtered_sourcemomentumList_2pt);
          PLEGMA_ScattCorrelator<float> corrZ3_2pt(source, filtered_sourcemomentumList_2pt);
          PLEGMA_ScattCorrelator<float> corrZ4_2pt(source, filtered_sourcemomentumList_2pt);
          PLEGMA_ScattCorrelator<float> corrZ5_2pt(source, filtered_sourcemomentumList_2pt);


          corrZ1_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z1");

          corrZ2_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z2");

          corrZ3_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z3");

          corrZ4_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z4");

          corrZ5_2pt.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z5");

          std::string pi2x=std::to_string(momentum_i2[0]);
          std::string pi2y=std::to_string(momentum_i2[1]);
          std::string pi2z=std::to_string(momentum_i2[2]);

          //Multiplying by the appropriate momentum phase finite momentum OET
          {

            site source_local = sourcePositions[isource];
            PLEGMA_Vector3D<double> vectorIn1;
            vectorIn1.absorb(vectorSource_stochastic,source_local[DIM_T]);
            vectorIn1.mulMomentumPhases(momentum_i2,-1);

          
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<double> vectorInOut_ppa, vectorInOut_pma;
            PLEGMA_Vector<float> vectorAuxF;
    
            solve1(vectorInOut_ppa, vectorInOut_pma, vectorIn1, source_local);
            vectorAuxD.pack_propagator(vectorInOut_ppa,
                                       vectorInOut_pma,
                                       source_local[DIM_T],
                                       HGC_totalL[DIM_T]/2);

            
            stochastic_oet_prop_fini_mom_packed.copy(vectorAuxD);
    

          }

          PLEGMA_Vector<float> st_oet_fini;
          st_oet_fini.copy(stochastic_oet_prop_fini_mom_packed);

          st_oet_fini.apply_gamma5();

          TIME(reductionsV3_Z.V3( st_oet_fini, glist_sink_meson, prop_packed, true));

          
          TIME(corrZ1_2pt.Recombination( reductionsV3_Z, reductionsV2_Z, false, 0, false,  0, true, false, false, false, true ));

          TIME(corrZ2_2pt.Recombination( reductionsV3_Z, reductionsV2_Z, false, 2, true,  0, true, false, false, true, false ));

          TIME(corrZ3_2pt.Recombination( reductionsV3_Z, reductionsV2_Z, true,  1, false, 0, true, false, false, false, true ));

          TIME(corrZ4_2pt.Recombination( reductionsV3_Z, reductionsV4_Z, false, 2, false, 0, true, false, false, false, true ));
          
          TIME(corrZ5_2pt.Recombination( reductionsV3_Z, reductionsV4_Z, true, 0, false, 0, true, false, false, false, true ));

          TIME(produceOutput(corrZ1_2pt, outfilename, "4pt"));
          TIME(produceOutput(corrZ2_2pt, outfilename, "4pt"));
          TIME(produceOutput(corrZ3_2pt, outfilename, "4pt"));
          TIME(produceOutput(corrZ4_2pt, outfilename, "4pt"));
          TIME(produceOutput(corrZ5_2pt, outfilename, "4pt"));



          TIME(corrPION.P_diagrams( stochastic_oet_prop_zero_mom_packed, stochastic_oet_prop_fini_mom_packed, i_mpi2));


         }
         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P_2pt";

         TIME(corrPION.apply_sign("P"));
         TIME(corrPION.writeHDF5( outfilename ));

      }
    }
  }//loop in finalize
  finalize();
  return 0;
}
