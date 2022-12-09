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


void produceOutput( PLEGMA_ScattCorrelator<double> source,
                      std::string outputFilename,
                      std::string diagram_name,
                      int n_stochastic_samples,
                      int n_coherent_source,
                      int *coherent_source_table_timeslice ){
    source.apply_phase();
    source.apply_sign(diagram_name);
    source.applyBoundaryConditions( true ,  n_coherent_source, coherent_source_table_timeslice);
    source.normalize_nstoch(n_stochastic_samples);
    source.writeHDF5( outputFilename );

  }



void produceOutput( PLEGMA_ScattCorrelator<double> source,
                      std::string outputFilename,
                      std::string diagram_name,
                      int n_coherent_source,
                      int *coherent_source_table_timeslice
                    ){
    source.apply_phase();
    source.apply_sign(diagram_name);
    source.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice );
    source.writeHDF5( outputFilename );

  }

void produceOutput( PLEGMA_ScattCorrelator<double> source,
                      std::string outputFilename,
                      std::string diagram_name
                    ){
    source.apply_phase();
    source.apply_sign(diagram_name);
    source.applyBoundaryConditions( true );
    source.writeHDF5( outputFilename );
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

  std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
  std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

  std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};//,S12,S13,S23,S41,S42,S43};

  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4};


  int n_stochastic_samples;


  auto add_options = [&](Options& options) {
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    options.set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    options.set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
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


    PLEGMA_Vector<float> vectorSource_stochastic;
    vectorSource_stochastic.randInit(rand_seed1);

    //Computing with ubaru insertion
    //Step (1) produce the stochastic sample
    //Step (2) produce standard point to all propagators
    //Step (3) produce sequential through the source
    //Step (4) produce stochastic propagators pieces
    //Step (5) doing the recombination
    //Step (6) doing the one end trick calculation for the Z diagrams

    std::vector<PLEGMA_Vector<float>*> stochastic_sources;

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.push_back(new PLEGMA_Vector<float>(HOST));
      vectorSource_stochastic.stochastic_Z(nroots);
      vectorSource_stochastic.unload();
      stochastic_sources[i]->copy(vectorSource_stochastic,HOST);
      vectorSource_stochastic.load();
    }

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
				   double run_mu, WHICHFLAVOR fl, int nSmear,  bool finalize) {
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
				     vector1.pointSource(source, isc/3, isc%3, DEVICE);
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

      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propDN;

      PLEGMA_Propagator<float> propUP_SL;
      PLEGMA_Propagator<float> propDN_SL;


      PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1_twopt);
      TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"));

      PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1_twopt);
      TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"));

      bool computed_light = false;
      // If twop_filename exists we hold the computation of the light props
      if(access( twop_filename.c_str(), F_OK ) == -1) {
	TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
	TIME(computePropagator(propDN, propDN_SL, -mu_ud, LIGHT, nsmearGauss, false));
	computed_light = true; 
      }

      //Computing T reductions+recombination
      {
        PLEGMA_ScattCorrelator<float> reductionsT1N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsT2N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
        //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
        TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
        //PLEGMA_printf("Nucleon T2 reduction\n");
        TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
        //PLEGMA_printf("Nucleon T2 reduction ready\n");
        TIME(corrNP.N_diagrams( reductionsT1N, reductionsT2N ));
        //PLEGMA_printf("Nucleon diagram ready\n");

      }

	//Computing T reductions+recombination
      {
        PLEGMA_ScattCorrelator<float> reductionsT1N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsT2N(source_reduction, sourcemomentumList_twopt.uniq_p(1));
        //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
        TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN));
        //PLEGMA_printf("Nucleon T2 reduction\n");
        TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN));
        //PLEGMA_printf("Nucleon T2 reduction ready\n");
        TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ));
	//PLEGMA_printf("Nucleon diagram ready\n");
      }

      //We implement the UD part first
      //The neutron piplus at the source
	

      int sequential_time_source=source[DIM_T];

      //We implement the following factors in this part of the calculation
      //source piplus 
      //B9-12
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAC_TUD;

	//W17,19
	std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V4_GAMMAF1D_TUD;

        //W18,20
	std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1D_TUD;

	//W21,23
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1TUD_U;

	//W22,24
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_TUD;

	//source pizero u
        //B3-6
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAC_TUU;

        //W5,6
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V4_GAMMAF1D_TUU;

        //W7,8
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1D_TUU;

        //W13,15
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1TUU_U;

        //W14,16
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_TUU;

	//source pizero d
	
	//B7,8
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAC_TDD;

        //W10,12
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V4_GAMMAF1TDD_U;

        //W9,11
        std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1TDD_U;
	



	for(int i=0; i< n_stochastic_samples; ++i) {
          for (int j=0; j<mpi2_threept.size();++j){
            try
            {
              reductions_UU_V3_GAMMAC_TUD.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));
 
	      reductions_UU_V3_GAMMAC_TUU.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));

              reductions_UU_V3_GAMMAC_TDD.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpc));


            }
	    catch(std::bad_alloc&){
                PLEGMA_printf("Memory allocation fails to store factors V3");
                exit(1);
            }
            for (int k=0; k<tSinks.size(); ++k){

	      try
	      {
                reductions_DD_V4_GAMMAF1D_TUD.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
                reductions_DD_V2_GAMMAF1D_TUD.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));

	        reductions_DD_V2_GAMMAF1TUD_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
                reductions_DD_V2_GAMMAF1U_TUD.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));

                reductions_DD_V4_GAMMAF1D_TUU.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
                reductions_DD_V2_GAMMAF1D_TUU.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));

                reductions_DD_V2_GAMMAF1TUU_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
                reductions_DD_V2_GAMMAF1U_TUU.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));


                reductions_DD_V2_GAMMAF1TDD_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));
                reductions_DD_V4_GAMMAF1TDD_U.push_back(new PLEGMA_ScattCorrelator<float>(source_reduction, list_mpf1_threept));



              }
              catch(std::bad_alloc&){
                PLEGMA_printf("Memory allocation fails to store factors V2 V4");
                exit(1);
              }
            }
	  }
	}

	//We first have a loop over all unique the source meson momentum p_i2
        for (int i_mpi2=0; i_mpi2<mpi2_threept.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2_threept[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList_threept.extract(momentum_i2, 0);

		
	  std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(4);
          std::vector<std::vector<int>> mpi2_filt  ;
          mpi2_filt.assign(mptot_filt.size(),momentum_i2);
          momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

          PLEGMA_ScattCorrelator<float> reductionsT1(source_reduction, sourcemomentumList_threept.uniq_p(1));
          PLEGMA_ScattCorrelator<float> reductionsT2(source_reduction, sourcemomentumList_threept.uniq_p(1));

		 
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
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vector2.mulMomentumPhases(momentum_i2,1);
              vectorAuxD.absorb(vector2,sequential_time_source);
	    }
            vectorAuxD2.absorbTimeslice(vectorAuxD,sequential_time_source, false);

            //Perform multiplication with glist_sink_meson[0]
            vectorAuxD2.apply_gamma_scatt(glist_sink_meson[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_sink_meson[0]=gamma_5 and momentum momentum_i2
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

	  PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus1(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus2(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus3(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_neutronpiplus4(source, list_mpi2ptot);

	  corrTproton_neutronpiplus1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq11");

          corrTproton_neutronpiplus2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq12");

          corrTproton_neutronpiplus3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq13");

          corrTproton_neutronpiplus4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq14");

	  TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS, propUP, propDN));
          TIME(corrTproton_neutronpiplus1.convertTreductiontoDiagram( reductionsT1, false, true, false ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS, propDN, propUP));
          TIME(corrTproton_neutronpiplus2.convertTreductiontoDiagram( reductionsT2, false, false, true ));

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP, propTS, propDN));
          TIME(corrTproton_neutronpiplus3.convertTreductiontoDiagram( reductionsT1, false, false, false ));

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP, propDN, propTS));
          TIME(corrTproton_neutronpiplus4.convertTreductiontoDiagram( reductionsT1, false, false, true));

	  //creating factors

          for (int i=0; i<n_stochastic_samples; ++i){

            vectorSource_stochastic.copy(*stochastic_sources[i],HOST);
	    vectorSource_stochastic.load();

	    //V3
            TIME(reductions_UU_V3_GAMMAC_TUD[i_mpi2*n_stochastic_samples+i]->V3( vectorSource_stochastic, glist_insertion,   propTS, true));


	    for (int k=0; k<tSinks.size(); ++k){
	      int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);

              int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3];
	      PLEGMA_Propagator<float> propTS_as_sink;
	      propTS_as_sink.pack_propagator_as_sink(propTS, global_fixSinkTime);

	      PLEGMA_Propagator<float> propSINGLE_as_sink;
	      propSINGLE_as_sink.pack_propagator_as_sink(propDN, global_fixSinkTime);

	      vectorSource_stochastic.apply_gamma5();

	      //V2 
              TIME(reductions_DD_V2_GAMMAF1D_TUD[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));

	      //V4
	      TIME(reductions_DD_V4_GAMMAF1D_TUD[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V4( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));


	      propSINGLE_as_sink.pack_propagator_as_sink(propUP, global_fixSinkTime);

	      //V2
              TIME(reductions_DD_V2_GAMMAF1TUD_U[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propTS_as_sink, propSINGLE_as_sink, false));
	      TIME(reductions_DD_V2_GAMMAF1U_TUD[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));

	    }//loop over source sink separations

	  }//loop over stochastic samples

	  //pi zero  at the source (U part)
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            //Performing the smearing

            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propUP, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vector2.mulMomentumPhases(momentum_i2,1);
              vectorAuxD.absorb(vector2,sequential_time_source);
            }
            vectorAuxD2.absorbTimeslice(vectorAuxD,sequential_time_source, false);

            //Perform multiplication with glist_sink_meson[0]
            vectorAuxD2.apply_gamma_scatt(glist_sink_meson[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_sink_meson[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert UP for sequential propagator UP  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1);
            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS.absorb(vectorAuxF, isc/3, isc%3);
          }

	  PLEGMA_ScattCorrelator<float> corrTproton_protonpizero1(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero2(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero3(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero4(source, list_mpi2ptot);

          corrTproton_protonpizero1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq21");

          corrTproton_protonpizero2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq22");

          corrTproton_protonpizero3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq23");

          corrTproton_protonpizero4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq24");

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS, propUP, propDN));
          TIME(corrTproton_protonpizero1.convertTreductiontoDiagram( reductionsT1, false, true, false ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS, propDN, propUP));
          TIME(corrTproton_protonpizero2.convertTreductiontoDiagram( reductionsT2, false, false, true ));

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP, propDN, propTS));
          TIME(corrTproton_protonpizero3.convertTreductiontoDiagram( reductionsT1, false, false, false ));

          TIME(reductionsT1.T2(glist_source_nucleon,glist_sink_nucleon, propUP, propDN, propTS));
          TIME(corrTproton_protonpizero4.convertTreductiontoDiagram( reductionsT1, false, false, true));

	   //creating factors

          for (int i=0; i<n_stochastic_samples; ++i){

            vectorSource_stochastic.copy(*stochastic_sources[i],HOST);
            vectorSource_stochastic.load();

            vectorSource_stochastic.apply_gamma5();

            //V3
            TIME(reductions_UU_V3_GAMMAC_TUU[i_mpi2*n_stochastic_samples+i]->V3( vectorSource_stochastic, glist_insertion,   propTS, true));


            for (int k=0; k<tSinks.size(); ++k){
	      int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);

              int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3];
              PLEGMA_Propagator<float> propTS_as_sink;
              propTS_as_sink.pack_propagator_as_sink(propTS, global_fixSinkTime);

              PLEGMA_Propagator<float> propSINGLE_as_sink;
              propSINGLE_as_sink.pack_propagator_as_sink(propDN, global_fixSinkTime);


              //V2
              TIME(reductions_DD_V2_GAMMAF1D_TUU[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));

              //V4
              TIME(reductions_DD_V4_GAMMAF1D_TUU[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V4( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));


              propSINGLE_as_sink.pack_propagator_as_sink(propUP, global_fixSinkTime);

              //V2
              TIME(reductions_DD_V2_GAMMAF1TUU_U[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propTS_as_sink, propSINGLE_as_sink, false));
              TIME(reductions_DD_V2_GAMMAF1U_TUU[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));

            }//loop over source sink separations

          }//loop over stochastic samples

	  for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            //Performing the smearing

            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vector2.mulMomentumPhases(momentum_i2,1);
              vectorAuxD.absorb(vector2,sequential_time_source);
            }
            vectorAuxD2.absorbTimeslice(vectorAuxD,sequential_time_source, false);

            //Perform multiplication with glist_sink_meson[0]
            vectorAuxD2.apply_gamma_scatt(glist_sink_meson[0]);
            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,-1);

            //Computing sequential propagators UD T_fii with insertion
            //glist_sink_meson[0]=gamma_5 and momentum momentum_i2
            PLEGMA_printf("Going to invert DN for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));
            //performing rotation to physical base
            vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,-1);
            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
            vectorAuxF.copy(vectorAuxD);
            propTS.absorb(vectorAuxF, isc/3, isc%3);
          }

          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero5(source, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrTproton_protonpizero6(source, list_mpi2ptot);

          corrTproton_protonpizero5.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq25");

          corrTproton_protonpizero6.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq26");

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP, propTS, propUP));
          TIME(corrTproton_protonpizero5.convertTreductiontoDiagram( reductionsT1, false, true, false ));

          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propUP, propTS, propDN));
          TIME(corrTproton_protonpizero6.convertTreductiontoDiagram( reductionsT2, false, false, true ));

	  //creating factors

          for (int i=0; i<n_stochastic_samples; ++i){

            vectorSource_stochastic.copy(*stochastic_sources[i],HOST);
            vectorSource_stochastic.load();

            //V3
            TIME(reductions_UU_V3_GAMMAC_TDD[i_mpi2*n_stochastic_samples+i]->V3( vectorSource_stochastic, glist_insertion,   propTS, true));


            for (int k=0; k<tSinks.size(); ++k){
              int tsinkMtsource = tSinks[k];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);

              int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3];
              PLEGMA_Propagator<float> propTS_as_sink;
              propTS_as_sink.pack_propagator_as_sink(propTS, global_fixSinkTime);

              PLEGMA_Propagator<float> propSINGLE_as_sink;
              propSINGLE_as_sink.pack_propagator_as_sink(propDN, global_fixSinkTime);

              vectorSource_stochastic.apply_gamma5();

              //V2
              TIME(reductions_DD_V2_GAMMAF1TDD_U[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V2( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));

              //V4
              TIME(reductions_DD_V4_GAMMAF1TDD_U[(i_mpi2*n_stochastic_samples+i)*tSinks.size()+k]->V4( vectorSource_stochastic,   glist_sink_nucleon, propSINGLE_as_sink, propTS_as_sink, false));


            }//loop over source sink separations

          }//loop over stochastic samples

	}//loop over source meson momentum

	/*********************************************
	 *
	 *
	 * Part II Stochastic inversion and contraction
	 *         with stochastic propagators
	 *         this will do not involve any 
	 *         sequential propagator
	 *
	 *********************************************/

	//UU stochastic piece
        for (int i=0; i<n_stochastic_samples; ++i){         

          std::vector<PLEGMA_Vector<float>*> stochastic_propags;

          for(int i=0; i< HGC_totalL[DIM_T]; ++i) {
            stochastic_propags.push_back(new PLEGMA_Vector<float>(HOST));
          }



	  //Step(1) load the souce from the host memory
	  PLEGMA_Vector<float> vectorSource_stochastic;
          PLEGMA_Vector<double> vectorAuxD1,vectorAuxD2;

	  vectorSource_stochastic.copy(*stochastic_sources[i],HOST);
          vectorSource_stochastic.load();
	  
	  vectorAuxD1.copy(vectorSource_stochastic);

          //Step(2) Smearing all the time slice
          TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

          //Step(3) We rotate the source to the physical basis
          TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

          PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
          for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
            //Step(4) pick out a particular timeslice from the source
            vectorAuxD2.absorbTimeslice(vectorAuxD1, timeidx);

            //Step(5) Solve
            TIME(solver.solve(vectorAuxD2, vectorAuxD2));
          
	    //Step(6) We rotate back the propagator to the physical basis
            TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2, +1));
            //Step(7) Smearing all the time slice in the propagator
            TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));
 
	    //Step(8) Save the propagator to the host memory
            vectorAuxD2.unload();
            stochastic_propags[i]->copy(vectorAuxD2,HOST);
            vectorAuxD2.load();


	  }

	  //Neutron pi plus V3 reductions
	  //V3
	  
          PLEGMA_Vector<float> vectorPropagator_stochastic;

	  
	  for (int k=0; k<tSinks.size(); ++k){
             int tsinkMtsource = tSinks[k];
             if(tsinkMtsource >= HGC_totalL[3])
               PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);

            int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3];
	    vectorPropagator_stochastic.pack_fermion_to_sink(stochastic_propags, global_fixSinkTime);
	  
	    //TIME(reductions_UU_V3_GAMMAC_TDD[i_mpi2*n_stochastic_samples+i]->V3( vectorPropagator_stochastic, glist_insertion,   propUP, true));

	  
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
        
      }  //keep track of
    }//source positions
  }//loop in finalize
  finalize();
  return 0;
}
