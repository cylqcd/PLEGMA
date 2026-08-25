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
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", 
                                            "alpha-gauss","momlist-filename","momlisttwopt-filename",
                                            "momlistthreept-filename","nsrc", "src-filename", "maxQsq",
                                            "twop-filename", "corr-file-format", "corr-space", "tSinks",
                                            "Projs", "threep-filename","confnumber"}; 
    


int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::vector<double> mu_s;
  std::vector<double> mu_c;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int nsmearGauss_c = 0;
  int startSource = 0;
  int confnumber_int;                

  std::string prOrNt = "proton";
  std::string srcInputFile = "./input.src";
  std::string outdiagramPrefix="";


  std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5};

  std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3};
  std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3};

  std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
  std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

  std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
  std::vector<GAMMAS_SCATT> gammas_insertion = {ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4,S_12,S_13,S_23,S_41,S_42,S_43};

  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};

  auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
    options.set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    options.set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
		     };
  add_options(*HGC_options);
  if(prOrNt != "proton" && prOrNt != "neutron") PLEGMA_error("This exec is only for nucleon, %s is not allowed",prOrNt.c_str());
  std::string outfilename;

  //=========================================================================================================//
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH);
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
    TIME(QUDA_solver solver(mu,1));

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



    
    for(int isource = startSource; isource < numSourcePositions; isource++){

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      std::vector<std::vector<int>> mpi2_twopt = sourcemomentumList_twopt.uniq_p(0);
      momList list_mpi2_twopt(1,{mpi2_twopt,},{0,});

      std::vector<std::vector<int>> mpf1_twopt = sourcemomentumList_twopt.uniq_p(1);
      momList list_mpf1_twopt(1,{mpf1_twopt,},{0,});

      std::vector<std::vector<int>> mpi2_threept = sourcemomentumList_threept.uniq_p(0);
      momList list_mpi2_threept(1,{mpi2_threept,},{0,});

      std::vector<std::vector<int>> mpf1_threept = sourcemomentumList_threept.uniq_p(1);
      momList list_mpf1_threept(1,{mpf1_threept,},{0,});

      site& source = sourcePositions[isource];
      site source_reduction=site({0,0,0,sourcePositions[isource][DIM_T]});

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear, bool finalize) {
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

      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string;
      threep_filename = given_threep_filename + src_string;
      free(src_string);
      
      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propDN;

      PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1_twopt);
      corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP");

      PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1_twopt);
      corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0");


      PLEGMA_ScattCorrelator<float> corrD(sourcePositions[isource], list_mpf1_twopt);
      corrD.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_delta,"D");


#if 1
      { // Whithin this scope we keep track also of the propagator non smeared on the sink

        PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE);
	PLEGMA_Propagator<float> propDN_SL(tSinks.size()>0 ? BOTH:NONE);

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
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_source_nucleon, propDN, propUP, propDN));
          //PLEGMA_printf("Nucleon T2 reduction\n");
          TIME(reductionsT2N.T2(glist_source_nucleon, glist_source_nucleon, propDN, propUP, propDN));
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

        std::vector<std::vector<int>> mpi2_threept = sourcemomentumList_threept.uniq_p(0);

        auto &momentum_i2 =  mpi2_threept[0];
        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList_threept.extract(momentum_i2, 0);
        momList filtered_sourcemomentumList_2pt = sourcemomentumList_twopt.extract(momentum_i2, 0);
        std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList_2pt.uniq_p(3);
        std::vector<std::vector<int>> mpi2_filt  ;
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);

        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

        PLEGMA_ScattCorrelator<float> corrD1(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrD2(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrD3(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrD4(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrD5(sourcePositions[isource], list_mpi2ptot);

        corrD1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_delta, "D1");
        corrD2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_delta, "D2");
        corrD3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_delta, "D3");
        corrD4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_delta, "D4");
        corrD5.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_delta, "D5");

        {


          PLEGMA_ScattCorrelator<float> reductionsT1(source_reduction, sourcemomentumList_twopt.uniq_p(1));
          PLEGMA_ScattCorrelator<float> reductionsT2(source_reduction, sourcemomentumList_twopt.uniq_p(1));


          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));
          TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));

          TIME(corrD.D_diagrams( reductionsT1, reductionsT2 ));

          PLEGMA_ScattCorrelator<float> reductionsT1D(source_reduction, mptot_filt);
          PLEGMA_ScattCorrelator<float> reductionsT2D(source_reduction, mptot_filt);

          TIME(reductionsT1D.T1(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));
          TIME(corrD1.convertTreductiontoDiagram( reductionsT1D, -1, false, true, true ));

          TIME(reductionsT2D.T2(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));
          TIME(corrD2.convertTreductiontoDiagram( reductionsT2D, -1, false, true, true ));

          TIME(reductionsT2D.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
          TIME(corrD3.convertTreductiontoDiagram( reductionsT2D, -1, false, true, true ));

          TIME(reductionsT1D.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propDN));
          TIME(corrD4.convertTreductiontoDiagram( reductionsT1D, -1, false, true, true ));

          TIME(reductionsT1D.T1(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
          TIME(corrD5.convertTreductiontoDiagram( reductionsT1D, -1, false, true, true ));

        }
          
        //write D
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_D";

        TIME( corrD.apply_phase() );
        TIME( corrD.apply_sign("D") );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        TIME( corrD1.writeHDF5(outfilename) );
        TIME( corrD2.writeHDF5(outfilename) );
        TIME( corrD3.writeHDF5(outfilename) );
        TIME( corrD4.writeHDF5(outfilename) );
        TIME( corrD5.writeHDF5(outfilename) );




        std::vector<int> filter={0,0,0};
        momList filtered_sourcemomentumList_pi20 = sourcemomentumList_threept.extract(filter, 0);


     
	for(size_t its = 0; its < tSinks.size(); its++){
	  int tsinkMtsource = tSinks[its];
	  if(tsinkMtsource >= HGC_totalL[3])
	    PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	  int signPer = (tsinkMtsource+source[3]) >= HGC_totalL[3] ? -1 : +1;
	  int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3]; 

          //Correlators for storing up and down insertion between the nucleon

          PLEGMA_ScattCorrelator<float> corrUp(source,  filtered_sourcemomentumList_pi20, tsinkMtsource+1 );
          PLEGMA_ScattCorrelator<float> corrDn(source,  filtered_sourcemomentumList_pi20, tsinkMtsource+1 );

          corrUp.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_delta, gammas_insertion, "M"+prOrNt+"Up");

          corrDn.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_delta, gammas_insertion, "M"+prOrNt+"Dn");



	  // 3D propagators at t_sink
	  PLEGMA_Propagator3D<float> propUP3D;
	  PLEGMA_Propagator3D<float> propDN3D;
	  PLEGMA_Gauge3D<double> smearedGauge3D_sink;
	  propUP3D.absorb(propUP, global_fixSinkTime);
	  propDN3D.absorb(propDN, global_fixSinkTime);

	  smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

	  WHICHPARTICLE nucleon = get_particle(prOrNt); 
	  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
          for (int alpha=0;alpha<2; ++alpha){
            for (int beta=0; beta<2; ++beta){

	    auto computeThreep = [&](double run_mu, PLEGMA_Propagator3D<float>& prop1, PLEGMA_Propagator3D<float>& prop2, int signProps, PLEGMA_Propagator<float> &propF, std::string fl) {
             /* if(not computed_light) {
                  TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
                  TIME(computePropagator(propDN, propDN_SL, -mu_ud, LIGHT, nsmearGauss, false));
                  propUP3D.absorb(propUP, global_fixSinkTime);
                  propDN3D.absorb(propDN, global_fixSinkTime);
                  computed_light = true;
                }*/

	      // ensuring mu positive
	      if(mu != run_mu) {
		updateOptions(LIGHT);
		mu = run_mu;
		solver.UpdateSolver();
	      }

              for(int i_pf1=0; i_pf1<mpf1_threept.size(); ++i_pf1){

                auto &momentum_f1 =  mpf1_threept[i_pf1];

                std::string filename = threep_filename + "_P" +std::to_string(alpha)+std::to_string(beta) + "_dt" + std::to_string(tsinkMtsource) + "_" + fl + "pf_x"+std::to_string(momentum_f1[0])+"_y"+std::to_string(momentum_f1[1])+"_z"+std::to_string(momentum_f1[2])+".h5";
                if(access( filename.c_str(), F_OK ) != -1) {
                  PLEGMA_printf("File %s already exists. Skipping...", filename.c_str());
                  return;
                }

                momList filtered_sinkList = sourcemomentumList_threept.extract(momentum_f1, 1);

                for (int sigma=0; sigma<3; sigma++){

	          PLEGMA_Propagator<float> seqProp;
				     
	          for(int nu = 0 ; nu < 4 ; nu++){

		    for(int c2 = 0 ; c2 < 3 ; c2++){
		      PLEGMA_Vector<double> vectorInOut;
                      {
		        PLEGMA_Vector3D<double> vectorAuxD1,vectorAuxD2;
		        PLEGMA_Vector3D<float> vectorAuxF;
		        if(&prop1 != &prop2){
		          vectorAuxF.seqSourceNucleonDelta(prop1, prop2, get_projector(alpha, beta), nucleon, nu, c2, sigma);
                        } 
		        else
                        {
		          vectorAuxF.seqSourceNucleonDelta(prop1, get_projector(alpha, beta), nucleon, nu, c2, sigma);
                        }
                             
                        // put a momentum in the sink
                        vectorAuxF.mulMomentumPhases(momentum_f1,-1);//At the sink the momenta should be -
                                                                     //We have initially -
                                                                     //The we have conjugation twice
                                                                     //so eventually we have -

                        vectorAuxF.conjugate();
		        vectorAuxF.apply_gamma(G5);
		        vectorAuxD1.copy(vectorAuxF);
		        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nsmearGauss, alphaGauss));
		        vectorInOut.absorb(vectorAuxD2, global_fixSinkTime);
                        //norm=vectorInOut.norm();
                        //PLEGMA_printf("After Norm calculating %e\n",norm);
                        //fflush(stdout);

		      }
		      double norm = vectorInOut.norm();
		      vectorInOut.scale(1/norm);

                      if (get_projector(alpha,beta)>7)
                      {
                        PLEGMA_Vector<double> vectorAuxD;
                        int sgn=run_mu/fabs(run_mu);
                        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,sgn));
                        TIME(vectorInOut.copy(vectorAuxD));
                      }

		      TIME(solver.solve(vectorInOut, vectorInOut));

                      if (get_projector(alpha,beta)>7){
                        PLEGMA_Vector<double> vectorAuxD;
                        int sgn=run_mu/(fabs(run_mu));
                        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,sgn));
                        TIME(vectorInOut.copy(vectorAuxD));
                      }

		      vectorInOut.scale(norm);
		      PLEGMA_Vector<float> vectorAuxF;
		      vectorAuxF.copy(vectorInOut);
		      seqProp.absorb(vectorAuxF, nu, c2);
		    }
                  }

                  seqProp.apply_gamma(G5);
	          seqProp.conjugate();

                  std::vector<std::vector<int>> mpc = filtered_sinkList.uniq_p(3);
                  momList list_mpc(1,{mpc,},{0,});

                  PLEGMA_ScattCorrelator<float> corr( source, list_mpc, tsinkMtsource+1);				     
	  
	          // LOCAL contractions
                  if (get_projector(alpha,beta)<8){
                    TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
                  }
                  else{
                    TIME(corr.contractNucleonThrp_local(seqProp, propF, 0, gammas));
                  }
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;

                  if (fl=="up"){
                    corrUp.absorbSourceSinkSpinMom(corr, alpha, beta, sigma, i_pf1 );
                  }
                  else{
                    corrDn.absorbSourceSinkSpinMom(corr, alpha, beta, sigma, i_pf1 );
                  }

                } //isospin

              } //momentum

	    };
	    if(nucleon == PROTON) {
	      TIME(computeThreep(-mu_ud, propUP3D, propDN3D, +1, propUP_SL, "up"));
	      TIME(computeThreep( mu_ud, propUP3D, propUP3D, -1, propDN_SL, "dn"));
	    } else {
	      TIME(computeThreep( mu_ud, propDN3D, propUP3D, -1, propDN_SL, "dn"));
	      TIME(computeThreep(-mu_ud, propDN3D, propDN3D, +1, propUP_SL, "up"));
	    }
	  }//beta
	}//alpha
        if (nucleon==PROTON){
           outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_dt" + std::to_string(tsinkMtsource)+"_protonup";
        }
        else{
           outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_dt" + std::to_string(tsinkMtsource)+"_neutronup";
        }
        TIME(corrUp.writeHDF5(outfilename));

        if (nucleon==PROTON){
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_dt" + std::to_string(tsinkMtsource)+"_protondn";
        }
        else{
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_dt" + std::to_string(tsinkMtsource) +"_neutrondn";
        }
        TIME(corrDn.writeHDF5(outfilename));

      }//tSink
    }//smearing
#endif
#if 0
      propUP.rotateToPhysicalBase_device(+1);
      propDN.rotateToPhysicalBase_device(-1);
      propUP.applyBoundaries_device(source[3]);
      propDN.applyBoundaries_device(source[3]);

      {
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	TIME(corr.contractMesons(propUP, propDN));
	
	char *dset1, *dset2;
	asprintf(&dset1, "twop_mesons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
	asprintf(&dset2, "twop_mesons_d[%+1.1e]u[%+1.1e]", -1*mu_ud, mu_ud);
	corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	free(dset1); free(dset2);
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractBaryons(propUP, propDN));
	THREAD(corr.writeFile(twop_filename, corr_file_format));
      }
      
      // Storing only the smaller and then computing on the fly the other
      int nSmaller = std::min(mu_s.size(),mu_c.size());
      char cSmaller = (nSmaller==(int)mu_s.size()) ? 's' : 'c';
      
      PLEGMA_Propagator<float> none(NONE);
      PLEGMA_Propagator<float> propS[nSmaller];
      for(int ismall=0; ismall < nSmaller; ismall++) {
	for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_factor[i] = 1;
	double run_mu = (cSmaller=='s') ? mu_s[ismall] : mu_c[ismall];
	int nsmear = (cSmaller=='s') ? nsmearGauss_s : nsmearGauss_c;
	TIME(computePropagator(propS[ismall], none, run_mu, (cSmaller=='s') ? STRANGE : CHARM, nsmear, true));
      }
      
      int nLarger = (cSmaller!='s') ? mu_s.size() : mu_c.size();
      if(nLarger > 0) {
	PLEGMA_Propagator<float> propL;
	for(int ilarge=0; ilarge < nLarger; ilarge++) {
	  double run_mu = (cSmaller!='s') ? mu_s[ilarge] : mu_c[ilarge];
	  int nsmear = (cSmaller!='s') ? nsmearGauss_s : nsmearGauss_c;
	  TIME(computePropagator(propL, none, run_mu, (cSmaller!='s') ? STRANGE : CHARM, nsmear, true));

	  if(nSmaller>0) {
	    for(int ismall=0; ismall < nSmaller; ismall++) {
	      PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? propS[ismall] : propL;
	      PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? propS[ismall] : propL;
	      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	      bool only_st = (ismall>0 && cSmaller=='s') || (ilarge>0 && cSmaller!='s');
	      bool only_ch = (ismall>0 && cSmaller=='c') || (ilarge>0 && cSmaller!='c');
#ifdef PLEGMA_UDSC_BARYONS
	      TIME(corr.contractBaryonsUDSC(propUP, propDN, propST, propCH, only_st, only_ch));
	      char * group;
	    
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]s[%+1.1e]c[%+1.1e]%s%s", mu_ud, -1*mu_ud, mu_s[cSmaller=='s'? ismall:ilarge], mu_c[cSmaller=='c'? ismall:ilarge],
		       only_st ? "_only-s" : "", only_ch ? "_only-c" : "");
	      corr.setGroups(group);
	      free(group);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
	      TIME(corr.contractMesons(propST, propCH));
	      char *dset1, *dset2;
	      asprintf(&dset1, "twop_mesons_s[%+1.1e]c[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], mu_c[cSmaller=='c'? ismall:ilarge]);
	      asprintf(&dset2, "twop_mesons_c[%+1.1e]s[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], mu_s[cSmaller=='s'? ismall:ilarge]);
	      corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));

	      if(!only_ch) {
		TIME(corr.contractMesons(propUP, propST));
		asprintf(&dset1, "twop_mesons_u[%+1.1e]s[%+1.1e]", mu_ud, mu_s[cSmaller=='s'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]u[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      
		TIME(corr.contractMesons(propDN, propST));
		asprintf(&dset1, "twop_mesons_d[%+1.1e]s[%+1.1e]", -1*mu_ud, mu_s[cSmaller=='s'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]d[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], -1*mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      }

	      if(!only_st) {
		TIME(corr.contractMesons(propUP, propCH));
		asprintf(&dset1, "twop_mesons_u[%+1.1e]c[%+1.1e]", mu_ud, mu_c[cSmaller=='c'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]u[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      
		TIME(corr.contractMesons(propDN, propCH));
		asprintf(&dset1, "twop_mesons_d[%+1.1e]c[%+1.1e]", -1*mu_ud, mu_c[cSmaller=='c'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]d[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], -1*mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      }
	    }
	  } else {
	    PLEGMA_Propagator<float> none(NONE);
	    PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? none : propL;
	    PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? none : propL;
	    PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	    bool only_st = (ilarge>0 && cSmaller!='s');
	    bool only_ch = (ilarge>0 && cSmaller!='c');
#ifdef PLEGMA_UDSC_BARYONS
	    TIME(corr.contractBaryonsUDSC(propUP, propDN, propST, propCH, only_st, only_ch));
	    char * group;

	    if(cSmaller=='s') {
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]c[%+1.1e]%s", mu_ud, -1*mu_ud, mu_c[ilarge], only_ch ? "_only-c" : "");
	    } else {
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]s[%+1.1e]%s", mu_ud, -1*mu_ud, mu_s[ilarge], only_st ? "_only-s" : "");
	    }
	    corr.setGroups(group);
	    free(group);
	    THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
	    if(!only_ch && !only_st) {
	      char *dset1, *dset2;
	      TIME(corr.contractMesons(propUP, (cSmaller=='s') ? propCH : propST));
	      if(cSmaller=='s') {
		asprintf(&dset1, "twop_mesons_u[%+1.1e]c[%+1.1e]", mu_ud, mu_c[ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]u[%+1.1e]", mu_c[ilarge], mu_ud);
	      } else {
		asprintf(&dset1, "twop_mesons_u[%+1.1e]s[%+1.1e]", mu_ud, mu_s[ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]u[%+1.1e]", mu_s[ilarge], mu_ud);
	      }

	      corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));
	      
	      TIME(corr.contractMesons(propDN, (cSmaller=='s') ? propCH : propST));
	      if(cSmaller=='s') {
		asprintf(&dset1, "twop_mesons_d[%+1.1e]c[%+1.1e]", -1*mu_ud, mu_c[ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]d[%+1.1e]", mu_c[ilarge], -1*mu_ud);
	      } else {
		asprintf(&dset1, "twop_mesons_d[%+1.1e]s[%+1.1e]", -1*mu_ud, mu_s[ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]d[%+1.1e]", mu_s[ilarge], -1*mu_ud);
	      }
	      corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));
	    }
	  }
	}
      } else {
#ifdef PLEGMA_UDSC_BARYONS
	PLEGMA_Propagator<float> none(NONE);
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	TIME(corr.contractBaryonsUDSC(propUP, propDN, none, none));
	char * group;
	
	asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
      }
#endif
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();
  return 0;
}
