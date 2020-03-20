
#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()


extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "time-dilution"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  bool timedilution;
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
//  std::string outfile_V3;
//  std::string outfile_V2;
//  std::string outfile_V4;
//  std::string path_V="";
//  std::string path_P="";

  HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outPropUP", "Path for saving the up propagator used", verbosity, outfile_upS);
  HGC_options->set("outPropDN", "Path for saving the up propagator used", verbosity, outfile_dnS);
  HGC_options->set("outPropSeq", "Path for saving the sequential propagator used", verbosity, outfile_SEQ);
//  HGC_options->set("outV3", "Path for saving the result of V3_reduction", verbosity, outfile_V3);
//  HGC_options->set("outV2", "Path for saving the result of V3_reduction", verbosity, outfile_V2);
//  HGC_options->set("outV4", "Path for saving the result of V3_reduction", verbosity, outfile_V4);
//  HGC_options->set("loadVector", "Path for loading V", verbosity, path_V);
//  HGC_options->set("loadProp", "Path for loading P", verbosity, path_P);

  //=========================================================================================================//
  initializePLEGMA();
  double start_time, tmp_time;
  {
    
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
    gauge.calculatePlaq();

    //Reading the momentum lists
    PLEGMA_printf("###Momentum list read from : %s", pathListMomenta.c_str());
    momList sourcemomentumList(3,pathListMomenta,{1,2});
    PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList.size());
    if(sourcemomentumList.empty())
      PLEGMA_error("momentumList empty");


    //List of gammas
    std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4,CG_1_G_4_G_5,CG_2_G_4_G_5,CG_3_G_4_G_5};
    std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4,CG_1_G_4_G_5,CG_2_G_4_G_5,CG_3_G_4_G_5};
    std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5,C,CG_5_G_4,CG_4};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5,C,CG_5_G_4,CG_4};
    std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID,G_5};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID,G_5};

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
    plaqQuda();

    PLEGMA_Gauge<double> contractGauge;
    contractGauge.copy(gauge);

    // Smearing
    PLEGMA_Gauge<double> smearedGauge;
    smearedGauge.APEsmearing(contractGauge, nsmearAPE, alphaAPE, 3);
    PLEGMA_printf("Plaquette after smearing:\n");
    smearedGauge.calculatePlaq();
   
    // apply boundary conditions since is needed for the covariant derivative
    applyBoundaryConditions(contractGauge,true);
    //ensuring mu positive
    QUDA_solver solver(mu);

    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);


    //Computing time-diluted stochastic propagators and stochastic source

    PLEGMA_Vector<float> vectorStoc_source(BOTH);
    PLEGMA_Vector<float> vectorStoc_propag(BOTH);
    PLEGMA_Vector<float> vectorStoc_source_arch(BOTH); 
    PLEGMA_Vector<double> vectorAuxD1(BOTH);
    PLEGMA_Vector<double> vectorAuxD2(BOTH);
    PLEGMA_Vector<double> vectorInOut;
    PLEGMA_printf("Build vector from scratch\n");
    //Note that we replace the f1<-f2 DN propagator with a stochastic one
    //in two steps actually
    //DN(x_f1 <- x_f2 ) = \phihat(x_f2)(x_f1)\xi^{dagger}(x_f2)(x_f2)
    //where x_f2 is the source
    //      x_f1 is the sink
    //so \phihat(x_f2)(x_f1) is the x_f1 coordinate of the stochastic 
    //propagator created at x_f2 for the down quark
    //=gamma_5*U(x_f2 <- x_f1)^dagger*gamma_5
    //=gamma_5*\xi(x_f1)(x_f1)*\phi(x_f2)(x_f1)^dagger*gamma_5
    //Here we compute phi and xi
    int nroots=4;
    //Step(1) Creating the time-diluted stochastic source
    vectorStoc_source.randInit(1234);
    vectorStoc_source.stochastic_Z(nroots);
    

    //Step(2) Smearing all the time slice
    vectorAuxD1.copy(vectorStoc_source);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss );
    //Save the smeared source in order to reuse it for oet.
    vectorStoc_source_arch.copy(vectorAuxD2);

    //We rotate the source to the physical basis
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);

    //In vectorAuxD2 we store the results for the inversion
    vectorAuxD2.scale(0.0);
    
    if (timedilution){
      PLEGMA_printf("#piNdiagramms: Full time dilution is turned on\n");
      for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
        //Step(3) pick out a particular timeslice from the source
        vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
        //Step(4) Solve
        solver.solve(vectorInOut, vectorInOut);
        //Step(5) absorbing the particular timeslice to a 4d vector
        vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
      }
    }
    else{
      PLEGMA_printf("#piNdiagramms: No time dilution is used n stochastic propagators\n");
      vectorInOut.copy(vectorAuxD1);
      solver.solve(vectorInOut, vectorInOut);
      vectorAuxD2.copy(vectorInOut);
    } 

    //vectorStoc_source_arch.writeLIME(outfile_V+"globalTfulltimedilution_source");
    //vectorStoc_source_arch.writeHDF5(outfile_V+"globalTfulltimedilution_source");
    vectorAuxD1.copy(vectorStoc_source_arch);
    vectorAuxD1.apply_gamma5();
    vectorStoc_source.copy(vectorAuxD1);

    //Step(6) We rotate back the propagator to the physical basis
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);

    //Step(7) Smearing all the time slice in the propagator
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss );
    
    //vectorAuxD1.writeLIME(outfile_V+"globalTfulltimedilution_propagator");
    //vectorAuxD1.writeHDF5(outfile_V+"globalTfulltimedilution_propagator");
    vectorStoc_propag.copy(vectorAuxD1);
    vectorStoc_propag.apply_gamma5();

    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      int sequential_time_source=sourcePositions[isource][3];

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);

    
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH);
      PLEGMA_Propagator<float> propDN(BOTH);
      PLEGMA_Propagator<float> propUPDN(BOTH);

      //to measure the smearing time
      tmp_time = 0;


      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      PLEGMA_printf("Read propagator from:\n");
      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<float> vectorRead(BOTH);
        std::string spin=std::to_string(isc/3);
        std::string col=std::to_string(isc%3);
        vectorRead.readFile("/cyclamen/home/fpittler/runs/plegma_develop_all_momenta_tuning_new/data/propagator/propagator_up_s"+spin+"_c"+col,LIME_FORMAT);
        vectorRead.load();
        propUP.absorb(vectorRead, isc/3, isc%3);
       }

      PLEGMA_printf("Read propagator from:\n");
      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<float> vectorRead(BOTH);
        std::string spin=std::to_string(isc/3);
        std::string col=std::to_string(isc%3);
        vectorRead.readFile("/cyclamen/home/fpittler/runs/plegma_develop_all_momenta_tuning_new/data/propagator/propagator_dn_s"+spin+"_c"+col,LIME_FORMAT);
        vectorRead.load();
        propDN.absorb(vectorRead, isc/3, isc%3);
       }


      /*
      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float> vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;
        //Create the source
        vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);

        //Smearing on the source
        start_time = MPI_Wtime();
        vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        //Rotation to the physical basis
        vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);       

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        solver.solve(vectorAuxD, vectorAuxD);

        //Rotation to the physical basis
        vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1);

        //Smearing at the sink
        start_time = MPI_Wtime();
        vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorAuxD);
        propUP.absorb(vectorAuxF, isc/3, isc%3);
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
            vectorAuxPrint.writeLIME(outfile_upS+"_s"+spin+"_c"+col);
            vectorAuxPrint.writeHDF5(outfile_upS+"_s"+spin+"_c"+col);
          }
        }


      // ensuring mu negative
      if(mu>0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float> vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;
        //(1 step) creating the point source
        vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);

        start_time = MPI_Wtime();
        //(2 step) doing the gaussian smearing on the source
        vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);

        //(3 step) rotation to the physical basis
        vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1);

        tmp_time += MPI_Wtime()-start_time;
        //(4 step) doing the inversion
        PLEGMA_printf("Going to invert DN for component %d\n", isc);
        solver.solve(vectorAuxD, vectorAuxD);

        start_time = MPI_Wtime();
        //(5 step) rotating to the physical base
        vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1);

        //(6 step) doing the smearing on the propagator
        vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorAuxD);

        propDN.absorb(vectorAuxF, isc/3, isc%3);
      }


      if(outfile_dnS!="")
        {
          PLEGMA_printf("Save propagator for the d quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propDN,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_dnS+"_s"+spin+"_c"+col);
            vectorAuxPrint.writeHDF5(outfile_dnS+"_s"+spin+"_c"+col);

          }
        }
*/
      std::vector<int> mom={0,0,0};
      
      // int source[4]={sourcePositions[isource][0],
      //                sourcePositions[isource][1],
      //                sourcePositions[isource][2],
      //                sourcePositions[isource][3]};

      site source=site({0,0,0,sourcePositions[isource][3]});
      std::string outfilename;

      //D diagram
      {
	std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
	momList list_mtot(1,{mtot,},{0,});
	PLEGMA_ScattCorrelator<float> corrD(sourcePositions[isource], list_mtot);

	//initialize diagram
	corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,"D");
      
	PLEGMA_ScattCorrelator<float> reductionsT1(source, mtot);
	PLEGMA_ScattCorrelator<float> reductionsT2(source, mtot);

	TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));
	//reductionsT1.writeHDF5("T1sourceforD");

	TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));
	//reductionsT2.writeHDF5("T2sourceforD");

	//write D
	outfilename="Ddiagramm_Antonino" ;
	
	PLEGMA_printf("DEBUG: 5th -- start D_diagram\n");
	TIME(corrD.D_diagramms( reductionsT1, reductionsT2 ));
	PLEGMA_printf("DEBUG: 5th -- start apply phase to D diagram\n");
	TIME( corrD.apply_phase() );
	PLEGMA_printf("DEBUG: 5th -- apply bounds to D diagram\n");
	TIME( corrD.applyBoundaryConditions( true ) );
	PLEGMA_printf("DEBUG: 5th -- write D diagram\n");
	TIME( corrD.writeHDF5(outfilename) );

      }
      

      PLEGMA_printf("DEBUG: write D diagram done\n");

      
      //N diagram
      std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
      momList list_mpf1(1,{mpf1,},{0,});
      PLEGMA_ScattCorrelator<float> corrN(sourcePositions[isource], list_mpf1 );

      //initialize diagram
      TIME(corrN.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"));
      PLEGMA_printf("DEBUG: corrN initialized\n");
      
      //diagramm_nucleon.setSource(source);
      PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
      PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
      
      TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
      //reductionsT1.writeHDF5("T1sourceforN");

      TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
      //reductionsT2.writeHDF5("T2sourceforN");

      //write N
      outfilename = "Ndiagramm_Antonino";
      TIME(corrN.N_diagramms( reductionsT1N, reductionsT2N ));
      TIME(corrN.apply_phase());
      TIME(corrN.applyBoundaryConditions( true ));
      TIME(corrN.writeHDF5( outfilename ));
    
      //P diagram
      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
      momList list_mpi2(1,{mpi2,},{0,});
      PLEGMA_ScattCorrelator<float> corrP(sourcePositions[isource], list_mpi2);
      corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }


      PLEGMA_Vector<float> vectortmp1;
      PLEGMA_Vector<float> vectortmp2;
          
      PLEGMA_Vector<float> stochastic_source_spin_diluted_momzero; 
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;

      //Using the already generated stochastic source and project it to a time-slice
      //Smearing was already performed
      //Creating oet time-slice source
      vectortmp1.absorbTimeslice(vectorStoc_source_arch, sequential_time_source);
      //Transforming to physical base
      vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);
      stochastic_source_spin_diluted_momzero.dilutespin(vectortmp2,0);

      for (int spinindex=0; spinindex<4; ++spinindex){
        PLEGMA_Vector<double> vectorAuxD;

        //tmp_time += MPI_Wtime()-start_time;       
        //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
        //Doing the zero momentum stochastic propagator with spin dilution
        vectorInOut.copy(stochastic_source_spin_diluted_momzero);
        //Doing the inversion
        solver.solve(vectorInOut, vectorInOut);
        //Rotate back immediately to the physical basis
        vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);
        //Gaussian smearing of the propagator
        vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        stochastic_propagator_momzero[spinindex].copy(vectorInOut);
        //tmp_time += MPI_Wtime()-start_time;
        //stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+"propagator_zero_momentum"+std::to_string(spinindex));
        //stochastic_propagator_momzero[spinindex].writeHDF5(outfile_V+"propagator_zero_momentum"+std::to_string(spinindex));
        if (spinindex<3){
          vectortmp1.dilutespindisplace(stochastic_source_spin_diluted_momzero,spinindex+1,spinindex);
          stochastic_source_spin_diluted_momzero.copy(vectortmp1);
        }
      }

      //We first have a loop over all unique the source meson momentum p_i2 
      for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	auto &momentum_i2 =  mpi2[i_mpi2];
	//List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);
 
	// 4pt diagrams
	
	PLEGMA_ScattCorrelator<float> corrB1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrB2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW3(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW4(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ3(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ4(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList);
	

	//T diagrams
	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
	std::vector<std::vector<int>> mpi2_filt;
	mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});
	PLEGMA_ScattCorrelator<float> corrT(sourcePositions[isource], list_mpi2ptot);
	corrT.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "T");	

	//initialize diagrams
	
	corrB1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "B1");
	corrB2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "B2");
	corrW1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W1");
	corrW2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W2");
	corrW3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W3");
	corrW4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W4");
	corrZ1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z1");
	corrZ2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z2");
	corrZ3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z3");
	corrZ4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z4");
	corrM.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "M");
	
	
        PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source, filtered_sourcemomentumList.uniq_p(2));

        //Loop over the different gamma structure for the source meson
        for (int i_gamma_i2=0; i_gamma_i2<glist_source_meson.size(); ++i_gamma_i2) {
	  GAMMAS_SCATT gamma_i2 = glist_source_meson[i_gamma_i2];
          // Computing sequential propagators f1 <- i_2 <- i_1 
          // so the sequential source source time is fixed
          // and the momentum is also fixed to be momentum_i2

          //smearing the 3D propagators
          
          PLEGMA_Propagator3D<float> propDN3D;      
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            vectorAuxF.absorb(propDN,isc/3, isc%3);
            vectorAuxD.copy(vectorAuxF);
            start_time = MPI_Wtime();

            //Performing the smearing
            vectorAuxD2.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
            tmp_time += MPI_Wtime()-start_time;

            //Perform multiplication with gamma_i2
            vectorAuxD2.apply_gamma_scatt(gamma_i2);

            //Perform rotation to the physical basis
            vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);

            
            vectorAuxF.copy(vectorAuxD);
            propDN3D.absorb(vectorAuxF, sequential_time_source, isc/3, isc%3);
          }
 
          propDN3D.mulMomentumPhases(momentum_i2,1);

          //Computing sequential propagators UD T_fii with insertion
          //gamma_i2 and momentum SinkMom
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorInOut;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD;
          
            vectorAuxF.absorb(propDN3D, sequential_time_source, isc/3, isc%3);
            vectorInOut.copy(vectorAuxF);
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            solver.solve(vectorInOut, vectorInOut);
            start_time = MPI_Wtime();

            //performing rotation to physical base
            vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);

            //performing smearing
            vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
            tmp_time += MPI_Wtime()-start_time;

            vectorAuxF.copy(vectorInOut);
            propUPDN.absorb(vectorAuxF, isc/3, isc%3);
          }


          if(outfile_SEQ!="")
          {
             PLEGMA_printf("Save sequential propagator for the ud \n");
             PLEGMA_Vector<float> vectorAuxPrint(BOTH);
             for(int isc = 0 ; isc < 12 ; isc++){
               std::string spin=std::to_string(isc/3);
               std::string col=std::to_string(isc%3);
               vectorAuxPrint.absorb(propUPDN,isc/3,isc%3);
               vectorAuxPrint.unload();
               vectorAuxPrint.writeLIME(outfile_SEQ+"_s"+spin+"_c"+col);
               vectorAuxPrint.writeHDF5(outfile_SEQ+"_s"+spin+"_c"+col);
             }
          }
          /*
          PLEGMA_printf("Smearing time %lf sec\n",tmp_time);
          PLEGMA_printf("Read propagator from:\n");
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<float> vectorRead(BOTH);
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);
            vectorRead.readFile("/cyclamen/home/fpittler/runs/plegma_develop_all_momenta_tuning_new/data/propagator/propagator_updn_s"+spin+"_c"+col,LIME_FORMAT);
            vectorRead.load();
            propUPDN.absorb(vectorRead, isc/3, isc%3);
          }*/
      


          //Compute triangle diagramms          
          
          PLEGMA_ScattCorrelator<float> reductionsT1triangle(source,  mptot_filt);
         
          PLEGMA_ScattCorrelator<float> reductionsT3triangle(source,  mptot_filt);

          PLEGMA_ScattCorrelator<float> reductionsT5triangle(source,  mptot_filt);
          TIME(reductionsT1triangle.T1(glist_source_nucleon, glist_sink_delta, propUPDN, propUP  , propUP));
          //reductionsT1triangle.writeHDF5("T1sourceforT");
          TIME(reductionsT3triangle.T1(glist_source_nucleon, glist_sink_delta, propUP  , propUPDN, propUP));
          //reductionsT3triangle.writeHDF5("T3sourceforT");
          TIME(reductionsT5triangle.T2(glist_source_nucleon, glist_sink_delta, propUP  , propUP, propUPDN));
          //reductionsT5triangle.writeHDF5("T5sourceforT");
	  
	  //Compute Diagram T 

          TIME(corrT.T_diagramms(reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, i_gamma_i2));
          //Compute Diagram B1 and B2 
	  
          TIME(reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUPDN));
          //reductionsV3.writeHDF5("V3sourceforB1");

          TIME(reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUP));
          //reductionsV2.writeHDF5("V2sourceforB1");

	  TIME(corrB1.B_diagramms(reductionsV3, reductionsV2, i_gamma_i2, 1));
	  
	  TIME(corrB2.B_diagramms(reductionsV3, reductionsV2, i_gamma_i2, 2));
          
          //Compute Diagram W1,W2
          
          TIME(reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUP));
          //reductionsV3.writeHDF5("V3sourceforW12");
          TIME(reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUPDN));
          //reductionsV2.writeHDF5("V2sourceforW12");

	  //PLEGMA_printf("DEBUG: start W1_diagram\n");
          TIME(corrW1.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 1));
	  //PLEGMA_printf("DEBUG: start W2_diagram\n");
	  TIME(corrW2.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 2));
	  //PLEGMA_printf("DEBUG: finish W2\n");
          
          //Compute Diagram W3,W4
          //
          TIME(reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUPDN, propUP));
          reductionsV2.writeHDF5("V2sourceforW34");

          TIME(corrW3.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 3));
	  TIME(corrW4.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 4));
	  
       } //loop over gamma i2
         
	  
       //Producing spin diluted stochastic propagators for diagram Z1,Z2,Z3,Z4
              
       std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momp_i2;

       std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted = {
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2))
         };

       std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted = {
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1))
       };


       PLEGMA_Vector<float> stochastic_source_spin_diluted_momp_i2; 
       //Using the already generated stochastic source and project it to a time-slice
       //Smearing was already performed
       //Creating oet time-slice source
       vectortmp1.absorbTimeslice(vectorStoc_source_arch, sequential_time_source);
       //Transforming to physical base
       vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);
       
       //Multiplying by the appropriate momentum phase

       std::vector<int> tmp_4Dmom= momentum_i2 ; 
       tmp_4Dmom.push_back(0);
       vectortmp2.mulMomentumPhases(tmp_4Dmom,-1);
       stochastic_source_spin_diluted_momp_i2.dilutespin(vectortmp2,0);
 
       
       for (int spinindex=0; spinindex<4; ++spinindex){
         PLEGMA_Vector<double> vectorAuxD;

         //tmp_time += MPI_Wtime()-start_time;       
         //stochastic_source_spin_diluted_momp_i2.writeLIME(outfile_V+"source_fini_momentum"+std::to_string(spinindex));
       
 
         vectorInOut.copy(stochastic_source_spin_diluted_momp_i2);
         //Doing the inversion
         solver.solve(vectorInOut, vectorInOut);

         //Rotate back immediately to the physical basis
         vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);

         //performing smearing
         vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);

         //Saving the propagator
         stochastic_propagator_momp_i2[spinindex].copy(vectorInOut);

         //tmp_time += MPI_Wtime()-start_time;         
         //stochastic_propagator_momp_i2[spinindex].writeLIME(outfile_V+"propagator_fini_momentum"+std::to_string(spinindex));
         
         if (spinindex<3){
           vectortmp1.dilutespindisplace(stochastic_source_spin_diluted_momp_i2,spinindex+1,spinindex);
           stochastic_source_spin_diluted_momp_i2.copy(vectortmp1);
         }

       }

       //Diagram Z1,Z2
       std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
       
       for (int i=0; i< 4; ++i){
         TIME(reductionsV3_diluted[i].V3( stochastic_propagator_momp_i2[i], gamma_5_t_sinkmeson, propUP));
         //reductionsV3_diluted[i].writeHDF5("V3sourceforZ"+std::to_string(i));

         TIME(reductionsV2_diluted[i].V4( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
         //reductionsV2_diluted[i].writeHDF5("V4sourceforZ"+std::to_string(i));
       }
       TIME(corrZ1.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 1 ));
       TIME(corrZ2.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 2 ));
 
      
       //Diagram Z3,Z4	
       for (int i=0; i< 4; ++i){

         TIME(reductionsV2_diluted[i].V2( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
         //reductionsV2_diluted[i].writeHDF5("V2sourceforZ"+std::to_string(i));

       }

       TIME(corrZ3.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 3 ));
       TIME(corrZ4.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 4 ));

       //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       
       outfilename = "Pdiagramm_Antonino";
       TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momp_i2, i_mpi2));
       
       //     TIME(corrT.writeHDF5("temporary_Tdiagramm" ));
       TIME(corrM.M_diagramms( corrN, stochastic_propagator_momzero, stochastic_propagator_momp_i2 ));
	
       //   TIME(corrT.writeHDF5("temporary2_Tdiagramm" ));

       //write everything
       //## T
       outfilename = "Tdiagramm_Antonino";
       
       TIME(corrT.apply_phase());
       TIME(corrT.applyBoundaryConditions( true ));

       TIME(corrT.writeHDF5( outfilename ));

       
       //## B
       outfilename = "Bdiagramm_Antonino";
       TIME(corrB1.apply_phase());
       TIME(corrB1.applyBoundaryConditions( true ));
       TIME(corrB1.writeHDF5( outfilename ));
       TIME(corrB2.apply_phase());
       TIME(corrB2.applyBoundaryConditions( true ));
       TIME(corrB2.writeHDF5( outfilename ));


//       TIME(corrT.writeHDF5("temporary3_Tdiagramm" ));
       //## W
       outfilename = "Wdiagramm_Antonino";
       TIME(corrW1.apply_phase());
       TIME(corrW1.applyBoundaryConditions( true ));
       TIME(corrW1.writeHDF5( outfilename ));
       TIME(corrW2.apply_phase());
       TIME(corrW2.applyBoundaryConditions( true ));
       TIME(corrW2.writeHDF5( outfilename ));
       TIME(corrW3.apply_phase());
       TIME(corrW3.applyBoundaryConditions( true ));
       TIME(corrW3.writeHDF5( outfilename ));
       TIME(corrW4.apply_phase());
       TIME(corrW4.applyBoundaryConditions( true ));
       TIME(corrW4.writeHDF5( outfilename ));
       //## Z

//       TIME(corrT.writeHDF5("temporary4_Tdiagramm" ));
       outfilename = "Zdiagramm_Antonino";
       TIME(corrZ1.apply_phase());
       TIME(corrZ1.applyBoundaryConditions( true ));
       TIME(corrZ1.writeHDF5( outfilename ));
       TIME(corrZ2.apply_phase());
       TIME(corrZ2.applyBoundaryConditions( true ));
       TIME(corrZ2.writeHDF5( outfilename ));
       TIME(corrZ3.apply_phase());
       TIME(corrZ3.applyBoundaryConditions( true ));
       TIME(corrZ3.writeHDF5( outfilename ));
       TIME(corrZ4.apply_phase());
       TIME(corrZ4.applyBoundaryConditions( true ));
       TIME(corrZ4.writeHDF5( outfilename ));

       //## M
       outfilename = "Mdiagramm_Antonino";
       //TIME(corrM.writeHDF5( "mdiagrammwithoutphase" ));
       TIME(corrM.apply_phase());
       TIME(corrM.applyBoundaryConditions( true ));
       TIME(corrM.writeHDF5( outfilename ));
      
      }//loop over unique set of momenta for p_i2
       
      //write P
      TIME(corrP.writeHDF5( outfilename ));

    } //loop over source position

  } 
  finalize();
  
  return 0;
}
