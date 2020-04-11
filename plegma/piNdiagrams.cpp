
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

using namespace plegma;
using namespace quda;

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "time-dilution","nstochSamples"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  bool timedilution;
  int n_stochastic_samples;
  int nroots=4;
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
  std::string outdiagramPrefix="";
  std::string outfile_V3;
  std::string outfile_V2;
  std::string outfile_V4;

  HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outPropUP", "Path for saving the up propagator used", verbosity, outfile_upS);
  HGC_options->set("outPropDN", "Path for saving the dn propagator used", verbosity, outfile_dnS);
  HGC_options->set("outPropSeq", "Path for saving the sequential propagator used", verbosity, outfile_SEQ);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
  HGC_options->set("outV3", "Path for saving the result of V2_reduction", verbosity, outfile_V3);
  HGC_options->set("outV2", "Path for saving the result of V3_reduction", verbosity, outfile_V2);
  HGC_options->set("outV4", "Path for saving the result of V4_reduction", verbosity, outfile_V4);

  //=========================================================================================================//
  initializePLEGMA();
  {

    //Storing only the smeared gauge
    PLEGMA_Gauge<double> smearedGauge;

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
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));


    //Get the confnumber for latfile
    std::istringstream iss(latfile);
    std::string tokenforfilename;
    while (std::getline(iss, tokenforfilename, '/')){}
    std::istringstream iss2(tokenforfilename);
    std::string confnumber;
    while (std::getline(iss2, confnumber, '.')){}

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
    std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_source_meson_T={ID};

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);


    //Computing time-diluted stochastic propagators and stochastic source

    std::vector<PLEGMA_Vector<double>*> stochastic_sources;
    std::vector<PLEGMA_Vector<double>*> stochastic_propags;

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.push_back(new PLEGMA_Vector<double>(HOST));
      stochastic_propags.push_back(new PLEGMA_Vector<double>(HOST));
    }

    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(1234);

    //PLEGMA_Vector<float> vectorStoc_source(BOTH);
    //PLEGMA_Vector<float> vectorStoc_propag(BOTH);
    //PLEGMA_Vector<double> vectorInOut;
    PLEGMA_printf("Start producing stochastic vectors and propagators\n");
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
    //Producing the stochastic source
    {
      
      PLEGMA_Vector<double> vectorAuxD1(BOTH);
      PLEGMA_Vector<double> vectorAuxD2(BOTH);
      PLEGMA_Vector<double> vectorInOut;
      PLEGMA_Vector<double> vectorSource(BOTH);
      vectorSource.randInit(1234);
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
        vectorAuxD1.copy(vectorSource);
        vectorAuxD1.apply_gamma5();

        vectorAuxD1.unload();
        stochastic_sources[i]->copy(vectorAuxD1,HOST);
        vectorAuxD1.load();
        vectorAuxD1.copy(vectorSource);
     
        //Step(3) Smearing all the time slice
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));
 
        //Step(4) We rotate the source to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //In vectorAuxD2 we store the results for the inversion
        vectorAuxD2.scale(0.0);
    
        if (timedilution){
          PLEGMA_printf("#piNdiagramms: Full time dilution is turned on\n");
          for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
            //Step(5) pick out a particular timeslice from the source
            vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
            //Step(6) Solve
            TIME(solver.solve(vectorInOut, vectorInOut));
            //Step(7) absorbing the particular timeslice to a 4d vector
            vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
          }
        }
        else{
          PLEGMA_printf("#piNdiagramms: No time dilution is used n stochastic propagators\n");
          vectorInOut.copy(vectorAuxD1);
          TIME(solver.solve(vectorInOut, vectorInOut));
          vectorAuxD2.copy(vectorInOut);
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
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber);
        }

        vectorAuxD2.apply_gamma5();

        //Step(9) Save the propagator to the host memory
        vectorAuxD2.unload();
        stochastic_propags[i]->copy(vectorAuxD2,HOST);
        vectorAuxD2.load();
        
      } //loop over the stochastic samples

    }

    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, sourcePositions[isource][DIM_T]);


      int sequential_time_source=sourcePositions[isource][DIM_T];

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);

      char *ssource;
      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource; 
      free(ssource);
  
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH);
      PLEGMA_Propagator<float> propDN(BOTH);
      PLEGMA_Propagator<float> propUPDN(BOTH);

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }
    
      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float>  vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;
        { // Smearing the source
          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
        }

        //Rotation to the physical basis
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));       

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        //Rotation to the physical basis
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1));

        //Smearing at the sink
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

        vectorAuxF.copy(vectorAuxD);
        propUP.absorb(vectorAuxF, isc/3, isc%3);
      }
      /*
      if(outfile_upS!="")
        {
          PLEGMA_printf("Save propagator for the up quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propUP,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            //vectorAuxPrint.writeHDF5(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
          }
        }
       */
      // ensuring mu negative
      if(mu>0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float>  vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;

        { // Smearing the source
          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
        }

        //(3 step) rotation to the physical basis
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1));

        //(4 step) doing the inversion
        PLEGMA_printf("Going to invert DN for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        //(5 step) rotating to the physical base
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1));

        //(6 step) doing the smearing on the propagator
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

        vectorAuxF.copy(vectorAuxD);

        propDN.absorb(vectorAuxF, isc/3, isc%3);
      }

      /*
      if(outfile_dnS!="")
        {
          PLEGMA_printf("Save propagator for the d quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propDN,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_dnS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            //vectorAuxPrint.writeHDF5(outfile_dnS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);

          }
        }*/

      std::vector<int> mom={0,0,0};
      
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
	outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_D";
	
	TIME( corrD.D_diagramms( reductionsT1, reductionsT2 ));
	TIME( corrD.apply_phase() );
	TIME( corrD.applyBoundaryConditions( true ) );
	TIME( corrD.writeHDF5(outfilename) );

      }
      
      //T diagram piN sink

      {
	momList list_pf1pf2comb = sourcemomentumList.extract({0,0,0}, 0);
        PLEGMA_ScattCorrelator<float> corrT_piNsink(sourcePositions[isource], list_pf1pf2comb);

        corrT_piNsink.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "T1"); 
 
        PLEGMA_ScattCorrelator<float> reductionsV2(source, list_pf1pf2comb.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source, list_pf1pf2comb.uniq_p(2));
 
        for (int i=0; i<n_stochastic_samples; ++i){
          PLEGMA_Vector<float> stochastic_propagator;
          PLEGMA_Vector<float> stochastic_source;

          stochastic_propagator.copy(*stochastic_propags[i],HOST);
          stochastic_source.copy(*stochastic_sources[i],HOST);

          stochastic_propagator.load();
          stochastic_source.load();

          TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propUP));
          //reductionsV3.writeHDF5("V3sourceforTpiNsink"+std::to_string(i));

          TIME(reductionsV2.V2( stochastic_source,     glist_sink_nucleon, propUP, propUP));
          //reductionsV2.writeHDF5("V2sourceforTpiNsink"+std::to_string(i));

          TIME(corrT_piNsink.T_diagramms_piNsink(reductionsV3, reductionsV2, true));

        }                
        outfilename=outdiagramPrefix+confnumber+sourcepositiontext+"_TpiNsink";

        TIME(corrT_piNsink.apply_phase());
        TIME(corrT_piNsink.applyBoundaryConditions( true ));
        TIME(corrT_piNsink.normalize_nstoch(n_stochastic_samples));
        TIME(corrT_piNsink.writeHDF5( outfilename ));

      }

      
      //N diagram

      outfilename=outdiagramPrefix+confnumber+sourcepositiontext+"_N";

      std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
      momList list_mpf1(1,{mpf1,},{0,});
      PLEGMA_ScattCorrelator<float> corrN(sourcePositions[isource], list_mpf1 );

      //initialize diagram
      TIME(corrN.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"));
      
      //Computing T reductions+recombination
      { 
        PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
        PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
      
        TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
        //reductionsT1N.writeHDF5("T1sourceforN");

        TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
        //reductionsT2N.writeHDF5("T2sourceforN");

        //write N
        TIME(corrN.N_diagramms( reductionsT1N, reductionsT2N ));
        TIME(corrN.apply_phase());
        TIME(corrN.applyBoundaryConditions( true ));
        TIME(corrN.writeHDF5(outfilename));

      }


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

      //We draw a different random vector for every source position
      vectorStoc_source_oet.stochastic_Z(nroots);
      
      //Store zero momentum oet propagators
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
      {
         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;          
 
         vectortmp1.absorbTimeslice(vectorStoc_source_oet, sequential_time_source);

         {  // Smearing the source
            
            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.absorb(vectortmp1, sourcePositions[isource][DIM_T]);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectortmp1.absorb(vector2,sourcePositions[isource][DIM_T]);
         }
 
         //Transforming to physical base
         vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);
 
          //Dilution     
         vectortmp1.dilutespin(vectortmp2,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorStoc_source_oet.copy(vectortmp1);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.copy(vectorStoc_source_oet);
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
           stochastic_propagator_momzero[spinindex].copy(vectortmp2);
           //stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+confnumber+"propagator_"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){
             vectortmp1.dilutespindisplace(vectorStoc_source_oet,spinindex+1,spinindex);
             vectorStoc_source_oet.copy(vectortmp1);
           }
         }
         
         vectortmp1.dilutespindisplace(vectorStoc_source_oet,0,3);
         vectorStoc_source_oet.copy(vectortmp1);
      }

      //We first have a loop over all unique the source meson momentum p_i2 
      for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	auto &momentum_i2 =  mpi2[i_mpi2];
	//List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);


        std::string pi2x=std::to_string(momentum_i2[0]);
        std::string pi2y=std::to_string(momentum_i2[1]);
        std::string pi2z=std::to_string(momentum_i2[2]);

 
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

            //Performing the smearing
            // Smearing the source
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source );
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorAuxD2.absorb(vector2,sourcePositions[isource][DIM_T]);
            }

            //Perform multiplication with gamma_i2
            vectorAuxD2.apply_gamma_scatt(gamma_i2);

            //Perform rotation to the physical basis
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1));

            
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
            TIME(solver.solve(vectorInOut, vectorInOut));

            //performing rotation to physical base
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));

            //performing smearing
            TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));

            vectorAuxF.copy(vectorInOut);
            propUPDN.absorb(vectorAuxF, isc/3, isc%3);
          }

          /*
          if(outfile_SEQ!="")
          {
             PLEGMA_printf("Save sequential propagator for the ud \n");
             PLEGMA_Vector<float> vectorAuxPrint(BOTH);
             for(int isc = 0 ; isc < 12 ; isc++){
               std::string spin=std::to_string(isc/3);
               std::string col=std::to_string(isc%3);
               vectorAuxPrint.absorb(propUPDN,isc/3,isc%3);
               vectorAuxPrint.unload();
               vectorAuxPrint.writeLIME(outfile_SEQ+confnumber+sourcepositiontext+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+spin+"_c"+col);
               //vectorAuxPrint.writeHDF5(outfile_SEQ+"_s"+spin+"_c"+col);
             }
          }*/
           


          //Compute triangle diagramms          
          
          PLEGMA_ScattCorrelator<float> reductionsT1triangle(source,  mptot_filt);
         
          PLEGMA_ScattCorrelator<float> reductionsT3triangle(source,  mptot_filt);

          PLEGMA_ScattCorrelator<float> reductionsT5triangle(source,  mptot_filt);
          TIME(reductionsT1triangle.T1(glist_source_nucleon, glist_sink_delta, propUPDN, propUP  , propUP));
          //reductionsT1triangle.writeHDF5("T1sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
          TIME(reductionsT3triangle.T1(glist_source_nucleon, glist_sink_delta, propUP  , propUPDN, propUP));
          //reductionsT3triangle.writeHDF5("T3sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
          TIME(reductionsT5triangle.T2(glist_source_nucleon, glist_sink_delta, propUP  , propUP, propUPDN));
          //reductionsT5triangle.writeHDF5("T5sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
	  
	  //Compute Diagram T 
          TIME(corrT.T_diagramms(reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, i_gamma_i2));


          for (int i=0; i<n_stochastic_samples; ++i){
            //Compute Diagram B1 and B2
            PLEGMA_Vector<float> stochastic_propagator;
            PLEGMA_Vector<float> stochastic_source;

            stochastic_propagator.copy(*stochastic_propags[i],HOST);
            stochastic_source.copy(*stochastic_sources[i],HOST);

            stochastic_propagator.load();
            stochastic_source.load();
 
	  
            TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propUPDN));
            //reductionsV3.writeHDF5("V3sourceforB1_sample"+std::to_string(i)+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z);

            TIME(reductionsV2.V2( stochastic_source,     glist_sink_nucleon, propUP, propUP));
            //reductionsV2.writeHDF5("V2sourceforB1_sample"+std::to_string(i)+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z);


	    TIME(corrB1.B_diagramms(reductionsV3, reductionsV2, i_gamma_i2, 1, true));
	  
	    TIME(corrB2.B_diagramms(reductionsV3, reductionsV2, i_gamma_i2, 2, true));
          
            //Compute Diagram W1,W2
          
            TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propUP));
            //reductionsV3.writeHDF5("V3sourceforW12_sample"+std::to_string(i)+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
            TIME(reductionsV2.V2( stochastic_source,     glist_sink_nucleon, propUP, propUPDN));
            //reductionsV2.writeHDF5("V2sourceforW12_sample"+std::to_string(i)+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z);

            TIME(corrW1.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 1, true));
	    TIME(corrW2.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 2, true));
          
            //Compute Diagram W3,W4
          
            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUPDN, propUP));
            //reductionsV2.writeHDF5("V2sourceforW34_sample"+std::to_string(i)+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z);

            TIME(corrW3.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 3, true));
	    TIME(corrW4.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 4, true));

          } //loop over stochastic samples
	  
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



       {
          PLEGMA_Vector<double> vectortmp1;
          PLEGMA_Vector<double> vectortmp2;
          PLEGMA_Vector<double> vectorSource_finite_mom;

          //Multiplying by the appropriate momentum phase
          vectorSource_finite_mom.copy(vectorStoc_source_oet);
          std::vector<int> tmp_4Dmom= momentum_i2 ; 
          tmp_4Dmom.push_back(0);
          vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);
          
          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
       
            for (int spinindex=0; spinindex<4; ++spinindex){
       
              vectortmp1.copy(vectorSource_finite_mom);
              //Doing the inversion
              TIME(solver.solve(vectortmp1, vectortmp1));

              //Rotate back immediately to the physical basis
              TIME(vectortmp2.rotateToPhysicalBasis(vectortmp1,+1));

              //performing smearing
              TIME(vectortmp1.gaussianSmearing(vectortmp2, smearedGauge, nsmearGauss, alphaGauss));

              //Saving the propagator
              stochastic_propagator_momp_i2[spinindex].copy(vectortmp1);
              //stochastic_propagator_momp_i2[spinindex].writeLIME(outfile_V+confnumber+"propagator_"+sourcepositiontext+"mompi2_"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(spinindex));
         
              if (spinindex<3){
                vectortmp1.dilutespindisplace(vectorSource_finite_mom,spinindex+1,spinindex);
                vectorSource_finite_mom.copy(vectortmp1);
              }
            }
          }
       }
         
       //Diagram Z1,Z2
       std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
       
       for (int i=0; i< 4; ++i){
         if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
  
           TIME(reductionsV3_diluted[i].V3( stochastic_propagator_momp_i2[i], gamma_5_t_sinkmeson, propUP));
           //reductionsV3_diluted[i].writeHDF5("V3sourceforZ_pi2"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(i));

         }
         else{
           TIME(reductionsV3_diluted[i].V3( stochastic_propagator_momzero[i], gamma_5_t_sinkmeson, propUP));
           //reductionsV3_diluted[i].writeHDF5("V3sourceforZ_pi2"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(i));
         }

         TIME(reductionsV2_diluted[i].V4( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
         //reductionsV2_diluted[i].writeHDF5("V4sourceforZ_pi2"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(i));
       }

       TIME(corrZ1.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 1 ));
       TIME(corrZ2.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 2 ));
 
      
       //Diagram Z3,Z4	
       for (int i=0; i< 4; ++i){

         TIME(reductionsV2_diluted[i].V2( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
         //reductionsV2_diluted[i].writeHDF5("V2sourceforZ_pi2"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(i));

       }

       TIME(corrZ3.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 3 ));
       TIME(corrZ4.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 4 ));

       //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
         TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momp_i2, i_mpi2));
       
       
         TIME(corrM.M_diagramms( corrN, stochastic_propagator_momzero, stochastic_propagator_momp_i2 ));
       }
       else{
         TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momzero, i_mpi2));


         TIME(corrM.M_diagramms( corrN, stochastic_propagator_momzero, stochastic_propagator_momzero ));
       }
	

       //write everything
       //## T
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
     
       TIME(corrT.apply_phase());
       TIME(corrT.applyBoundaryConditions( true ));

       TIME(corrT.writeHDF5(outfilename));

       
       //## B
       
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
       TIME(corrB1.apply_phase());
       TIME(corrB1.applyBoundaryConditions( true ));
       TIME(corrB1.normalize_nstoch(n_stochastic_samples));
       TIME(corrB1.writeHDF5( outfilename ));
       TIME(corrB2.apply_phase());
       TIME(corrB2.applyBoundaryConditions( true ));
       TIME(corrB2.normalize_nstoch(n_stochastic_samples));
       TIME(corrB2.writeHDF5( outfilename ));


       //## W
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";

       TIME(corrW1.apply_phase());
       TIME(corrW1.applyBoundaryConditions( true ));
       TIME(corrW1.normalize_nstoch(n_stochastic_samples));
       TIME(corrW1.writeHDF5(outfilename));
       TIME(corrW2.apply_phase());
       TIME(corrW2.applyBoundaryConditions( true ));
       TIME(corrW2.normalize_nstoch(n_stochastic_samples));
       TIME(corrW2.writeHDF5(outfilename));
       TIME(corrW3.apply_phase());
       TIME(corrW3.applyBoundaryConditions( true ));
       TIME(corrW3.normalize_nstoch(n_stochastic_samples));
       TIME(corrW3.writeHDF5(outfilename));
       TIME(corrW4.apply_phase());
       TIME(corrW4.applyBoundaryConditions( true ));
       TIME(corrW4.normalize_nstoch(n_stochastic_samples));
       TIME(corrW4.writeHDF5(outfilename));
       //## Z

       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

       TIME(corrZ1.apply_phase());
       TIME(corrZ1.applyBoundaryConditions( true ));
       TIME(corrZ1.writeHDF5( outfilename ));
       TIME(corrZ2.apply_phase());
       TIME(corrZ2.applyBoundaryConditions( true ));
       TIME(corrZ2.writeHDF5( outfilename  ));
       TIME(corrZ3.apply_phase());
       TIME(corrZ3.applyBoundaryConditions( true ));
       TIME(corrZ3.writeHDF5( outfilename ));
       TIME(corrZ4.apply_phase());
       TIME(corrZ4.applyBoundaryConditions( true ));
       TIME(corrZ4.writeHDF5( outfilename ));

       //## M
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";

       //TIME(corrM.writeHDF5( "mdiagrammwithoutphase" ));
       TIME(corrM.apply_phase());
       TIME(corrM.applyBoundaryConditions( true ));
       TIME(corrM.writeHDF5( outfilename ));
      
      }//loop over unique set of momenta for p_i2
       
      //write P

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrP.writeHDF5( outfilename ));

    } //loop over source position

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      stochastic_propags.pop_back();
    }


  } 
  finalize();
  
  return 0;
}
