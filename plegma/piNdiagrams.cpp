#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

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
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
//  std::string outfile_V3;
//  std::string outfile_V2;
//  std::string outfile_V4;
//  std::string path_V="";
//  std::string path_P="";
  
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
    momList sourcemomentumList(pathListMomenta);


    //List of gammas
    std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_1};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_1};
    std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_ext_sink={ID};
    std::vector<GAMMAS_SCATT> glist_ext_source={ID};

    
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
    //In vectorAuxD2 we store the results
    vectorAuxD2.scaleVector(0.0);
    
    if (timedilutionflagstring.compare("on")==0){
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

    vectorStoc_source_arch.writeLIME(outfile_V+"globalTfulltimedilution_source");
    vectorAuxD1.copy(vectorStoc_source_arch);
    vectorAuxD1.apply_gamma5();
    vectorStoc_source.copy(vectorAuxD1);

    //Step(6) We rotate back the propagator to the physical basis
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);
    //Step(7) Smearing all the time slice in the propagator
    vectorAuxD1.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss );
    
    vectorAuxD1.writeLIME(outfile_V+"globalTfulltimedilution_propagator");
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

      std::vector<int> mom={0,0,0};
      PLEGMA_ScattCorrelator<float> diagramm(MOMENTUM_SPACE, mom);

      int source[4]={sourcePositions[isource][0],
                     sourcePositions[isource][1],
                     sourcePositions[isource][2],
                     sourcePositions[isource][3]};

      diagramm.setSource(source);


      PLEGMA_ScattCorrelator<float> reductionsT1(MOMENTUM_SPACE, sourcemomentumList.uniq_p(3));
      PLEGMA_ScattCorrelator<float> reductionsT2(MOMENTUM_SPACE, sourcemomentumList.uniq_p(3));


      reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propUP, propUP);
      reductionsT1.writeASCII("T1sourceforD");

      reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propUP, propUP);
      reductionsT2.writeASCII("T2sourceforD");


      std::string outfilename="Ddiagramm_Antonino" ;
      diagramm.D_diagramms( reductionsT1, reductionsT2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, outfilename);

      //N diagram
      PLEGMA_ScattCorrelator<float> diagramm_nucleon(MOMENTUM_SPACE, mom);
      diagramm_nucleon.setSource(source);
      
      reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP);
      reductionsT1.writeASCII("T1sourceforN");

      reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP);
      reductionsT2.writeASCII("T2sourceforN");


      outfilename = "Ndiagramm_Antonino";
      diagramm_nucleon.N_diagramms( reductionsT1, reductionsT2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, outfilename);


      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }


      //We first have a loop over all unique the source meson momentum p_i2 
      for (auto momentum_i2 : sourcemomentumList.uniq_p(0)) {


        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList(sourcemomentumList.extract(momentum_i2, 0));

        PLEGMA_ScattCorrelator<float> reductionsV2(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(2));

        //Loop over the different gamma structure for the source meson
        for (auto gamma_i2 : glist_source_meson) {
         
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
          //gamma4 and momentum SinkMom
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
             }
          }

          PLEGMA_printf("Smearing time %lf sec\n",tmp_time);

          //Compute triangle diagramms          
          

          PLEGMA_ScattCorrelator<float> reductionsT1triangle(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(3));
         
          PLEGMA_ScattCorrelator<float> reductionsT3triangle(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(3));

          PLEGMA_ScattCorrelator<float> reductionsT5triangle(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(3));
          reductionsT1triangle.T1(glist_source_nucleon, glist_sink_nucleon, propUPDN, propUP  , propUP);
          reductionsT1triangle.writeHDF5("T1sourceforT");
          reductionsT3triangle.T1(glist_source_nucleon, glist_sink_nucleon, propUP  , propUPDN, propUP);
          reductionsT3triangle.writeHDF5("T3sourceforT");
          reductionsT5triangle.T2(glist_source_nucleon, glist_sink_nucleon, propUP  , propUPDN, propUP);
          reductionsT1triangle.writeHDF5("T5sourceforT");

          outfilename="Tdiagramm_Antonino";
          diagramm.T_diagramms(filtered_sourcemomentumList, reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, gamma_i2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired,  outfilename);


          //Compute Diagram B1 and B2 
          reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUPDN);
          reductionsV3.writeHDF5("V3sourceforB1");

          reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUP);
          reductionsV2.writeHDF5("V2sourceforB1");
 
          outfilename="Bdiagramm_Antonino" ;
          diagramm.B_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_ext_source, glist_ext_sink, gamma_i2, glist_source_nucleon, outfilename);

     
          //diagramm.B_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 1 );
          //diagramm.writeHDF5("B1Diagramm");

          //diagramm.B_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 2 );
          //diagramm.writeHDF5("B2Diagramm");

          //Compute Diagram W1,W2
           
          //PLEGMA_ScattCorrelator<float> diagramW(MOMENTUM_SPACE, sinkMom_Meson);

          reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUP);
          reductionsV3.writeHDF5("V3sourceforW12");
          reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUPDN);
          reductionsV2.writeHDF5("V2sourceforW12");

          outfilename= "Wdiagramm_Antonino";
          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_ext_source, glist_ext_sink, gamma_i2, glist_source_nucleon, outfilename, 1);

          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_ext_source, glist_ext_sink, gamma_i2, glist_source_nucleon, outfilename, 2);

          //diagramW.W_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 1 );
          //diagramW.writeHDF5("W1Diagramm");

          //diagramW.W_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 2 );
          //diagramW.writeHDF5("W2Diagramm");

          //Compute Diagram W3,W4
          //reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUP); maybe does not have to be recomputed
          reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUPDN, propUP);
          reductionsV2.writeHDF5("V2sourceforW34");

          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_ext_source, glist_ext_sink, gamma_i2, glist_source_nucleon, outfilename, 3);

          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_ext_source, glist_ext_sink, gamma_i2, glist_source_nucleon, outfilename, 4);

          //diagramW.W_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 3 );
          //diagramW.writeHDF5("W3Diagramm");

          //diagramW.W_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 4 );
          //diagramW.writeHDF5("W4Diagramm");          
          //
       } //loop over gamma i2
         
       std::string outfilename="Zdiagramm_Antonino" ;

      
       //Producing spin diluted stochastic propagators for diagram Z1,Z2,Z3,Z4
              
       std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
       std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momp_i2;

       std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted = {
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(2))
         };

       std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted = {
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Nucleon)
       };


       PLEGMA_Vector<float> stochastic_source_spin_diluted_momp_i2; 
       PLEGMA_Vector<float> stochastic_source_spin_diluted_momzero; 
       PLEGMA_Vector<float> vectortmp1;
       PLEGMA_Vector<float> vectortmp2;
          
       //Using the already generated stochastic source and project it to a time-slice
       //Smearing was already performed
       //Creating oet time-slice source
       vectortmp1.absorbTimeslice(vectorStoc_source_arch, sequential_time_source);
       //Transforming to physical base
       vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);
       
 
       stochastic_source_spin_diluted_momzero.dilutespin(vectortmp2,0);

       //Multiplying by the appropriate momentum phase

       std::vector<int> tmp_4Dmom= momentum_i2 ; 
       tmp_4Dmom.push_back(0);
       vectortmp2.mulMomentumPhases(tmp_4Dmom,1);
       stochastic_source_spin_diluted_momp_i2.dilutespin(vectortmp2,0);
 
       
       for (int spinindex=0; spinindex<4; ++spinindex){
         PLEGMA_Vector<double> vectorAuxD;

         //tmp_time += MPI_Wtime()-start_time;       
         stochastic_source_spin_diluted_momp_i2.writeLIME(outfile_V+"source_fini_momentum"+std::to_string(spinindex));
       
         //tmp_time += MPI_Wtime()-start_time;
         stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));
 
         vectorInOut.copy(stochastic_source_spin_diluted_momp_i2);
         //Doing the inversion
         solver.solve(vectorInOut, vectorInOut);

         //Rotate back immediately to the physical basis
         vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);

         //performing smearing
         vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);

         //Saving the propagator
         stochastic_propagator_momp_i2[spinindex].copy(vectorAuxD);

         //tmp_time += MPI_Wtime()-start_time;         
         stochastic_propagator_momp_i2[spinindex].writeLIME(outfile_V+"propagator_fini_momentum"+std::to_string(spinindex));

         //Doing the same for zero momentum
         vectorInOut.copy(stochastic_source_spin_diluted_momzero);

         //Doing the inversion
         solver.solve(vectorInOut, vectorInOut);

         //Rotate back immediately to the physical basis
         vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);

         //Gaussian smearing of the propagator
         vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
         
         stochastic_propagator_momzero[spinindex].copy(vectorInOut);
         //tmp_time += MPI_Wtime()-start_time;

         stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+"propagator_zero_momentum"+std::to_string(spinindex));
         
         if (spinindex<3){
           vectortmp1.dilutespindisplace(stochastic_source_spin_diluted_momp_i2,spinindex+1,spinindex);
           vectortmp2.dilutespindisplace(stochastic_source_spin_diluted_momzero,spinindex+1,spinindex);
           stochastic_source_spin_diluted_momp_i2.copy(vectortmp1);
           stochastic_source_spin_diluted_momzero.copy(vectortmp2);
         }

       }      

       //Diagram Z1,Z2
       std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);       
       std::vector<GAMMAS_SCATT>  sourcemeson_t_gamma_5=apply_gamma5_scatt_gamma(glist_source_meson,RIGHT);
       for (int i=0; i< 4; ++i){
         reductionsV3_diluted[i].V3( stochastic_propagator_momp_i2[i], gamma_5_t_sinkmeson, propUP);
         reductionsV3_diluted[i].writeHDF5("V3sourceforZ"+std::to_string(i));

         reductionsV2_diluted[i].V4( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP);
         reductionsV2_diluted[i].writeHDF5("V4sourceforZ"+std::to_string(i));
       }

       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_ext_source, glist_ext_sink, sourcemeson_t_gamma_5, glist_source_nucleon, outfilename, 1);


       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_ext_source, glist_ext_sink, sourcemeson_t_gamma_5, glist_source_nucleon, outfilename, 2);
 
      
       //Diagram Z3,Z4	
       for (int i=0; i< 4; ++i){
         reductionsV2_diluted[i].V2( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP);
         reductionsV2_diluted[i].writeHDF5("V2sourceforZ"+std::to_string(i));

       }

       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_ext_source, glist_ext_sink, sourcemeson_t_gamma_5, glist_source_nucleon,  outfilename, 3);


       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_ext_source, glist_ext_sink, sourcemeson_t_gamma_5, glist_source_nucleon,  outfilename, 4);



       //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       
       std::vector<GAMMAS_SCATT> glist_sourcemeson_g5 = apply_gamma5_scatt_gamma(glist_source_meson,RIGHT);       
       std::vector<GAMMAS_SCATT> glist_sinkmeson_g5 = apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
       outfilename = "Pdiagramm_Antonino";
       diagramm.P_diagramms( momentum_i2, glist_source_meson, glist_sink_meson, stochastic_propagator_momzero, stochastic_propagator_momp_i2, outfilename);
       
       outfilename = "Mdiagramm_Antonino";
       diagramm.M_diagramms( sourcemomentumList, filtered_sourcemomentumList, diagramm_nucleon, glist_sourcemeson_g5, glist_sinkmeson_g5, stochastic_propagator_momzero, stochastic_propagator_momp_i2, outfilename);


      }//loop over unique set of momenta for p_i2

    } //loop over source position

  } 
  finalize();
  
  return 0;
}
