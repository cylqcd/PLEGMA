#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename"};
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
    std::vector<GAMMAS> glist_source_nucleon={G4};
    std::vector<GAMMAS> glist_sink_nucleon={G4};
    std::vector<GAMMAS> glist_sink_meson={ONE};
    std::vector<GAMMAS> glist_source_meson={ONE};
    
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
        //vectorAuxD.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;
        vectorInOut.copy(vectorAuxD);

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        solver.solve(vectorInOut, vectorInOut);

        //Smearing at the sink
        start_time = MPI_Wtime();
        //vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorInOut);
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
        vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);

        start_time = MPI_Wtime();
        //vectorAuxD.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;
        
        vectorInOut.copy(vectorAuxD);

        PLEGMA_printf("Going to invert DN for component %d\n", isc);
        solver.solve(vectorInOut, vectorInOut);

        start_time = MPI_Wtime();
        //vectorInOut.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorInOut);

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
          }
        }
      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      //Computing stochastic propagators and stochastic source

      PLEGMA_Vector<float> vectorStoc_source(BOTH);//FP: Do we really need both here?
      PLEGMA_Vector<float> vectorStoc_propag(BOTH);
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
      QUDA_solver solver(mu);
      vectorStoc_source.randInit(1234);
      vectorStoc_source.stochastic_Z(nroots);

      vectorInOut.copy(vectorStoc_source);
      //Smearing the source
      //vectorInOut.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
      
      solver.solve(vectorInOut, vectorInOut);

      //Smearing the propagator    
      //vectorInOut.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
      vectorStoc_propag.copy(vectorInOut);
      vectorStoc_propag.writeLIME(outfile_V+"globalTpropagator_zeromomentum");

      vectorStoc_source.apply_gamma5();
      vectorStoc_source.writeLIME(outfile_V+"globalTsource_zeromomentum");

      //We first have a loop over all unique the source meson momentum p_i2 
      for (auto momentum_i2 : sourcemomentumList.uniq_p(0)) {


        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList(sourcemomentumList.extract(momentum_i2, 0));

        PLEGMA_ScattCorrelator<float> reductionsV2(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(MOMENTUM_SPACE, filtered_sourcemomentumList.uniq_p(2));

        std::vector<int> mom={0,0,0};
        PLEGMA_ScattCorrelator<float> diagramm(MOMENTUM_SPACE, mom);


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
            vectorAuxF.absorb(propDN,isc/3, isc%3);
            vectorAuxD.copy(vectorAuxF);
            start_time = MPI_Wtime();
            //vectorAuxD.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
            tmp_time += MPI_Wtime()-start_time;
            vectorAuxD.apply_gamma(gamma_i2);
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
            solver.solve(vectorInOut, vectorInOut);
            start_time = MPI_Wtime();
            //vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
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

          //Compute Diagram B1 and B2 
          reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUPDN);
          reductionsV3.writeHDF5("V3sourceforB1");

          reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUP);
          reductionsV2.writeHDF5("V2sourceforB1");
 
          std::string outfilename="Bdiagramm_Antonino" ;
          diagramm.B_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, gamma_i2, glist_source_nucleon, outfilename);

     
          //diagramm.B_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 1 );
          //diagramm.writeHDF5("B1Diagramm");

          //diagramm.B_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 2 );
          //diagramm.writeHDF5("B2Diagramm");

          //Compute Diagram W1,W2
           
          //PLEGMA_ScattCorrelator<float> diagramW(MOMENTUM_SPACE, sinkMom_Meson);

          reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUP);
          reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUPDN);


          outfilename= "Wdiagramm_Antonino";
          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, gamma_i2, glist_source_nucleon, outfilename, 1);

          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, gamma_i2, glist_source_nucleon, outfilename, 2);

          //diagramW.W_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 1 );
          //diagramW.writeHDF5("W1Diagramm");

          //diagramW.W_diagramms(glist_source_nucleon, reductionsV3, reductionsV2, 2 );
          //diagramW.writeHDF5("W2Diagramm");

          //Compute Diagram W3,W4
          //reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUP); maybe does not have to be recomputed
          reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUPDN, propUP);

          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, gamma_i2, glist_source_nucleon, outfilename, 3);

          diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, gamma_i2, glist_source_nucleon, outfilename, 4);

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
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Meson),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Meson),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Meson),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Meson)
         };

       std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted = {
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Nucleon),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Nucleon),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Nucleon),
           PLEGMA_ScattCorrelator<float>(MOMENTUM_SPACE, sinkMom_Nucleon)
       };


       PLEGMA_Vector<float> stochastic_source_spin_diluted_momp_i2; 
       PLEGMA_Vector<float> stochastic_source_spin_diluted_momzero; 
       PLEGMA_Vector<float> vectortmp1;
       PLEGMA_Vector<float> vectortmp2;
          
      
       vectorStoc_source.randInit(4321);
       vectortmp1.absorbTimeslice(vectorStoc_source, sequential_time_source);
       vectortmp2.copy(vectortmp1);

       //Multiplying by the appropriate momentum phase
       //vectortmp1 <-- source with finite momentum
       //vectortmp2 <-- source wuth zero momentum
       vectortmp1.mulMomentumPhases(momentum_i2,1);

       for (int spinindex=0; spinindex<3; ++spinindex){

         stochastic_source_spin_diluted_momp_i2.dilutespin(vectortmp1, spinindex);
         //Ideally doing the smearing on the source only on a 3D vector
         //stochastic_source_spin_diluted_momp_i2.gaussianSmearing(stochastic_source_spin_diluted_momp_i2, smearedGauge, nsmearGauss, alphaGauss);
         //tmp_time += MPI_Wtime()-start_time;       
         stochastic_source_spin_diluted_momp_i2.writeLIME(outfile_V+"source_fini_momentum"+std::to_string(spinindex));
       

         stochastic_source_spin_diluted_momzero.dilutespin(vectortmp2, spinindex);
         //Ideally doing the smearing om the source only on a 3D vector
         //stochastic_source_spin_diluted_momźero.gaussianSmearing(stochastic_source_spin_diluted_momzero, smearedGauge, nsmearGauss, alphaGauss);
         //tmp_time += MPI_Wtime()-start_time;
         stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));


 
         vectorInOut.copy(stochastic_source_spin_diluted_momp_i2);
         solver.solve(vectorInOut, vectorInOut);
         stochastic_propagator_momp_i2[spinindex].copy(vectorInOut);
         //Ideally doing the smearing on the propagator only on a 3D vector
         //stochastic_propagator_momp_i2[spinindex].gaussianSmearing(stochastic_propagator_momp_i2[spinindex], smearedGauge, nsmearGauss, alphaGauss);
         //tmp_time += MPI_Wtime()-start_time;         
         stochastic_propagator_momp_i2[spinindex].writeLIME(outfile_V+"propagator_fini_momentum"+std::to_string(spinindex));


          vectorInOut.copy(stochastic_source_spin_diluted_momzero);
          solver.solve(vectorInOut, vectorInOut);
          stochastic_propagator_momzero[spinindex].copy(vectorInOut);
          //Ideally doing the smearing on the propagator only on a 3D vector
          //stochastic_propagator_momzero[0].gaussianSmearing(stochastic_propagator_momzero[0], smearedGauge, nsmearGauss, alphaGauss);
          //tmp_time += MPI_Wtime()-start_time;

          stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+"propagator_zero_momentum"+std::to_string(spinindex));

       }      

       //Diagram Z1,Z2
       for (int i=0; i< 4; ++i){
         reductionsV3_diluted[i].V3( stochastic_propagator_momp_i2[i], glist_sink_meson, propUP);
         reductionsV2_diluted[i].V4( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP);
       }

       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_meson, glist_source_nucleon, outfilename, 1);


       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_meson, glist_source_nucleon, outfilename, 2);
 
      
       //Diagram Z3,Z4
       for (int i=0; i< 4; ++i){
         reductionsV2_diluted[i].V2( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP);
       }

       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_meson, glist_source_nucleon,  outfilename, 3);


       diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_meson, glist_source_nucleon,  outfilename, 4);


      }//loop over unique set of momenta for p_i2

    } //loop over source position

  } 
  finalize();
  
  return 0;
}


