#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename","sinkMom_Nucleon", "sinkMom_Meson", "sourceMom_Meson"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  std::string outfile_V="";
  std::string outfile_S="";
  std::string outfile_V3;
  std::string outfile_V2;
  std::string outfile_V4;
  std::string path_V="";
  std::string path_P="";
  
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outProp", "Path for saving the propagator used", verbosity, outfile_S);
  HGC_options->set("outV3", "Path for saving the result of V3_reduction", verbosity, outfile_V3);
  HGC_options->set("outV2", "Path for saving the result of V3_reduction", verbosity, outfile_V2);
  HGC_options->set("outV4", "Path for saving the result of V3_reduction", verbosity, outfile_V4);
  HGC_options->set("loadVector", "Path for loading V", verbosity, path_V);
  HGC_options->set("loadProp", "Path for loading P", verbosity, path_P);

  //=========================================================================================================//
  initializePLEGMA();
  double start_time, tmp_time;
  {
    
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
    gauge.calculatePlaq();
    
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
    for(int isource = 0 ; isource < numSourcePositions; isource++){
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
        vectorAuxD.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;
        vectorInOut.copy(vectorAuxD);

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        solver.solve(vectorInOut, vectorInOut);

        //Smearing at the sink
        start_time = MPI_Wtime();
        vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorInOut);
        propUP.absorb(vectorAuxF, isc/3, isc%3);
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
        vectorAuxD.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;
        
        vectorInOut.copy(vectorAuxD);

        PLEGMA_printf("Going to invert DN for component %d\n", isc);
        solver.solve(vectorInOut, vectorInOut);

        start_time = MPI_Wtime();
        vectorInOut.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorInOut);

        propDN.absorb(vectorAuxF, isc/3, isc%3);
      }

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }
      // we are computing sequential propagators i_2 <- i_1 
      // so source time is fixed
      int sequential_time_source=sourcePositions[isource][3];

      //smearing the 3D propagators
      PLEGMA_Propagator3D<float> propDN3D;
      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorAuxD;
        PLEGMA_Vector<float> vectorAuxF;
        vectorAuxF.absorb(propDN,isc/3, isc%3);
        vectorAuxD.copy(vectorAuxF);
        start_time = MPI_Wtime();
        vectorAuxD.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        tmp_time += MPI_Wtime()-start_time;
        vectorAuxD.mulMomentumPhases(sourceMom_Meson,-1);
        vectorAuxD.apply_gamma(G4);
        vectorAuxF.copy(vectorAuxD);
        propDN3D.absorb(vectorAuxF, sequential_time_source, isc/3, isc%3);
      }

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
          vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
          tmp_time += MPI_Wtime()-start_time;
          vectorAuxF.copy(vectorAuxD);
          propUPDN.absorb(vectorAuxF, isc/3, isc%3);
      }

      PLEGMA_printf("Smearing time %lf sec\n",tmp_time);

      //Next prepare the stochastic propagator and vector
      PLEGMA_Vector<float> vectorStoc_source(BOTH);
      PLEGMA_Vector<float> vectorStoc_propag(BOTH);
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
      solver.solve(vectorStoc_propag, vectorStoc_source);

    
      
      //Compute Diagram B1 and B2 
      vectorStoc_source.apply_gamma5();
      std::vector<GAMMAS> glist_sink_nucleon={G4};
      std::vector<GAMMAS> glist_sink_meson={ONE};
      PLEGMA_ScattCorrelator<float> reductionsV2(MOMENTUM_SPACE, sinkMom_Nucleon);
      PLEGMA_ScattCorrelator<float> reductionsV3(MOMENTUM_SPACE, sinkMom_Meson);

      reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUPDN);
      reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUP);

      //Compute Diagram W1
      
     
    }

  }
  finalize();
  
  return 0;
}


