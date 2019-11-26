#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  std::string outfile_V;
  std::string outfile_S;
  std::string outfile_V3;
  
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outProp", "Path for saving the propagator used", verbosity, outfile_S);
  HGC_options->set("outV3", "Path for saving the result of V3_reduction", verbosity, outfile_V3);

  //=========================================================================================================//
  initializePLEGMA();
  {
    int nsmearAPE=4;
    float alphaAPE=0.1;
    int nsmearGauss=4;
    float alphaGauss=0.1;
    int source[4]={4,3,12,1};

  
    // Allocation done on BOTH, DEVICE and HOST
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    {
      PLEGMA_Gauge<double> gauge;
      // Reading from Lime file and loading to device
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();

      // Smearing
      smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();

    }
  
    QUDA_solver solver(mu);
    PLEGMA_Propagator<float> propUP(BOTH);
    
    if(mu != mu_ud) {
      for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_factor[i] = mu_ud_factor[i];
      mu = mu_ud;
      solver.UpdateSolver();
    }

    //create propagator
    for(int isc = 0 ; isc < 12 ; isc++){
      PLEGMA_Vector<double> vectorInOut, vectorAuxD;
      PLEGMA_Vector<float> vectorAuxF;
      
      vectorAuxD.pointSource( source, isc/3, isc%3, DEVICE);
      
      vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      
      solver.solve(vectorInOut, vectorInOut);
      vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propUP.absorb(vectorAuxF, isc/3, isc%3);
    }
    propUP.unload();
    propUP.writeLIME(outfile_S);

    //create vector field
    PLEGMA_Vector<float> vectorStoc(BOTH);
    int nroots=4;
    vectorStoc.randInit(1234);
    vectorStoc.stochastic_Z(nroots);
    solver.solve(vectorStoc, vectorStoc);
    
    vectorStoc.unload();
    vectorStoc.writeLIME(outfile_V);

    //do V3 reduction
    std::vector<int> mom={0,0,1};
    std::vector<GAMMAS> glist={G4,};
    PLEGMA_ScattCorrelator<float> V3reduction(MOMENTUM_SPACE, mom);
    
    V3reduction.V3( vectorStoc, glist, propUP);

    std::vector<std::string> d={"V3",};
    std::vector<std::string> g={"/",};
  
    V3reduction.setDatasets(d);
    V3reduction.setGroups(g);
  
    V3reduction.writeHDF5(outfile_V3);
  }
  finalize();

  return 0;
}


