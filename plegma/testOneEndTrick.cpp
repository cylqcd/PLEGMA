#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsrc", "twop-filename"};
  
int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  //=========================================================================================================//
  initializePLEGMA();
  double start_time, tmp_time;
  {
    // {
    //   // Reading from Lime file and loading to device
    //   PLEGMA_Gauge<double> gauge;
    //   gauge.readFile(latfile, LIME_FORMAT);
    //   gauge.load();
    //   gauge.calculatePlaq();
      
    //   // Loading to QUDA and computing plaquette also there
    //   initGaugeQuda(gauge, true);
    //   plaqQuda();
    // }
    // QUDA_solver solver(mu);
    PLEGMA_Field<double> out(BOTH, SCALAR);
    out.setUnit({0,});

    int t0=0;
    std::vector<int> mom{0,0,0};

    PLEGMA_FT<double> corr( mom, 3, false);

    corr.apply( out, FT_NAIVE, 1);
    corr.writeASCII( twop_filename.c_str(), t0);
    
    // PLEGMA_Field<double> out(BOTH, SCALAR);
    // out.zero_device();
    
    // PLEGMA_Vector<double> vectorAux(DEVICE), vectorInOut(DEVICE);
    // int t0=10;
    // vectorAux.randInit(1234);
    
    // for(int isrc=0; isrc<numSourcePositions; isrc++){
    //   // - stochastic source at fixed timeslice t0
    
    //   vectorAux.stochastic_Z(4);
    //   vectorInOut.absorbTimeslice(vectorAux,t0);

    //   // - compute \phi(t,x)^alfa

    //   solver.solve( vectorInOut,  vectorInOut);

    //   // - reduce by summing over all his spincolor components
    //   out.sumModVector(vectorInOut, true);
    // }

    // out.cscale(1./numSourcePositions);
    
    // // - sum over x
    // std::vector<int> mom{0,0,0};
 
    // PLEGMA_FT<double> corr( mom, 3, false);

    // corr.apply( out, FT_NAIVE, 1);
    // corr.writeASCII( twop_filename.c_str(), t0);
    
    // // propUP.rotateToPhysicalBase_device(+1);
    // // propUP.applyBoundaries_device(sourcePositions[isource][3]);

  }
  
  finalize();
  return 0;
}

