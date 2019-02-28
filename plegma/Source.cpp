#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;

int main(int argc, char **argv)
{

  initializeOptions(argc, argv, false); // Add list of options
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();
  

  // Check the point source
  PLEGMA_Vector<double> vectorAuxD(BOTH);

  vectorAuxD.pointSource(sourcePositions[0], 0, 0, DEVICE);
  vectorAuxD.unload();
  vectorAuxD.norm2Host();
 
  std::cout<<vectorAuxD.H_elem()[((0*N_COLS+0)*(HGC_localL[0] * HGC_localL[1] * HGC_localL[2] * HGC_localL[3] ))*2]<<std::endl;

  // finalize the QUDA library
  saveTuneCache(true);
  endQuda();
    
  // finalize the communications layer
  finalize();

  return 0;
}
