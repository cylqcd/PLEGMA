#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  
  PLEGMA_params params;
  initialize(argc, argv, &params);
  
  print_status();

  // Check the point source
  PLEGMA_Vector<double> vectorAuxD(BOTH);

  vectorAuxD.random(1234);
  //vectorAuxD.unloadVector();
  //vectorAuxD.norm2Host();
 
 // std::cout<<vectorAuxD.H_elem()[((0*N_COLS+0)*(params.lL[0] * params.lL[1] * params.lL[2] * params.lL[3] ))*2]<<std::endl;

  // finalize the QUDA library
  saveTuneCache(true);
  endQuda();
    
  // finalize the communications layer
  finalizeComms();

  return 0;
}
