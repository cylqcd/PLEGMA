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

  // Check the point source
  PLEGMA_Vector<double> vectorAuxD(BOTH);

  vectorAuxD.pointSource(params.sourcePosition[0], 0, 0, DEVICE);
  vectorAuxD.unload();
  vectorAuxD.norm2Host();
 
  std::cout<<vectorAuxD.H_elem()[((0*N_COLS+0)*(params.lL[0] * params.lL[1] * params.lL[2] * params.lL[3] ))*2]<<std::endl;

  // finalize the QUDA library
  saveTuneCache(true);
  endQuda();
    
  // finalize the communications layer
  finalize();

  return 0;
}
