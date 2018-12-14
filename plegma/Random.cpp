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

  PLEGMA_Vector<double> vectorAuxD(BOTH);

  char stochfilename[100];
  int nroots = 4;
  std::snprintf( stochfilename,100, "1node_Z_%d_stochastic_source.lime", nroots);
  vectorAuxD.random(1234, nroots);
  vectorAuxD.unload();
  vectorAuxD.norm2Host();
  vectorAuxD.write(stochfilename); 
  
  finalize();

  return 0;
}
