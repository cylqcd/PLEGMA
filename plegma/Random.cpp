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
  PLEGMA_Vector<double> vectorOutD(HOST);
  int nroots=2;
  char stochfilename[100];
  std::snprintf( stochfilename,100, "Z_%d_stochastic_source.lime", nroots);
  vectorAuxD.stochastic_Z(1234,2);
  vectorAuxD.unload();
  vectorAuxD.unpack(vectorOutD.H_elem());
  vectorOutD.norm2Host();
  vectorOutD.write(stochfilename); 
  
  finalize();

  return 0;
}
