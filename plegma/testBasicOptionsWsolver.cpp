#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];
extern QudaDagType dagger;
extern bool kernel_pack_t;
extern int solution_accumulator_pipeline;
extern QudaInverterType inv_type;
extern QudaInverterType precon_type;

int main(int argc, char **argv)
{
  PLEGMA_params params;
  {// block for getting options
    Options opt(argc,argv);
    basicOptionsWsolver(opt,&params,true);
    // add more options within the block
  }
  initPlegma(argc, argv, &params);
  
  
  finalize();
 
  return 0;
}
