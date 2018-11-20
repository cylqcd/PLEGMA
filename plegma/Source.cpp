#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  PLEGMA_params params;
  read_command_line(argc, argv, &params);
  
  // initialize QMP/MPI, QUDA comms grid and RNG 
  initComms(argc, argv, params.procs);

  // initialize the QUDA library
  initQuda(device);
  print_info();

  // initialize PLEGMA info
  initialize(&params);
  print_status();

  // Check the point source
  PLEGMA_Vector<double> vectorAuxD(BOTH);

  vectorAuxD.pointSource(params.sourcePosition[0], 0, 0, DEVICE);
  vectorAuxD.unload();
  vectorAuxD.norm2Host();
  
  // finalize the QUDA library
  saveTuneCache(true);
  endQuda();
    
  // finalize the communications layer
  finalizeComms();

  return 0;
}
