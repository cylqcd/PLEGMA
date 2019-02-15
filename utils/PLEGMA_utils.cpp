#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace quda;
#define ALLOCATE
#include "utils/PLEGMA_params.h"
#include "utils/QUDA_params.h"
#undef ALLOCATE

void initialize(int argc, char **argv, bool withQuda) {
  
  Options opt(argc,argv);

  // initialize QMP/MPI, QUDA comms grid and RNG
  // we need to do it first for enabling the printing
  opt.setForced("procs","Set number of processors (X Y Z T), e.g. 1 1 1 1", 0,
		procs[0], procs[1], procs[2], procs[3]);
  for(int i=0; i<4; i++) if( procs[i] <= 0 )
			   PLEGMA_error("Error with dim %d: Negative proc or not divisor of dim\n", i);
  initComms(argc, argv, procs);

  // Reading plegma options
  plegmaOptions(opt);

  if(withQuda) {
    qudaOptions(opt);
    // initialize the QUDA library
    initQuda(device);
    if(verbosity>0) infoQuda();
    qudaInitialized=true;
  }

  // initialize PLEGMA params
  PLEGMA_init(dims, procs, verbosity);
  PLEGMA_status();
}

void finalize() {
  PLEGMA_end();
  
  saveTuneCache(false);
  // finalize the QUDA library
  if(qudaInitialized) {
    finalizeGaugeQuda();
    endQuda();
  }
  
  // finalize the communications layer
  finalizeComms();
}

void createMom(int *Nmom, int momElem[][3], int Q_sq){
  int counter = 0;

  for(int iQ = 0 ; iQ <= Q_sq ; iQ++){
    for(int nx = iQ ; nx >= -iQ ; nx--)
      for(int ny = iQ ; ny >= -iQ ; ny--)
        for(int nz = iQ ; nz >= -iQ ; nz--){
          if( nx*nx + ny*ny + nz*nz == iQ ){
            momElem[counter][0] = nx;
            momElem[counter][1] = ny;
            momElem[counter][2] = nz;
            counter++;
          }
        }
  }
  *Nmom = counter;
}
