#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace quda;
#define ALLOCATE
#include "utils/PLEGMA_params.h"
#include "utils/QUDA_params.h"
#undef ALLOCATE

void initialize(int argc, char **argv, bool withQuda) {
  MPI_Init( NULL, NULL ); // initializing MPI only to control the printing
  
  Options opt(argc,argv);
  plegmaOptions(opt);
  
  // initialize QMP/MPI, QUDA comms grid and RNG
  MPI_Finalize(); // initializing the wanted communications here
  initComms(argc, argv, procs);

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
