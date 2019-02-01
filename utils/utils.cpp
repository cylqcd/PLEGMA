#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace quda;

void initialize(int argc, char **argv, PLEGMA_params *params) {
  read_command_line(argc, argv, params);
  
  // initialize QMP/MPI, QUDA comms grid and RNG 
  initComms(argc, argv, params->procs);

  // initialize the QUDA library
  initQuda(device);
  print_info();

  // initialize PLEGMA params
  PLEGMA_init(params);
  print_status();
}

void finalize() {
  PLEGMA_end();
  
  // finalize the QUDA library
  saveTuneCache(false);
  finalizeGaugeQuda();
  endQuda();
  
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

