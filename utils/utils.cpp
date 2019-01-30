#include <PLEGMA.h>
#include <cmath>
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

/* ================ Small introduction to Hierarchical probing ============
# There is an unsigned integer "k" running from 1 until ...
# From this integer we can specify several important quantities regarding the coloring
# The total number of colors is given by N_{hc} = 2 * 2^{d(k-1)} where d is the number of dimensions
# The distance seperating neighbors carrying the same color is D=2^k
# The extent of the elementary coloring block is given L_u=2^{k-1}
# A condition must be fulfilled in order to be able to do the coloring for a specific k
# The condition must be that the number of blocks in each direction must be even
# And that Ls%(2*Lu)=0 and Lt%(2*Lu)=0
 */

class Hprobing{
private:
  int k;  // index for the coloring distance
  int Nc; // Number of colors = Number of Hadamard vectors
  short d; // Number of dimension of Hprob (For now d=4)
  unsigned int* Vc; // array to hold the coloring of the lattice
public:
  Hprobing(int k_probing, int d=4);
};

Hprobing::Hprobing(int k_probing, int d):k(k_probing),d(d){
  if(!GK_init_PLEGMA_flag){ fprintf(stderr, "Error PLEGMA should be initialized before use this class"); exit(-1);}
  if(d != 4) errorQuda("Hierarchical probing supports only 4D coloring up to now");
  if(k<=0) errorQuda("The index of the Hprobing should greater than zero");
  Nc = 2*std::pow(2,d*(k-1));
  
}

/*
@brief: This function computes the colors for the elementary color block
Inputs:
lc: Pointer to the array where we want to store the colors
Nc: The number of colors we want to put in the block
Lu: The extent of the color block
d: Dimension, either 2 or 3
 */
