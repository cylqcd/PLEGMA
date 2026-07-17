#include <PLEGMA_Random.h>
#include <PLEGMA_Random.cuh>
using namespace plegma;

//--------------------------//
// class PLEGMA_Random //
//--------------------------//

// This is is a class which initialized the Random Number Generator, and creates functions for the Field class:
// Field: one complex number per spacetime point.
// Gauge: one SU(3) link variable per spacetime point X spacetime dimension.
// Vector: 12 complex numbers (3colour x 4spin) per spacetime point.
// Vector3D: as above, but only defined on a single timeslice.
// Propagator: 12x12 complex matrix per sink position 
// (usually a whole spacetime volume.)
// Propagtor3D: as above, but with sinks only at one timeslice.
// A cuRAND RNG needs 3 parameters:
// seed: A seed is a number which generates a sequence of random numbers
// sequence number: Sequence number is which index from the sequence will be picked.
// offset: its the shift from which the sequence is started to be read.

// Idea: use global lexicographic symbol for each lattice site along with deflated
// spin-color index.
// On the GPU we have "sid" available, which is the space-time position of the local
// volume. We deflate the sid into color and spin space to get the 1-D position in 
// local volume, then we use 4-D position to deflate it further to get the
// global ids.
// Useful variables: 
// c_procPosition: position of the processors
// rngArg computes the required input to the RNG
PLEGMA_RNG::PLEGMA_RNG(int seedin, int rng_sizes) {

  state = NULL;
  seed = seedin;
  rng_size = rng_sizes;
  // printf("Number of rng_size[1.25]: %d\n", rng_size);

  rank_offset = 0;

  Init();
#if defined(XORWOW)
  PLEGMA_printf("Using curandStateXORWOW\n");
#elif defined(RG32k3a)
  PLEGMA_printf("Using curandStateMRG32k3a\n");
#else
  PLEGMA_printf("Using curandStateMRG32k3a\n");
#endif
}

/**
  @brief Initialize CURAND RNG states
 */
void PLEGMA_RNG::Init() {
  //printf("Number of rng_size[2]: %d\n", rng_size);
  AllocateRNG();
  launch_random_init(state, seed, rank_offset, rng_size);
}

/**
  @brief Allocate Device memory for CURAND RNG states
 */
void PLEGMA_RNG::AllocateRNG() {

  //printf("Number of rng_size[1.5]: %d\n", rng_size);
  if (rng_size>0 && state == NULL) {
    cudaMalloc((void**)&state, rng_size * sizeof(cuRNGState));
    cudaMemset( state , 0 , rng_size * sizeof(cuRNGState) );
    PLEGMA_printf("Allocated array of random numbers with rng_size: %.2f MB\n",((float)rng_size * (float)sizeof(cuRNGState))/(1024*1024));
  } else {
    PLEGMA_error("Array of random numbers not allocated, array size: %d !\nExiting...\n",rng_size);
  }
}

/*! @brief Destructor !*/
PLEGMA_RNG::~PLEGMA_RNG(){
  cudaFree(state);
  PLEGMA_printf("Free array of random numbers with rng_size: %.2f MB\n", ((float)rng_size  * (float)sizeof(cuRNGState))/(1024*1024));
  rng_size = 0;
  state = NULL;
  checkQudaError();
}

/*! @brief Generating random numbers from random distribution */
/*! @brief Restore CURAND array states initialization */
void PLEGMA_RNG::restore() {
  cudaMemcpy(state, backup_state, rng_size * sizeof(cuRNGState), cudaMemcpyHostToDevice);
  checkQudaError();
  hostFree(backup_state, rng_size * sizeof(cuRNGState));
}

/*! @brief Backup CURAND array states initialization */
void PLEGMA_RNG::backup() {
  hostMalloc(backup_state, rng_size * sizeof(cuRNGState));
  cudaMemcpy(backup_state, state, rng_size * sizeof(cuRNGState), cudaMemcpyDeviceToHost);
  checkQudaError();
}

