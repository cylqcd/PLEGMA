#include "PLEGMA_kernel_utils.cuh"
#include <PLEGMA_Random.h>


using namespace plegma;
  /**
     @brief CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param length_field length of the CURAND RNG state array
     @param offset offset of the RNG sequence
  */
__global__ void random_init_kernel(RNGState *state, int seed, int offset){
    
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("Field length %d", length_field);
    //Each thread gets same seed, a different sequence number, no offset
    //Determine the global id of the field.
    int seq_number  = LEXIC_1DL_1DG(sid);
    //printf("Number of threads %d and seq number %d\n", sid, seq_number);

    random_init(seed, seq_number, offset, state[sid]);
}

  /**
     @brief Call CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param field_deg_free Degrees of Freedom of the field
     @param offset offset of the RNG sequence
  */
void launch_random_init( RNGState *state, int seed, int offset, int rng_size){
    
    dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
    //PLEGMA_printf("Number of volume[3]: %d\n", HGC_localVolume * field_deg_free);
    dim3 gridDim( (rng_size + blockDim.x -1)/blockDim.x , 1 , 1);
    random_init_kernel<<<gridDim,blockDim>>>(state, seed, offset );
    //cudaDeviceSynchronize();
}
