#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_Random.h>
using namespace plegma;
  /**
     @brief CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param length_field length of the CURAND RNG state array
     @param offset offset of the RNG sequence
  */
__global__ void random_init_kernel(cuRNGState *state, int seed, int length_field, int offset){
    
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    if ( sid < length_field ){
        // Each thread gets same seed, a different sequence number, no offset
        //Determine the global id of the field.
        //printf("Number of threads %d and offset %d\n", sid, offset);
        curand_init(seed, sid, offset, &state[sid]);
    }
}

  /**
     @brief Call CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param field_deg_free Degrees of Freedom of the field
     @param offset offset of the RNG sequence
  */
void launch_random_init( cuRNGState *state, int seed, int field_deg_free, int offset){
    
    dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
    //printfQuda("Number of volume[3]: %d\n", GK_localVolume * field_deg_free);
    dim3 gridDim( (GK_localVolume * field_deg_free + blockDim.x -1)/blockDim.x , 1 , 1);
    random_init_kernel<<<gridDim,blockDim>>>(state, seed, field_deg_free * GK_localVolume, offset );
    //checkCudaError();
    cudaDeviceSynchronize();
}
