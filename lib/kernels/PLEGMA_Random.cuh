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
     @brief CUDA kernel to generate random number from the CURAND RNG states
     @param state CURAND RNG state array

  */
template<typename Float>
__global__ void genUniform_kernel(cuRNGState *state, Float *inOut, int length_field){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    if(sid < length_field ){
        inOut[sid] = Random<Float>(state[sid]);
    }
    /*Float2<Float> *inOut2 = (Float2<Float> *) inOut;
    #pragma unroll
    for(int i = 0 ; i < lenght_field ; ++i){
        Float x = Random<Float>(state);
        Float y = Random<Float>(state);
        Float2<Float> tmp;
        inOut2[i*c_stride + sid] = 
    }*/

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
template<typename Float>
void set_random( PLEGMA_RNG rng_state, PLEGMA_Field<Float> &inOut, int field_deg_free){
    
    dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim( (GK_localVolume * field_deg_free + blockDim.x -1)/blockDim.x , 1 , 1);
    genUniform_kernel<<<gridDim,blockDim>>>(rng_state.State(), inOut.D_elem(),  field_deg_free * GK_localVolume );
}

