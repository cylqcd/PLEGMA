#include <PLEGMA_global.h>
#include <curand_kernel.h>
#ifndef _PLEGMA_RANDOM_H
#define _PLEGMA_RANDOM_H

namespace plegma {

    /////////////////////////  
    // CLASS: PLEGMA_RNG   //
    /////////////////////////
    
/**
 *  Define the curand random number generator used in the library
 *  XORWOW- XOR bit dependent RNG
 *  MRG32K3a- MRG32 dependent RNG
 *  */
#if defined(XORWOW)
  typedef struct curandStateXORWOW cuRNGState;
#elif defined(MRG32k3a)
  typedef struct curandStateMRG32k3a cuRNGState;
#else
  typedef struct curandStateMRG32k3a cuRNGState;
#endif
  enum DIST { Uniform, Normal };

  /**
   *    @brief Class declaration to initialize and hold CURAND RNG states
   *    */
  class PLEGMA_RNG {
    public:
      /*! Constructor */
      PLEGMA_RNG();
      /*! free array */
      virtual ~PLEGMA_RNG();
      /*! initialize curand rng states with seed */
      void Init(int seedin);
      /*! @brief allocate curand rng states array in device memory */
      void AllocateRNG(int rng_sizes);
      /*! @brief return curand rng array size */
      int Size() const { return rng_size;};
      int Rank_Offset(){ return rank_offset;};
      int Seed(){ return seed;};
      __host__ __device__ __inline__ cuRNGState* State(){ return state;};
      /*! @brief Restore CURAND array states initialization */
      void restore();
      /*! @brief Backup CURAND array states initialization */
      void backup();
      /*! array with current curand rng state */
      //cuRNGState *state;
    protected:
      /*! array with current curand rng state */
      cuRNGState *state;
      /*! array for backup of current curand rng state */
      cuRNGState *backup_state;
      /*! initial rng seed */
      int seed;
      /*! @brief number of curand states */
      int rng_size;
      /*! @brief offset in the index, in case of multigpus */
      int rank_offset;
      /*! @brief CURAND array states initialization */
      void INITRNG(int rng_size, int seed, int rank_offsetin);
  };


  /**
   *    TEMPLATE SPECIALIZATION BECAUSE CURAND HAS TWO FUNCTIONS FOR FLOAT AND DOUBLE.
   *    @brief Return a random number between a and b
   *    @param state curand rng state
   *    @param a lower range
   *    @param b upper range
   *    @return  random number in range a,b
   *    */

  template<typename Float, DIST sampling>
    inline  __device__ Float PLEGMA_Random(cuRNGState &state, Float a, Float b){
      Float res;
      return res;
    }

  template<>
    inline  __device__ float PLEGMA_Random<float, Uniform>(cuRNGState &state, float a, float b){
      return a + (b - a) * curand_uniform(&state);
    }

  template<>
    inline  __device__ double PLEGMA_Random<double, Uniform>(cuRNGState &state, double a, double b){
      return a + (b - a) * curand_uniform_double(&state);
    }
  
  template<>
    inline  __device__ float PLEGMA_Random<float, Normal>(cuRNGState &state, float a, float b){
      return a + b * curand_normal(&state);
    }

  template<>
    inline  __device__ double PLEGMA_Random<double, Normal>(cuRNGState &state, double a, double b){
      return a + b * curand_normal_double(&state);
    }

  /**
   *    @brief Return a random number between 0 and 1
   *    @param state curand rng state
   *    @return  random number in range 0,1
   *    */
  template<typename Float, DIST sampling>
    inline  __device__ Float PLEGMA_Random(cuRNGState &state){
      Float res;
      return res;
    }

  template<>
    inline  __device__ float PLEGMA_Random<float, Uniform>(cuRNGState &state){
      return curand_uniform(&state);
    }

  template<>
    inline  __device__ double PLEGMA_Random<double, Uniform>(cuRNGState &state){
      return curand_uniform_double(&state);
    }
  
  template<>
    inline  __device__ float PLEGMA_Random<float, Normal>(cuRNGState &state){
      return curand_normal(&state);
    }

  template<>
    inline  __device__ double PLEGMA_Random<double, Normal>(cuRNGState &state){
      return curand_normal_double(&state);
    }

}
#endif
