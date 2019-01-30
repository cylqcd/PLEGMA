#include <cublas_v2.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_Random.h>
using namespace plegma;

template<typename FloatOut,typename FloatIn1, typename FloatIn2, typename Float>
static __global__ void xpby_kernel(FloatOut *z, FloatIn1 *x, FloatIn2 *y, Float beta, int length_field){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;
  generic2<FloatOut> Rz(z);
  generic2<FloatIn1> Ry(y);
  generic2<FloatIn2> Rx(x);
  for(int i = 0 ; i < length_field ; i++)
    Rz.set(i, sid, Rx.get(i,sid) + beta*Ry.get(i,sid));
}

template<typename Float>
static void xpby(PLEGMA_Field<Float> &Fz, PLEGMA_Field<Float> &Fx, PLEGMA_Field<Float> &Fy, Float beta){
  if(Fz.Field_length() != Fx.Field_length()) errorQuda("Error input, output fields do not match");
  if(Fz.Field_length() != Fy.Field_length()) errorQuda("Error input, output fields do not match");
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps,"xpby_kernel",xpby_kernel<Float,Float,Float,Float>,Fz.D_elem(), Fx.D_elem(),
	     Fy.D_elem(),beta,Fz.Field_length());
  checkCudaError();
}

template<typename FloatInOut>
static __global__ void copy_side_to_ghost_kernel(FloatInOut *f, int dir, int sign, int length_field){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_surface3D[dir]) return;
  generic2<FloatInOut> F(f);
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir) {
      id[i] = sign==1 ? (c_localL[dir]-1):0;
    } else {
      id[i] = tmp_sid % c_localL[i];
      tmp_sid /= c_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  sidStride ss;
  ss.sid = c_sideGhost[dir + N_DIMS*sign]*length_field + sid;
  ss.stride = c_surface3D[dir];
  for(int i = 0 ; i < length_field ; i++)
    F.set(i, ss, F.get(i,vid));
}

template<typename Float>
static void copy_side_to_ghost(PLEGMA_Field<Float> &f, int dir){
  if( GK_dimBreak[dir%N_DIMS] ){
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (GK_surface3D[dir%N_DIMS] + blockDim.x -1)/blockDim.x , 1 , 1);
    copy_side_to_ghost_kernel<<<gridDim,blockDim>>>(f.D_elem(), dir%N_DIMS, dir/N_DIMS, f.Field_length());
    checkCudaError();
  }
}

template<typename FloatInOut>
static __global__ void copy_corner_to_ghost_kernel(FloatInOut *f, int dir1, int dir2, int sign1, int sign2, int length_field){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_surface2D[dir1][dir2]) return;
  generic2<FloatInOut> F(f);
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir1) {
      id[i] = sign1==1 ? (c_localL[dir1]-1):0;
    } else if(i==dir2) {
      id[i] = sign2==1 ? (c_localL[dir2]-1):0;      
    } else {
      id[i] = tmp_sid % c_localL[i];
      tmp_sid /= c_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  sidStride ss;
  ss.sid = c_cornerGhost[dir1 + N_DIMS*sign1][dir2 + N_DIMS*sign2]*length_field + sid;
  ss.stride = c_surface2D[dir1][dir2];
  for(int i = 0 ; i < length_field ; i++)
    F.set(i, ss, F.get(i,vid));
}

template<typename Float>
static void copy_corner_to_ghost(PLEGMA_Field<Float> &f, int dir1, int dir2){
  if( (dir1%N_DIMS != dir2%N_DIMS ) && GK_dimBreak[dir1%N_DIMS] && GK_dimBreak[dir2%N_DIMS] ){
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (GK_surface2D[dir1%N_DIMS][dir2%N_DIMS] + blockDim.x -1)/blockDim.x , 1 , 1);
    copy_corner_to_ghost_kernel<<<gridDim,blockDim>>>(f.D_elem(), dir1%N_DIMS, dir2%N_DIMS, dir1/N_DIMS, dir2/N_DIMS, f.Field_length());
    checkCudaError();
  }
}

/**
  @brief CUDA kernel to generate random number from the CURAND RNG states
  @param state CURAND RNG state array, which is used to calculate Z_n noise

 */
template<int n>
__inline__ __device__ Float2<float> rootsunity(int order){
  Float2<float> root;
  root.x = cos(2.0*M_PI*((float)order)/(float)n);
  root.y = sin(2.0*M_PI*((float)order)/(float)n);
  return root;
}
template<>
__inline__ __device__ Float2<float> rootsunity<2>(int order){
  Float2<float> root;
  if(order == 0){
    root.x = 1.0;
    root.y = 0.0;
  }
  else{
    root.x = -1.0;
    root.y = 0.0;
  }
  return root;
}

template<>
__inline__ __device__ Float2<float> rootsunity<4>(int order){
  Float2<float> root;
  if(order == 0){
    root.x = 1.0/sqrt(2.0);
    root.y = 1.0/sqrt(2.0);
  }
  else if(order == 1){
    root.x = -1.0/sqrt(2.0);
    root.y = 1.0/sqrt(2.0);
  }
  else if(order == 2){
    root.x = -1.0/sqrt(2.0);
    root.y = -1.0/sqrt(2.0);
  }
  else{
    root.x = 1.0/sqrt(2.0);
    root.y = -1.0/sqrt(2.0);
  }
  return root;
}

template<typename Float, int n>
__global__ void genStochasticUniform_kernel(cuRNGState *state, int length_field, Float *inout){

  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if( n < 2) return;

  for( int i = 0; i < length_field; ++i){

    Float tmp = PLEGMA_Random<Float, Uniform>(state[sid]);

    for( int order = 0; order <= n-1; ++order){

      if( tmp  < ((Float)order+1.0)/(Float)n ){

        inout2[sid + i*(c_threads)] = rootsunity<n>(order);
        break;
      }
    }
  }
} 

template<typename Float, int n>
void set_stochastic( PLEGMA_RNG &rng_state, PLEGMA_Field<Float> &inOut, int field_deg_free, int rng_size){

  dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim( (rng_size  + blockDim.x -1)/blockDim.x , 1 , 1);
  genStochasticUniform_kernel<Float, n><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem());
}
template<typename Float>
__global__ void genRandomUniform_kernel(cuRNGState *state, int length_field, Float *inout){
  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  for( int i = 0; i < length_field; ++i){
    inout2[sid] = PLEGMA_Random<Float, Uniform>(state[sid]);
  }
} 

template<typename Float>
__global__ void genRandomNormal_kernel(cuRNGState *state, int length_field, Float *inout){
  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  for( int i = 0; i < length_field; ++i){
    inout2[sid] = PLEGMA_Random<Float, Normal>(state[sid]);
  }
} 

template<typename Float>
void set_random( PLEGMA_RNG &rng_state, PLEGMA_Field<Float> &inOut, int field_deg_free, int rng_size, DIST sampling){

  dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim( (rng_size + blockDim.x -1)/blockDim.x , 1 , 1);
  if(sampling == Uniform)
    genRandomUniform_kernel<Float><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem());
  else if( sampling == Normal)
    genRandomNormal_kernel<Float><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem());
  else
    errorQuda("The given distribution is not defined.\n");
}

