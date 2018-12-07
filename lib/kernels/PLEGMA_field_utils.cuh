#include <cublas_v2.h>
#include <PLEGMA_kernel_utils.cuh>
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
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  xpby_kernel<<<gridDim,blockDim>>>(Fz.D_elem(), Fx.D_elem(), Fy.D_elem(),beta,Fz.Field_length());
  checkCudaError();
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
