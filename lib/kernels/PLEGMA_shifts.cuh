#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
static __global__ void shifts1_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= in.volume()) return;
  in.setSid(sid);
  if(dirOr<4) in.shift<Minus>(dirOr%4); //:bring the elem from behind the current pos
  else in.shift<Plus>(dirOr%4);
  out.setSid(sid);
  for(int i = 0 ; i < in.site_size; i++){
    out.set(i, in.get(i));
  }
}


template<typename Float>
static void shiftField(PLEGMA_Field<Float> &Fin, PLEGMA_Field<Float> &Fout, int dirOr){
  assert(Fin.checkVolume(Fout));
  ProfileStruct ps( Fin.Total_length() );
  tuneAndRun(ps, "shifts1_kernel", shifts1_kernel<Float>, toField2<pFloat2>(Fin), toField2<pFloat2>(Fout),dirOr);
  checkCudaError();
}


template<typename Float>
static __global__ void shifts2_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr1, int dirOr2) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= in.volume()) return;
  in.setSid(sid);
  if(dirOr1<4 && dirOr2<4) in.shift<MinusMinus>(dirOr1%4,dirOr2%4);
  else if(dirOr1<4) in.shift<MinusPlus>(dirOr1%4,dirOr2%4);
  else if(dirOr2<4) in.shift<PlusMinus>(dirOr1%4,dirOr2%4);
  else in.shift<PlusPlus>(dirOr1%4,dirOr2%4);
  out.setSid(sid);
  for(int i = 0 ; i < in.site_size; i++){
    out.set(i, in.get(i));
  }
}


template<typename Float>
static void shiftField(PLEGMA_Field<Float> &Fin, PLEGMA_Field<Float> &Fout, int dirOr1, int dirOr2){
  assert(Fin.checkVolume(Fout));
  ProfileStruct ps( Fin.Total_length() );
  tuneAndRun(ps, "shifts2_kernel", shifts2_kernel<Float>, toField2<pFloat2>(Fin), toField2<pFloat2>(Fout),dirOr1,dirOr2);
  checkCudaError();
}


template<typename Float>
static __global__ void shifts3_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr1, int dirOr2, int dirOr3) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= in.volume()) return;
  in.setSid(sid);
  if(dirOr1<4 && dirOr2<4 && dirOr3<4) in.shift<MinusMinusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr1<4 && dirOr2<4) in.shift<MinusMinusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr2<4 && dirOr3<4) in.shift<PlusMinusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr1<4 && dirOr3<4) in.shift<MinusPlusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr1<4) in.shift<MinusPlusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr2<4) in.shift<PlusMinusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr3<4) in.shift<PlusPlusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else in.shift<PlusPlusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  out.setSid(sid);
  for(int i = 0 ; i < in.site_size; i++){
    out.set(i, in.get(i));
  }
}


template<typename Float>
static void shiftField(PLEGMA_Field<Float> &Fin, PLEGMA_Field<Float> &Fout, int dirOr1, int dirOr2, int dirOr3){
  assert(Fin.checkVolume(Fout));
  ProfileStruct ps( Fin.Total_length() );
  tuneAndRun(ps, "shifts3_kernel", shifts3_kernel<Float>, toField2<pFloat2>(Fin), toField2<pFloat2>(Fout),dirOr1,dirOr2,dirOr3);
  checkCudaError();
}
