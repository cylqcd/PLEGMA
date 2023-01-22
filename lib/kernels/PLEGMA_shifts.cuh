#include "PLEGMA_kernel_utils.cuh"
using namespace plegma;

template<typename Float>
static __global__ void shifts1_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= in.volume()) return;
  in.setSid(sid);
  if(dirOr<4) in.template shift<Minus>(dirOr%4); //:bring the elem from behind the current pos
  else in.template shift<Plus>(dirOr%4);
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
  checkQudaError();
}


template<typename Float>
static __global__ void shifts2_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr1, int dirOr2) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= in.volume()) return;
  in.setSid(sid);
  if(dirOr1<4 && dirOr2<4) in.template shift<MinusMinus>(dirOr1%4,dirOr2%4);
  else if(dirOr1<4) in.template shift<MinusPlus>(dirOr1%4,dirOr2%4);
  else if(dirOr2<4) in.template shift<PlusMinus>(dirOr1%4,dirOr2%4);
  else in.template shift<PlusPlus>(dirOr1%4,dirOr2%4);
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
  checkQudaError();
}


template<typename Float>
static __global__ void shifts3_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr1, int dirOr2, int dirOr3) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= in.volume()) return;
  in.setSid(sid);
  if(dirOr1<4 && dirOr2<4 && dirOr3<4) in.template shift<MinusMinusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr1<4 && dirOr2<4) in.template shift<MinusMinusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr2<4 && dirOr3<4) in.template shift<PlusMinusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr1<4 && dirOr3<4) in.template shift<MinusPlusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr1<4) in.template shift<MinusPlusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr2<4) in.template shift<PlusMinusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else if(dirOr3<4) in.template shift<PlusPlusMinus>(dirOr1%4,dirOr2%4,dirOr3%4);
  else in.template shift<PlusPlusPlus>(dirOr1%4,dirOr2%4,dirOr3%4);
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
  checkQudaError();
}
