#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
static __global__ void shifts_kernel(pFloat2<Float> in, pFloat2<Float> out, int dirOr) {
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
  tuneAndRun(ps, "shifts_kernel", shifts_kernel<Float>, toField2<pFloat2>(Fin), toField2<pFloat2>(Fout),dirOr);
  checkCudaError();
}
