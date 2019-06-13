#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
static __global__ void shifts_kernel(Float *in, Float *out, int length_field, int dirOr) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= DGC_localVolume) return;
  generic2<Float> R(in);
  Float2<Float> *out2 = (Float2<Float> *) out;
  for(int i = 0 ; i < length_field ; i++){
    Float2<Float> tmp = (dirOr<4) ? R.get<Minus>(i,sid,length_field,dirOr%4) : R.get<Plus>(i,sid,length_field,dirOr%4);
    out2[i*DGC_localVolume + sid] = tmp;
  }
}


template<typename Float>
static void shiftField(PLEGMA_Field<Float> &Fin, PLEGMA_Field<Float> &Fout, int dirOr){
  if(Fin.Field_length() != Fout.Field_length()) PLEGMA_error("Error input, output fields do not match");
  ProfileStruct ps( HGC_localVolume );
  tuneAndRun(ps, "shifts_kernel", shifts_kernel<Float>, Fin.D_elem(), Fout.D_elem(),Fin.Field_length(),dirOr);
  checkCudaError();
}
