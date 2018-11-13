#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
static __global__ void shifts_kernel(Float *in, Float *out, int length_field, int dirOr) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  generic2 R(in);
  Float2<Float> *out2 = (Float2<Float> *) out;
  #pragma unroll
  for(int i = 0 ; i < length_field ; i++)
    out[i*c_stride + sid] = (dirOr<4)?getPlus(i,length_field,dirOr%4,sid):getMinus(i,length_field,dirOr%4,sid);
}

template<typename Float>
static void shiftField(PLEGMA_Field &Fin, PLEGMA_Field &Fout, int dirOr){
  if(Fin.Field_length() != Fout.Fin.Field_length()) errorQuda("Error input, output fields do not match");
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  shifts_kernel<Float><<<gridDim,blockDim>>>(Fin.D_elem(), Fout.D_elem(),Fin.Field_length(),dirOr );
  checkCudaError();
}
