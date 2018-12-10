#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
struct ArgsShifts{
  Float *in,  *out; int length_field, dirOr;
  __host__ void operator()(dim3 blocks, dim3 threads, int shared, const cudaStream_t stream);
};

template<typename Float>
static __global__ void shifts_kernel( ArgsShifts<Float> args ){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= c_threads) return;
  generic2<Float> R(args.in);
  Float2<Float> *out2 = (Float2<Float> *) args.out;
#pragma unroll
  for(int i = 0 ; i < args.length_field ; i++){
    Float2<Float> tmp = (args.dirOr<4) ? R.get<Minus>(i,sid,args.length_field,args.dirOr%4) : R.get<Plus>(i,sid,args.length_field,args.dirOr%4);
    out2[i*c_stride + sid] = tmp;
  }
}

template<typename Float>
__host__ void ArgsShifts<Float>::operator()(dim3 blocks, dim3 threads, int shared, const cudaStream_t stream){
  shifts_kernel<<<blocks,threads,shared,stream>>>(*this);
}

template<typename Float>
static void shiftField(PLEGMA_Field<Float> &Fin, PLEGMA_Field<Float> &Fout, int dirOr){
  if(Fin.Field_length() != Fout.Field_length()) errorQuda("Error input, output fields do not match");
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  ArgsShifts<Float> kernel_args = { Fin.D_elem(), Fout.D_elem(), Fin.Field_length(), dirOr }; 
  ProfileStruct kernel_ps(GK_localVolume);
  PLEGMA_kernel_tuner<ArgsShifts,Float> tuner(&kernel_args, kernel_ps);
  tuner.apply();
  
  checkCudaError();
}
