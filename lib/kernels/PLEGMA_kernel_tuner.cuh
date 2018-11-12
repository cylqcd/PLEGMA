#include <PLEGMA_global.h>

struct ProfileStruct{
  // flops
  // bytes
  // sharedBytesPerThread
  // sharedBytesPerBlock
  // volumeCB
};


template<typename ArgStruct>
class PLEGMA_kernel_tuner : public Tunable{

protected:

  void *kernel;
  ArgStruct args;
  ProfileStruct ps;
  
  long long flops() const {
    return ps.flops * ps.volumeCB;
  }

  long long bytes() const{
    return arg.out.Bytes() + (2*3+1)*arg.in.Bytes() + arg.nParity*2*3*arg.U.Bytes()*meta.VolumeCB();
  }

  bool tuneGridDim() const { return false; }
  unsigned int minThreads() const { return ps.volumeCB; }

  
public:

  PLEGMA_kernel_tuner( void *my_kernel, ArgStruct my_args, ProfileStruct my_ps );
  void apply(const cudaStream_t &stream){
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    kernel<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(args);
  }

};

template<typename ArgStruct>
PLEGMA_kernel_tuner<ArgStruct>::PLEGMA_kernel_tuner(void *my_kernel, ArgStruct my_args, ProfileStruct my_ps){
  kernel = my_kernel;
  args = myargs;
  
};
