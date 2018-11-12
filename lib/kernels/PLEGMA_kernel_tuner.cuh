#include <PLEGMA_global.h>

// struct that contains all variables
//  necessary for the tuning evaluation
struct ProfileStruct{
  long long flops;
  long long outBytes;
  long long inpBytes;
  long long siteBytes;
  // sharedBytesPerThread
  // sharedBytesPerBlock
  long long volumeCB;
};

// structure that contains all arguments necessary
//  to run the plaquette kernel
template<typename Float, typename FloatG>
struct ArgsPlaquette{
  gaugeTex<FloatG> gaugeTex;
  Float *partial_plaq
};


// class to perform the kernel tuning
template<typename ArgsStruct>
class PLEGMA_kernel_tuner : public Tunable{

protected:

  void *kernel;
  ArgsStruct args;
  ProfileStruct ps;
  
  long long flops() const {
    return ps.flops * ps.volumeCB;
  }

  long long bytes() const{
    return ps.inpBytes + ps.outBytes + ps.siteBytes * ps.volumeCB;
  }

  // check these
  bool tuneGridDim() const { return false; }
  unsigned int minThreads() const { return ps.volumeCB; }

  
public:

  PLEGMA_kernel_tuner( void *my_kernel, ArgsStruct my_args, ProfileStruct my_ps );
  void apply(const cudaStream_t &stream){
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    kernel<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(args);
  }

};

template<typename ArgsStruct>
PLEGMA_kernel_tuner<ArgsStruct>::PLEGMA_kernel_tuner(void *my_kernel, ArgsStruct my_args, ProfileStruct my_ps){
  kernel = my_kernel;
  args = my_args;
  ps = my_ps;
};
