#include <PLEGMA_global.h>
#include <tune_quda.h>
using namespace quda;

#ifndef PLEGMA_KERNEL_TUNER_H
#define PLEGMA_KERNEL_TUNER_H

#define THREADS_PER_BLOCK 64

#ifndef PLEGMA_NO_TUNING
#include <cupti_profiler.h>
#endif

extern __device__ cudaDeviceProp devProp;

// struct that contains all variables
//  necessary for the tuning evaluation
struct ProfileStruct{
  long long flops; // n. flop per lattice pt
  long long outBytes; // output dimension
  long long inpBytes; // input dimension
  long long siteBytes; // bytes per lattice pt
  long long volume;
  long long stride;
  bool tuneY; // tune for the second dimension of thread blocks
  bool sharedMemory;
  unsigned int sharedBytesPerThread;

  TuneParam tp;
  
  ProfileStruct()=default;
  ProfileStruct(long long vol, unsigned int shBPT=0, bool tY=false){
    flops = 0;
    outBytes = 0;
    inpBytes = 0;
    siteBytes = 0;
    volume = vol;
    stride = vol;
    tuneY = tY;
    sharedMemory = (shBPT>0) ? true : false;
    sharedBytesPerThread = shBPT;
  };
};

template<int ...>
struct seq { };

template<int N, int ...S>
struct gens : gens<N-1, N-1, S...> { };

template<int ...S>
struct gens<0, S...> {
  typedef seq<S...> type;
};

// class to perform the kernel tuning
template<class ...types>
class PLEGMA_kernel_tuner : public Tunable{

protected:

  void (*kernel)(types...); // initialised only when tuning is required
  std::tuple<types...> args; // see above

  std::string kernelName;
  
  char volString[TuneKey::aux_n];
  bool onlyTuning;
  bool tuned;

  ProfileStruct &ps;
  
  long long flops() const {
    return ps.flops * ps.volume;
  }

  long long bytes() const{
    return ps.inpBytes + ps.outBytes + ps.siteBytes * ps.volume;
  }

  bool tuneGridDim() const { return false; }
  unsigned int minThreads() const { return ps.volume; }

  unsigned int sharedBytesPerThread() const {
    if ( ps.sharedMemory ) return ps.sharedBytesPerThread;
    else return 0;
  }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const {
    if ( ps.sharedMemory ) return param.block.x;
    else return 0;
  }
  TuneKey tuneKey() const { return TuneKey(volString, kernelName.c_str(), aux); }

  unsigned int maxBlockSize(const TuneParam &param) const { return MAX_THREADS / (param.block.y*param.block.z); }

  // launching utilities  
  template<int ...S>
  void callKernel(dim3 grid3d, dim3 block3d, int shared, const cudaStream_t stream, seq<S...>) {
    (*kernel)<<<grid3d,block3d,shared,stream>>>(std::get<S>(args)...);
  }
  void launchKernel( dim3 grid3d, dim3 block3d, int shared, const cudaStream_t stream) {
    callKernel(grid3d,block3d,shared,stream,typename gens<sizeof...(types)>::type());
  }
  void calculateFlops(){
    
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (ps.volume + blockDim.x -1)/blockDim.x , 1 , 1);
    std::vector<std::string> event_names {};
    std::vector<std::string> metric_names {
	   "flop_count_dp",
	   "flop_count_sp",
	   "dram_read_transactions",
	   "dram_write_transactions",
	};
    cupti_profiler::profiler profiler(event_names, metric_names);

    profiler.start();
    launchKernel(gridDim,blockDim,THREADS_PER_BLOCK*ps.sharedBytesPerThread,0);
    profiler.stop();

    kernelName = (std::string) profiler.get_kernel_names()[0]; 

    auto metrics = profiler.get_metric_values(kernelName.c_str());
    ps.flops = metrics[0].metricValueUint64 + metrics[1].metricValueUint64;

      }
public:

  // ctor
  PLEGMA_kernel_tuner( ProfileStruct &myps, void(* mykernel)(types...), types... kArgs ) : ps(myps) {
    kernel = mykernel;
    args = std::tuple<types...>(kArgs...);
    sprintf(volString, "%lld", ps.volume);
    sprintf(aux, "volume=%lld,stride=%d,Ndims=%d,Ncols=%d", ps.volume, ps.stride, N_DIMS, N_COLS);
    onlyTuning = false;
    tuned = false;
  } 

  // initialisation
  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    if( ps.tuneY ) param.block.y = 2; // not needed at the moment
  }
  
  // tuning functions
  void tune();
  void run();
  void apply(const cudaStream_t &stream);
  void apply();

  // utilities
  int getGridDimX(){ return ps.tp.grid.x; }
  
};

template<class ...types>
void PLEGMA_kernel_tuner<types...>::tune(){
#ifdef PLEGMA_NO_TUNING
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  ps.tp.block = blockDim;
  dim3 gridDim( (ps.volume + blockDim.x -1)/blockDim.x , 1 , 1);
  ps.tp.grid = gridDim;
  ps.tp.shared_bytes = THREADS_PER_BLOCK*ps.sharedBytesPerThread;
  tuned = true;
#else
  onlyTuning = true;
  apply();
  onlyTuning = false;
#endif
}

// apply tuning and/or running with/without tuning
template<class ...types>
void PLEGMA_kernel_tuner<types...>::apply(const cudaStream_t &stream){
#ifdef PLEGMA_NO_TUNING
  // asked for no tuning, using default parameters
  tune();
  run();
#else
  // performing tuning if we need to
  // calculate number of flops
  if( ps.flops==0 )
    calculateFlops( );
  // tune
  ps.tp = tuneLaunch(*this, getTuning(), getVerbosity());
  tuned = true;
  if( onlyTuning && !activeTuning() ) return;
  launchKernel(ps.tp.grid,ps.tp.block,ps.tp.shared_bytes,stream);
#endif
}

template<class ...types>
void PLEGMA_kernel_tuner<types...>::apply(){ apply(0); }
  
template<class ...types>
void PLEGMA_kernel_tuner<types...>::run(){
#ifdef PLEGMA_NO_TUNING
  if(!tuned) tune();
  launchKernel(ps.tp.grid,ps.tp.block,ps.tp.shared_bytes,0);
#else
  if(!tuned) ps.tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
  launchKernel(ps.tp.grid,ps.tp.block,ps.tp.shared_bytes,0);
#endif
}

template<class ...types>
void tune(ProfileStruct &ps, void (*kernel)(types...), types... kArgs){
  PLEGMA_kernel_tuner<types...> tuner(ps, kernel, kArgs...);
  tuner.tune();
}

template<class ...types>
void run(ProfileStruct &ps, void(* kernel)(types...), types... kArgs){
  PLEGMA_kernel_tuner<types...> tuner(ps, kernel, kArgs...);
  tuner.run();
}

template<class ...types>
void tuneAndRun(ProfileStruct &ps, void(* kernel)(types...), types... kArgs){
  PLEGMA_kernel_tuner<types...> tuner(ps, kernel, kArgs...);
  tuner.apply();
}

#endif
