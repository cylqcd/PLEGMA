#include <PLEGMA_global.h>
#include <tune_quda.h>
using namespace quda;

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
};


// class to perform the kernel tuning
template<typename ArgsStruct>
class PLEGMA_kernel_tuner : public Tunable{

protected:

  void (*kernel)(ArgsStruct);
  ArgsStruct *args;
  ProfileStruct ps;
  char volString[TuneKey::aux_n];
  bool onlyTuning;
  TuneParam tp;
  bool tuned;

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
  TuneKey tuneKey() const { return TuneKey(volString, typeid(*kernel).name(), aux); }

  unsigned int maxBlockSize(const TuneParam &param) const { return MAX_THREADS / (param.block.y*param.block.z); }
  
public:

  // ctor
  PLEGMA_kernel_tuner( void (*my_kernel)(ArgsStruct), ArgsStruct *my_args, ProfileStruct my_ps );
  void tune();
  void run();
  void apply(const cudaStream_t &stream);
  void apply();

  // initialisation
  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    if( ps.tuneY ) param.block.y = 2;
  }

  // utility parameter returns
  int getGridDimX(){ return tp.grid.x; }
  
};

template<typename ArgsStruct>
PLEGMA_kernel_tuner<ArgsStruct>::PLEGMA_kernel_tuner(void (*my_kernel)(ArgsStruct), ArgsStruct *my_args, ProfileStruct my_ps){
  kernel = my_kernel;
  args = my_args;
  ps = my_ps;
  sprintf(volString, "%lld", ps.volume);
  sprintf(aux, "volume=%lld,stride=%d,Ndims=%d,Ncols=%d", ps.volume, ps.stride, N_DIMS, N_COLS);
  onlyTuning = false;
  tuned=false;
};

template<typename ArgsStruct>
void PLEGMA_kernel_tuner<ArgsStruct>::tune(){
#ifdef PLEGMA_NO_TUNING
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  tp.block = blockDim;
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  tp.grid = gridDim;
  tp.shared_bytes = THREADS_PER_BLOCK*ps.sharedBytesPerThread;    
  tuned = true;
#else
  onlyTuning = true;
  apply();
  onlyTuning = false;
#endif
}

// apply tuning and/or running with/without tuning
template<typename ArgsStruct>
void PLEGMA_kernel_tuner<ArgsStruct>::apply(const cudaStream_t &stream){
#ifdef PLEGMA_NO_TUNING
  // asked for no tuning, using defaultparameters
  tune();
  run();
#else
  // performing tuning if we need to
  tp = tuneLaunch(*this, getTuning(), getVerbosity());
  tuned = true;
  if( onlyTuning && !activeTuning() ) return;
  (*kernel)<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(*args);
#endif
}

template<typename ArgsStruct>
void PLEGMA_kernel_tuner<ArgsStruct>::apply(){ apply(0); }

template<typename ArgsStruct>
void PLEGMA_kernel_tuner<ArgsStruct>::run(){
#ifdef PLEGMA_NO_TUNING
  if(!tuned) tune();
  (*kernel)<<<tp.grid,tp.block,tp.shared_bytes>>>(*args);
#else
  if(!tuned) tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
  (*kernel)<<<tp.grid,tp.block,tp.shared_bytes,0>>>(*args);
#endif
}
