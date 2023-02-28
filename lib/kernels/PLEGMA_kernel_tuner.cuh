#include <PLEGMA_global.h>
#include <tune_quda.h>
#include <comm_quda.h>
#include <PLEGMA_utils.h>

#ifdef __NVCC__
#include <targets/cuda/quda_cuda_api.h>
#elif __HIP__
#include <targets/hip/quda_hip_api.h>
#endif

using namespace quda;

#ifndef PLEGMA_KERNEL_TUNER_H
#define PLEGMA_KERNEL_TUNER_H

#define THREADS_PER_BLOCK 64


//extern __device__ cudaDeviceProp devProp;

// struct that contains all variables
//  necessary for the tuning evaluation
struct ProfileStruct{
  bool tuned;
  size_t flops; 
  size_t outBytes; 
  size_t inpBytes;
  size_t texBytes;
  size_t min_volume;
  size_t volume;
  size_t max_volume;
  bool sharedMemory;
  bool tune_globally;
  unsigned int sharedBytesPerThread;

  TuneParam tp;
  int4 aux_range;
  
  ProfileStruct()=default;
  ProfileStruct(size_t vol, unsigned int shBPT=0){
    tuned = false;
    flops = 0;
    outBytes = 0;
    inpBytes = 0;
    texBytes = 0;
    min_volume = vol;
    volume = vol;
    max_volume = vol;
    sharedMemory = (shBPT>0) ? true : false;
    sharedBytesPerThread = shBPT;
    aux_range=make_int4(1,1,1,1);
    tune_globally=true;
  };
};

template<int ...>
struct sequ { };

template<int N, int ...S>
struct gens : gens<N-1, N-1, S...> { };

template<int ...S>
struct gens<0, S...> {
  typedef sequ<S...> type;
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

  ProfileStruct &ps;
  
  long long flops() const {
    return ps.flops*ps.tp.block.x;
  }

  long long bytes() const{
    return ps.inpBytes + ps.outBytes + ps.texBytes;
  }

  std::string perfString(float time) const {
    std::string s = Tunable::perfString(time);
    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << time << " s, " << s;
    return ss.str();
  }

  bool tuneGridDim() const { return false; }
  unsigned int minThreads() const { return ps.min_volume; }

  bool tuneSharedBytes() const { return false; }
  unsigned int sharedBytesPerThread() const {
    if ( ps.sharedMemory ) return ps.sharedBytesPerThread;
    else return 0;
  }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const {
    if ( ps.sharedMemory ) return sharedBytesPerThread() * param.block.x;
    else return 0;
  }

  bool tuneVolume() const { if(ps.max_volume > ps.min_volume) return true; else return false; }
  // until possible increasing the volume at powers of 2 (i.e. adding ps.volume to itself),
  // then touching max_volume and then exceeding of 1 to move on.
  size_t volumeStep() const { return std::max(std::min(ps.max_volume - ps.volume, ps.volume), (size_t) 1); }
  bool advanceVolume(TuneParam &param) const {
    bool ret;
    ps.volume += volumeStep();
    if (ps.volume > ps.max_volume) {
      ps.volume = ps.min_volume;
      ret = false;
    } else {
      ret = true;
    }
    resetBlockDim(param);
    if (!tuneGridDim()) {
      param.grid = dim3((minThreads()+param.block.x-1)/param.block.x, 1, 1);
      param.grid.x *= ps.volume/ps.min_volume;
    }
    
    return ret;
  }

  bool advanceBlockDim(TuneParam &param) const {
    bool ret = Tunable::advanceBlockDim(param);
    if(param.shared_bytes < sharedBytesPerBlock(param))
      param.shared_bytes = sharedBytesPerBlock(param);
    if(!tuneGridDim() && tuneVolume())
      param.grid.x *= ps.volume/ps.min_volume;
    return ret;
  }
  
  bool advanceTuneParam(TuneParam &param) const {
    return Tunable::advanceTuneParam(param) || advanceVolume(param);
  }

  TuneKey tuneKey() const { return TuneKey(volString, kernelName.c_str(), aux); }

  bool tuneAuxDim() const { if(ps.aux_range.x!=1 || ps.aux_range.y!=1 || ps.aux_range.z!=1 || ps.aux_range.w!=1) return true; else return false; }
  bool advanceAux(TuneParam &param) const {
    if(tuneAuxDim()) {
      int max = ps.aux_range.x*ps.aux_range.y*ps.aux_range.z*ps.aux_range.w;
      int4 aux = param.aux; // starting from 0
      int current = (((aux.w-1)*ps.aux_range.z + aux.z - 1)*ps.aux_range.y + aux.y - 1)*ps.aux_range.x + aux.x - 1;
      if(current < max-1) {
	current++;

	param.aux.x = current%ps.aux_range.x + 1;
	current/=ps.aux_range.x;
	param.aux.y = current%ps.aux_range.y + 1;
	current/=ps.aux_range.y;
	param.aux.z = current%ps.aux_range.z + 1;
	current/=ps.aux_range.z;
	param.aux.w = current%ps.aux_range.w + 1;

	return true;
      } else {
	param.aux = make_int4(1,1,1,1);
	return false;
      }
    } else {
      return false;
    }}


  unsigned int maxBlockSize(const TuneParam &param) const { return MAX_THREADS / (param.block.y*param.block.z); }

  // launching utilities  
  template<int ...S>
  void callKernel(TuneParam tp, const qudaStream_t stream, sequ<S...>) {
    if( typeid(ProfileStruct &)==typeid(std::get<0>(args))) {
      // in case ProfileStruct is the first argument we call it as a function
      (*kernel)(std::get<S>(args)...);
    } else {
      #ifdef __NVCC__
      (*kernel)<<<tp.grid,tp.block,tp.shared_bytes,quda::target::cuda::get_stream(stream)>>>(std::get<S>(args)...);
      #elif __HIP__
      (*kernel)<<<tp.grid,tp.block,tp.shared_bytes, quda::target::hip::get_stream(stream)>>>(std::get<S>(args)...);
      #endif
    }      
   // cudaDeviceSynchronize();
  }
  void launchKernel( TuneParam tp, const qudaStream_t stream) {
    callKernel(tp,stream,typename gens<sizeof...(types)>::type());
  }
  
public:

  // ctor
 PLEGMA_kernel_tuner( ProfileStruct &ps, std::string kname, void (*kernel)(types...), types... kArgs ) :
  kernel(kernel), args(std::tuple<types...>(kArgs...)), ps(ps), onlyTuning(false) {
    sprintf(volString, "%lldx%lldx%lldx%lld", HGC_localL[0], HGC_localL[1], HGC_localL[2], HGC_localL[3]);
    sprintf(aux, "volume=%lld,Ndims=%d,Ncols=%d,maxvolume=%d,aux_range=(%d,%d,%d,%d)", ps.volume, N_DIMS, N_COLS, ps.max_volume, ps.aux_range.x, ps.aux_range.y, ps.aux_range.z, ps.aux_range.w);
    kernelName = kname + (std::string) typeid(*kernel).name(); // with cupti no longer necessary
    setPolicyTuning(ps.tune_globally);
  }

  ~PLEGMA_kernel_tuner(){
    setPolicyTuning(false);
  }
    
  // initialisation
  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
  }
  
  // tuning functions
  void tune();
  void run();
  void apply(const qudaStream_t &stream);
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
  ps.tuned = true;
#else
  onlyTuning = true;
  apply();
  onlyTuning = false;
#endif
}

// apply tuning and/or running with/without tuning
template<class ...types>
void PLEGMA_kernel_tuner<types...>::apply(const qudaStream_t &stream){
#ifdef PLEGMA_NO_TUNING
  // asked for no tuning, using default parameters
  tune();
  run();
#else
  // performing tuning if we need to tune
  if( !ps.tuned && !activeTuning() && ps.tune_globally ) comm_barrier(); //syncronizing 
  if( !ps.tuned ) ps.tp = tuneLaunch(*this, getTuning(), (QudaVerbosity) HGC_verbosity);
  if( !ps.tuned ) qudaGetLastError(); // ensuring that the error state has been clean
  if( !activeTuning() ) ps.tuned = true;
  if( onlyTuning && !activeTuning() ) return;

  launchKernel(ps.tp,stream);

  // HACK: For unknown reason, the Out Of Memory error state is not seen in QUDA/lib/tune.cpp
  // by error = cudaGetLastError(); (line 765).
  // So here we use jitify_error to communicate to the tuner the failure of the kernel.
  qudaError_t error = qudaGetLastError();
  if( activeTuning() && ps.tune_globally ) {
    //double tmp = error;
    std::vector<double> tmp;
    tmp.push_back(error);
    quda::comm_allreduce_max(tmp);
    error = (qudaError_t) tmp[0];
  }
  //if( error != cudaSuccess ) jitify_error = (CUresult) error;
  if( !activeTuning() ) checkQudaError();
#endif
}

template<class ...types>
void PLEGMA_kernel_tuner<types...>::apply(){ apply(device::get_stream(0)); }
  
template<class ...types>
void PLEGMA_kernel_tuner<types...>::run(){
#ifdef PLEGMA_NO_TUNING
  if(!ps.tuned) tune();
  launchKernel(ps.tp.grid,ps.tp.block,ps.tp.shared_bytes,0);
#else
  if(!ps.tuned) ps.tp = tuneLaunch(*this, QUDA_TUNE_NO, (QudaVerbosity) HGC_verbosity);
  launchKernel(ps.tp,device::get_stream(0));
#endif
  checkQudaError();
}

template<class ...types, class ...typesK>
void tune(ProfileStruct &ps, std::string kname, void (*kernel)(typesK...), types&&... kArgs){
  PLEGMA_kernel_tuner<typesK...> tuner(ps, kname, kernel, kArgs...);
  tuner.tune();
}

template<class ...types, class ...typesK>
void run(ProfileStruct &ps, std::string kname, void(* kernel)(typesK...), types&&... kArgs){
  PLEGMA_kernel_tuner<typesK...> tuner(ps, kname, kernel, kArgs...);
  tuner.run();
}

template<class ...types, class ...typesK>
void tuneAndRun(ProfileStruct &ps, std::string kname, void(* kernel)(typesK...), types&&... kArgs){
  PLEGMA_kernel_tuner<typesK...> tuner(ps, kname, kernel, kArgs...);
  tuner.apply();
}

#endif
