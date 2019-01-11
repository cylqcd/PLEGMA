#include <PLEGMA_global.h>
#include <tune_quda.h>
// Classes to count flops and reads/writes
// #include <PLEGMA_kernel_counter.cuh>
using namespace quda;

#ifndef PLEGMA_KERNEL_TUNER_H
#define PLEGMA_KERNEL_TUNER_H

#define THREADS_PER_BLOCK 64

/* CUPTI functionalities disabled for the moment
 * since it's impossible to make it work properly
#ifndef PLEGMA_NO_TUNING
#include <cupti_profiler.h>
#endif
*/

extern __device__ cudaDeviceProp devProp;
extern __device__ long unsigned int *counterOps;
extern __device__ long unsigned int *counterReads;
extern __device__ long unsigned int *counterWrites;

// struct that contains all variables
//  necessary for the tuning evaluation
struct ProfileStruct{
  bool measured;
  long unsigned int flops; 
  long unsigned int outBytes; 
  long unsigned int inpBytes;
  long long texBytes;
  long long volume;
  long long stride;
  bool tuneY; // tune for the second dimension of thread blocks
  bool sharedMemory;
  unsigned int sharedBytesPerThread;

  TuneParam tp;
  
  ProfileStruct()=default;
  ProfileStruct(long long vol, unsigned int shBPT=0, bool tY=false){
    measured = false;
    flops = 0;
    outBytes = 0;
    inpBytes = 0;
    texBytes = 0;
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
    return ps.flops*ps.tp.block.x;
  }

  long long bytes() const{
    return ps.inpBytes + ps.outBytes + ps.texBytes;
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
    cudaDeviceSynchronize();
  }
  void launchKernel( dim3 grid3d, dim3 block3d, int shared, const cudaStream_t stream) {
    callKernel(grid3d,block3d,shared,stream,typename gens<sizeof...(types)>::type());
  }

  // calculate flops with CUPTI
  /*
  void calculateFlops(){
    
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (ps.volume + blockDim.x -1)/blockDim.x , 1 , 1);

    launchKernel(gridDim,blockDim,THREADS_PER_BLOCK*ps.sharedBytesPerThread,0);

    std::vector<std::string> event_names { "warps_launched",
					   "active_cycles" };
    std::vector<std::string> metric_names {"flop_count_dp",
					   "flop_count_sp",
					   "dram_read_throughput",
					   "dram_write_throughput",
					   "tex_cache_throughput"};
    cupti_profiler::profiler profiler(event_names, metric_names);

    profiler.start();
    for(int i=0; i<profiler.get_passes()+1; i++){
      launchKernel(gridDim,blockDim,THREADS_PER_BLOCK*ps.sharedBytesPerThread,0);
      cudaDeviceSynchronize();
    }
    profiler.stop();

    kernelName = (std::string) profiler.get_kernel_names()[0];
    printf("### %s ###\n",kernelName.c_str());
   
    auto metrics = profiler.get_metric_values(kernelName.c_str());
    ps.flops = metrics[0].metricValueUint64 + metrics[1].metricValueUint64;
    //ps.inpBytes = metrics[2].metricValueUint64;
    //ps.outBytes = metrics[3].metricValueUint64;
    //ps.texBytes = metrics[4].metricValueUint64;
    ps.measured = true;
    printf("Flops %d\n",ps.flops);
  }
  */
  
public:

  // ctor
  PLEGMA_kernel_tuner( ProfileStruct &myps, void(* mykernel)(types...), types... kArgs ) : ps(myps) {
    kernel = mykernel;
    args = std::tuple<types...>(kArgs...);
    sprintf(volString, "%lld", ps.volume);
    sprintf(aux, "volume=%lld,stride=%d,Ndims=%d,Ncols=%d", ps.volume, ps.stride, N_DIMS, N_COLS);
    kernelName = (std::string) typeid(*kernel).name(); // with cupti no longer necessary
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

  // TODO: Restore if we find a way to cast flops as counters
  /* 
  void calculateFlops(){
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (ps.volume + blockDim.x -1)/blockDim.x , 1 , 1);
    //resetCounters( ps.volume );
    int size = ps.volume;
    long unsigned int *dummy = new long unsigned int[size];
    long unsigned int loc, glob;
      
    for(int i=0; i<size; ++i)
      dummy[i] = 0;                                                                                                         

    cudaMalloc(&counterOps,size*sizeof(long unsigned int));
    cudaMalloc(&counterReads,size*sizeof(long unsigned int));
    cudaMalloc(&counterWrites,size*sizeof(long unsigned int));
    cudaMemcpy(&counterOps, &dummy, size*sizeof(long unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(&counterReads, &dummy, size*sizeof(long unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(&counterWrites, &dummy, size*sizeof(long unsigned int),cudaMemcpyHostToDevice);
    
    launchKernel(gridDim,blockDim,ps.sharedBytesPerThread,0);

    loc = 0;
    cudaMemcpy(&dummy, &counterOps, size*sizeof(long unsigned int),cudaMemcpyDeviceToHost);
    for(int i=0; i<size; i++){
      if (i<100) printf("%d \t %d\n",i,dummy[i]);
      loc += dummy[i];
    }
    MPI_Allreduce(&loc,&glob,1,MPI_UNSIGNED_LONG,MPI_SUM,MPI_COMM_WORLD);
    ps.flops = glob;

    loc = 0;
    cudaMemcpy(&dummy, &counterReads, size*sizeof(long unsigned int),cudaMemcpyDeviceToHost);
    for(int i=0; i<size; i++)
      loc += dummy[i];
    MPI_Allreduce(&loc,&glob,1,MPI_UNSIGNED_LONG,MPI_SUM,MPI_COMM_WORLD);
    ps.inpBytes = glob;

    loc = 0;
    cudaMemcpy(&dummy, &counterWrites, size*sizeof(long unsigned int),cudaMemcpyDeviceToHost);
    for(int i=0; i<size; i++)
      loc += dummy[i];
    MPI_Allreduce(&loc,&glob,1,MPI_UNSIGNED_LONG,MPI_SUM,MPI_COMM_WORLD);
    ps.outBytes = glob;
    
    printf("Flops %ld\n", ps.flops);
    printf("In Bytes %ld\n", ps.inpBytes);

    cudaFree(&counterOps);
    cudaFree(&counterWrites);
    cudaFree(&counterReads);
    delete [] dummy;
  }
  */
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

/*
template<class T>
T dummyCasting(T arg) {
  return arg;
}

template<class T, class U>
T dummyCasting(U arg) {
  //T dummy;
  //return dummy;
  return T();
}

template<template<class> class T, class U>
T<counter> dummyCasting(T<U> arg){
  return T<counter>(counter(arg));
}

// implementation to be changed so that flops do not get calculated if already known
template<typename ...types, typename ...args>
void calcFlops(ProfileStruct &ps, void(* kernel)(types...), args... kArgs){
  PLEGMA_kernel_tuner<types...> tuner(ps, kernel, dummyCasting<types>(kArgs)...);
  tuner.calculateFlops(); // calculates the number of flops and stores them in ps
}
*/

#endif
