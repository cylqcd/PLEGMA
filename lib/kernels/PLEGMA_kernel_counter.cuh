#ifndef PLEGMA_KERNEL_COUNTER_CUH
#define PLEGMA_KERNEL_COUNTER_CUH

extern __device__ long unsigned int *counterOps;
extern __device__ long unsigned int *counterReads;
extern __device__ long unsigned int *counterWrites;

__device__ class counter{

public:

  // constructor
  __device__ counter() = default;
  template<class T>
  __device__ counter(T t){}

  template<template<class,class> class S, template<class> class T, class U>
  __device__ operator S<T<U>,U>(){ return S<T<U>,U>(); }

  template<template<class,class> class S, template<class> class T, class U>
  __device__ operator S<T<counter>,U>(){ return S<T<counter>,U>(); }
  
  template<class T>
  __device__ operator T () const { return T(); }
  template<template<class> class T, class U>
  __device__ operator T<U>() const { return T<U>(); }

  // assignment and derived
  template<class T>
  __device__ counter& operator=(T t){ return *this; }
  template<class T>
  __device__ counter& operator+=(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }
  template<class T>
  __device__ counter& operator-=(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }
  template<class T>
  __device__ counter& operator*=(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }
  template<class T>
  __device__ counter& operator/=(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }  

  // usual operations
  template<class T>
  __device__ counter& operator+(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    if(sid == 0 ) printf("I am running\n");
    counterOps[sid]++;
    return *this;
  }
  template<class T>
  __device__ counter& operator-(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }
  template<class T>
  __device__ counter& operator*(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }
  template<class T>
  __device__ counter& operator/(T t){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterOps[sid]++;
    return *this;
  }

  // increase methods
  static __device__ void incReads(){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterReads[sid]++;
  }
  static __device__ void incWrites(){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    counterWrites[sid]++;
  }
  
};


#endif
