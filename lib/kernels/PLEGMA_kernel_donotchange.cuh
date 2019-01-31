#ifndef PLEGMA_KERNEL_DONOTTCHANGE_CUH
#define PLEGMA_KERNEL_DONOTTCHANGE_CUH

__device__ class DoNotChange{

public:
  
  // constructor
  __device__ DoNotChange() = default;
  template<class T>
  __device__ DoNotChange(T t){}

  template<template<class,class> class S, template<class> class T, class U>
  __device__ operator S<T<U>,U>(){ return S<T<U>,U>(); }

  template<template<class,class> class S, template<class> class T, class U>
  __device__ operator S<T<DoNotChange>,U>(){ return S<T<DoNotChange>,U>(); }

  template<class T>
  __device__ operator T () const { return T(); }
  template<template<class> class T, class U>
  __device__ operator T<U>() const { return T<U>(); }

  
  // assignment and derived
  template<class T>
  __device__ DoNotChange& operator=(T t){ return *this; }
  template<class T>
  __device__ DoNotChange& operator+=(T t){ return *this; }
  template<class T>
  __device__ DoNotChange& operator-=(T t){ return *this; }
  template<class T>
  __device__ DoNotChange& operator*=(T t){ return *this; }
  template<class T>
  __device__ DoNotChange& operator/=(T t){ return *this; }

};

#endif
