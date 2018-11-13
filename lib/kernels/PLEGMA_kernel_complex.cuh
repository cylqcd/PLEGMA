#include <PLEGMA.h>
#include <errno.h>
#include <mpi.h>  
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>

#ifndef PLEGMA_KERNEL_COMPLEX_CUH
#define PLEGMA_KERNEL_COMPLEX_CUH

namespace plegma {
  template<typename Float> struct Float2 {
    Float x;
    Float y;
    inline __device__ Float2() = default;
    inline __device__ Float2(float2 arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    inline __device__ Float2(double2 arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    template<typename FloatIn>
    inline __device__ Float2( Float2<FloatIn> arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    // supposing FloatIn Real
    template<typename FloatIn>
    inline __device__ Float2<Float>(FloatIn arg) {
      this->x = arg;
      this->y = 0.;
    }
    template<typename FloatIn>
    inline __device__ Float2<Float>(FloatIn arg, COMPLEX ri) {
      if(ri == IMAG) {
	this->x = 0.;
	this->y = arg;
      } else {
	this->x = arg;
	this->y = 0.;
      }	
    }
    inline __device__ Float2<Float> conj() {
      this->y *= -1.;
      return *this;
    }
  };

  inline __device__ Float2<float> operator*(const Float2<float> a, const Float2<float> b){
    Float2<float> res;
    res.x = a.x*b.x - a.y*b.y;
    res.y = a.x*b.y + a.y*b.x;
    return res;
  }

  template<typename Float>
  inline __device__ Float2<double> operator*(const Float2<double> a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a.x*b.x - a.y*b.y;
    res.y = a.x*b.y + a.y*b.x;
    return res;
  }

  template<typename Float>
  inline __device__ Float2<Float> operator*(const Float a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a*b.x;
    res.y = a*b.y;
    return res;
  }

  template<typename Float>
  inline __device__ Float2<Float> operator+(const Float2<Float> a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
  }

  inline __device__ Float2<double> operator+(const Float2<double> a, const Float2<float> b){
    Float2<double> res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
  }

  inline __device__ Float2<double> operator+(const Float2<float> a, const Float2<double> b){
    return b+a;
  }

  template<typename Float>
  inline __device__ Float2<Float> operator-(const Float2<Float> a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  inline __device__ Float2<double> operator-(const Float2<double> a, const Float2<float> b){
    Float2<double> res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  inline __device__ Float2<double> operator-(const Float2<float> a, const Float2<double> b){
    Float2<double> res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  template<typename Float>
  inline __device__ Float2<Float> conj( const Float2<Float> a){
    Float2<Float> res;
    res.x = a.x;
    res.y = -a.y;
    return res;
  }


}
#endif
