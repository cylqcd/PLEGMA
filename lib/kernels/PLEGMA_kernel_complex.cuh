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
    inline __host__ __device__ Float2() = default;
    inline __host__ __device__ Float2(float2 arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    inline __host__ __device__ Float2(double2 arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    template<typename FloatIn>
    inline __host__ __device__ Float2( Float2<FloatIn> arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    // supposing FloatIn Real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>(FloatIn arg) {
      this->x = arg;
      this->y = 0.;
    }

    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>(FloatIn x, FloatIn y){
      this->x = x;
      this->y = y;
    }

    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>(FloatIn arg, COMPLEX ri) {
      if(ri == IMAG) {
	this->x = 0.;
	this->y = arg;
      } else {
	this->x = arg;
	this->y = 0.;
      }	
    }
    inline __host__ __device__ void conj() {
      this->y *= -1.;
    }

    template<typename FloatIn>
    inline __host__ __device__ void operator+=(const Float2<FloatIn> a){
      this->x = this->x + a.x;
      this->y = this->y + a.y;
    }

    template<typename FloatIn>
    inline __host__ __device__ void operator-=(const Float2<FloatIn> a){
      this->x = this->x - a.x;
      this->y = this->y - a.y;
    }

  };

  inline __host__ __device__ Float2<float> operator*(const Float2<float> a, const Float2<float> b){
    Float2<float> res;
    res.x = a.x*b.x - a.y*b.y;
    res.y = a.x*b.y + a.y*b.x;
    return res;
  }

  template<typename Float>
  inline __host__ __device__ Float2<double> operator*(const Float2<double> a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a.x*b.x - a.y*b.y;
    res.y = a.x*b.y + a.y*b.x;
    return res;
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> operator/(const Float2<Float> x, const Float2<Float> y){
  Float2<Float> res;
  res.x = (x.x * y.x + x.y * y.y) / (y.x * y.x + y.y * y.y);
  res.y = (x.y * y.x - x.x * y.y) / (y.x * y.x + y.y * y.y);
  return res;
}

  template<typename Float>
  inline __host__ __device__ Float2<Float> operator/(const Float2<Float> a, const Float b){
    Float2<Float> res;
    res.x = a.x/b;
    res.y = a.y/b;
    return res;
  }
  
  template<typename Float>
  inline __host__ __device__ Float2<Float> operator*(const Float a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a*b.x;
    res.y = a*b.y;
    return res;
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> operator+(const Float2<Float> a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
  }
  
  inline __host__ __device__ Float2<double> operator+(const Float2<double> a, const Float2<float> b){
    Float2<double> res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
  }

  inline __host__ __device__ Float2<double> operator+(const Float2<float> a, const Float2<double> b){
    return b+a;
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> operator-(const Float2<Float> a, const Float2<Float> b){
    Float2<Float> res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  inline __host__ __device__ Float2<double> operator-(const Float2<double> a, const Float2<float> b){
    Float2<double> res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  inline __host__ __device__ Float2<double> operator-(const Float2<float> a, const Float2<double> b){
    Float2<double> res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> conj( const Float2<Float> a){
    Float2<Float> res;
    res.x = a.x;
    res.y = -a.y;
    return res;
  }

  template<typename Float>
  inline __host__ __device__ Float norm2(const Float2<Float> a){
    return a.x*a.x + a.y*a.y;
  }

  template<typename Float>
  inline __host__ __device__ Float norm(const Float2<Float> a){
    return sqrt(norm2(a));
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> cpow(const Float2<Float> x , const Float a){
    Float2<Float> res;
    res.x = pow(norm(x),a) * cos( atan2(x.y,x.x) * a);
    res.y = pow(norm(x),a) * sin( atan2(x.y,x.x) * a);
    return res;
  }

}
#endif
