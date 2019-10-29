#include <errno.h>
#include <mpi.h>  
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <PLEGMA_global.h>


#ifndef PLEGMA_KERNEL_COMPLEX_CUH
#define PLEGMA_KERNEL_COMPLEX_CUH

namespace plegma {
  template<typename Float> struct Float2 {
    Float x;
    Float y;
    inline __host__ __device__ Float2() = default;
    inline __host__ __device__ Float2(const float2& arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    inline __host__ __device__ Float2(const double2& arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    template<typename FloatIn>
    inline __host__ __device__ Float2(const Float2<FloatIn>& arg) {
      this->x = arg.x;
      this->y = arg.y;
    } 
    // supposing FloatIn Real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>(const FloatIn& arg) {
      this->x = arg;
      this->y = 0.;
    }

    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>(const FloatIn& x, const FloatIn& y){
      this->x = x;
      this->y = y;
    }

    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>(const FloatIn& arg, const COMPLEX& ri) {
      if(ri == IMAG) {
	this->x = 0.;
	this->y = arg;
      } else {
	this->x = arg;
	this->y = 0.;
      }	
    }

    // Conjugate
    inline __host__ __device__ Float2<Float>& conj() {
      this->y *= -1.;
      return *this;
    }

    // Equality
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator=(const Float2<FloatIn>& a) {
      this->x = a.x;
      this->y = a.y;
      return *this;
    }

    // SumEq with complex
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator+=(const Float2<FloatIn>& a) {
      this->x += a.x;
      this->y += a.y;
      return *this;
    }

    // SumEq with real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator+=(const FloatIn& a) {
      this->x += a;
      return *this;
    }

    // Sum
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator+(const FloatIn& a) const {
      return Float2<Float>(*this)+=a;
    }

    // DiffEq with complex
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator-=(const Float2<FloatIn>& a) {
      this->x -= a.x;
      this->y -= a.y;
      return *this;
    }

    // DiffEq with real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator-=(const FloatIn& a) {
      this->x -= a;
      return *this;
    }

    // Diff
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator-(const FloatIn& a) const {
      return Float2<Float>(*this)-=a;
    }

    // Mul with complex
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator*(const Float2<FloatIn>& a) const {
      Float2<Float> res(*this);
      res.x = a.x*this->x - a.y*this->y;
      res.y = a.x*this->y + a.y*this->x;
      return res;
    }
    
    // MulEq with complex
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator*=(const Float2<FloatIn>& a) {
      return *this = (*this*a);
    }

    // MulEq with real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator*=(const FloatIn& a) {
      this->x *= a;
      this->y *= a;
      return *this;
    }

    // Mul with real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator*(const FloatIn& a) const {
      return Float2<Float>(*this)*=a;
    }

    // Div with complex
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator/(const Float2<FloatIn>& a) const {
      Float2<Float> res;
      res.x = (this->x * a.x + this->y * a.y) / (a.x * a.x + a.y * a.y);
      res.y = (this->y * a.x - this->x * a.y) / (a.x * a.x + a.y * a.y);
      return res;
    }

    // DivEq with complex
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator/=(const Float2<FloatIn>& a) {
      return *this = *this/a;
    }

    // DivEq with real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& operator/=(const FloatIn& a) {
      this->x /= a;
      this->y /= a;
      return *this;
    }
    
    // Div with real
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator/(const FloatIn& a) const {
      return Float2<Float>(*this)/=a;
    }

    // Norm
    inline __host__ __device__ Float norm2() const {
      return this->x*this->x + this->y*this->y;
    }
    inline __host__ __device__ Float norm() const {
      return sqrt(norm2());
    }

    // Power
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float>& cpow(const FloatIn& a) {
      Float _atan = atan2(this->y,this->x);
      Float _norm = norm();
      this->x = pow(_norm,a) * cos(_atan*a);
      this->y = pow(_norm,a) * sin(_atan*a);
      return *this;
    }
  };

  template<typename FloatA, typename FloatB>
  inline __host__ __device__ Float2<FloatB> operator*(const FloatA& a, const Float2<FloatB>& b) {
    return b*a;
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> conj( const Float2<Float>& a) {
    return Float2<Float>(a).conj();
  }

  template<typename Float>
  inline __host__ __device__ Float norm2(const Float2<Float>& a) {
    return a.norm2();
  }

  template<typename Float>
  inline __host__ __device__ Float norm(const Float2<Float>& a) {
    return a.norm();
  }

  template<typename FloatX, typename FloatA>
  inline __host__ __device__ Float2<FloatX> cpow(const Float2<FloatX>& x , const FloatA& a) {
    return Float2<FloatX>(x).cpow(a);
  }

}
#endif
