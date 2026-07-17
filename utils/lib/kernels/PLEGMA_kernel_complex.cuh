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
    inline __host__ __device__ void conj() {
      this->y = -this->y;
    }

    // Checks if the number is zero
    inline __host__ __device__ bool isZero() {
      return this->x == 0 && this->y == 0;
    }

    // Sum with complex
    template<typename FloatIn>
    inline __host__ __device__ void operator+=(const Float2<FloatIn>& a) {
      this->x += a.x;
      this->y += a.y;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator+(const Float2<FloatIn>& a) const {
      Float2<Float> res;
      res.x = this->x + a.x;
      res.y = this->y + a.y;
      return res;
    }

    // Sum with real
    template<typename FloatIn>
    inline __host__ __device__ void operator+=(const FloatIn& a) {
      this->x += a;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator+(const FloatIn& a) const {
      Float2<Float> res;
      res.x = this->x + a;
      res.y = this->y;
      return res;
    }
    
    // Diff with complex
    template<typename FloatIn>
    inline __host__ __device__ void operator-=(const Float2<FloatIn>& a) {
      this->x -= a.x;
      this->y -= a.y;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator-(const Float2<FloatIn>& a) const {
      Float2<Float> res;
      res.x = this->x - a.x;
      res.y = this->y - a.y;
      return res;
    }

    // Diff with real
    template<typename FloatIn>
    inline __host__ __device__ void operator-=(const FloatIn& a) {
      this->x -= a;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator-(const FloatIn& a) const {
      Float2<Float> res;
      res.x = this->x - a;
      res.y = this->y;
      return res;
    }

    // Mul with complex
    template<typename FloatIn>
    inline __host__ __device__ void operator*=(const Float2<FloatIn>& a) const {
      Float2<Float> b(*this);
      this->x = a.x*b.x - a.y*b.y;
      this->y = a.x*b.y + a.y*b.x;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator*(const Float2<FloatIn>& a) const {
      Float2<Float> res;
      res.x = a.x*this->x - a.y*this->y;
      res.y = a.x*this->y + a.y*this->x;
      return res;
    }
    
    // Mul with real
    template<typename FloatIn>
    inline __host__ __device__ void operator*=(const FloatIn& a) {
      this->x *= a;
      this->y *= a;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator*(const FloatIn& a) const {
      Float2<Float> res;
      res.x = a*this->x;
      res.y = a*this->y;
      return res;
    }

    // Div with complex
    template<typename FloatIn>
    inline __host__ __device__ void operator/=(const Float2<FloatIn>& a) {
      Float2<Float> b(*this);
      FloatIn den = (a.x * a.x + a.y * a.y);
      this->x = (a.x*b.x + a.y*b.y) / den;
      this->y = (a.x*b.y - a.y*b.x) / den;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator/(const Float2<FloatIn>& a) const {
      Float2<Float> res;
      FloatIn den = (a.x * a.x + a.y * a.y);
      res.x = (this->x * a.x + this->y * a.y) / den;
      res.y = (this->y * a.x - this->x * a.y) / den;
      return res;
    }
    
    // Div with real
    template<typename FloatIn>
    inline __host__ __device__ void operator/=(const FloatIn& a) {
      this->x /= a;
      this->y /= a;
    }
    template<typename FloatIn>
    inline __host__ __device__ Float2<Float> operator/(const FloatIn& a) const {
      Float2<Float> res;
      res.x = this->x/a;
      res.y = this->y/a;
      return res;
    }
    
    // Norm
    inline __host__ __device__ Float norm2() const {
      return this->x*this->x + this->y*this->y;
    }
    inline __host__ __device__ Float norm() const {
      return sqrt(norm2());
    }
    inline __host__ __device__ Float givex() const {
      return this->x;
    }
    inline __host__ __device__ Float givey() const {
      return this->y;
    }

    // Power
    template<typename FloatIn>
    inline __host__ __device__ void cpow(const FloatIn& a) {
      Float _atan = atan2(this->y,this->x);
      Float _norm = norm();
      this->x = pow(_norm,a) * cos(_atan*a);
      this->y = pow(_norm,a) * sin(_atan*a);
    }
  };

  template<typename FloatA, typename FloatB>
  inline __host__ __device__ Float2<FloatB> operator*(const FloatA& a, const Float2<FloatB>& b) {
    return b*a;
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> conj(const Float2<Float>& a) {
    Float2<Float> res;
    res.x = a.x;
    res.y = -a.y;
    return res;
  }

  template<typename Float>
  inline __host__ __device__ Float norm2(const Float2<Float>& a) {
    return a.norm2();
  }

  template<typename Float>
  inline __host__ __device__ Float norm(const Float2<Float>& a) {
    return a.norm();
  }

  template<typename Float>
  inline __host__ __device__ Float2<Float> cpow(const Float2<Float>& x , const Float& a) {
    Float _atan = atan2(x.y,x.x);
    Float _norm = x.norm();
    Float2<Float> res;
    res.x = pow(_norm,a) * cos(_atan*a);
    res.y = pow(_norm,a) * sin(_atan*a);
    return res;
  }
}
#endif
