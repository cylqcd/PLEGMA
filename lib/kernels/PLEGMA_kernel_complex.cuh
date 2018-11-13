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

  template<typename Float>
  struct Gauge2 {
    Float2<Float> * arr;
    inline __device__ Gauge2(Float* pointer) {
      arr = (Float2<Float> *) pointer;
    }
    inline __device__ Float2<Float> get(short int dir, int a, int b, int sid, int stride) {
      return arr[((dir*N_COLS + a)*N_COLS + b)*stride + sid];
    }
    inline __device__ Float2<Float> get(short int dir, int a, int b, int sid) {
      return get(dir,a,b,sid,c_stride);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], short int dir, int sid, int stride) {
      #pragma unroll
      for(int a=0; a<N_COLS; a++) {
        #pragma unroll
	for(int b=0; b<N_COLS; b++) {
	  G[a][b] = get(dir, a, b, sid, stride);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], short int dir, int sid) {
      get(G, dir, sid, c_stride);
    }
    inline __device__ void getPlus(Float2<Float> G[N_COLS][N_COLS], short int dirLink, short int dirPlus, int sid) {
      int id[4] = GET_ID(sid);
      int sidPlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ?
	(c_plusGhost[dirPlus]*N_DIMS*N_COLS*N_COLS + LEXIC_3D(dirPlus,id)) : LEXIC_PLUS(dirPlus, id);
      int stridePlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ? c_surface[dirPlus] : c_stride;
      get(G, dirLink, sidPlus, stridePlus);
    }
    inline __device__ void getMinus(Float2<Float> G[N_COLS][N_COLS], short int dirLink, short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      int sidMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ?
	(c_minusGhost[dirMinus]*N_DIMS*N_COLS*N_COLS + LEXIC_3D(dirMinus,id)) : LEXIC_MINUS(dirMinus, id);
      int strideMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ? c_surface[dirMinus] : c_stride;
      get(G, dirLink, sidMinus, strideMinus);
    }
  };

}
#endif
