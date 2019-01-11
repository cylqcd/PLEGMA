#ifndef PLEGMA_KERNEL_TEXTURE_CUH
#define PLEGMA_KERNEL_TEXTURE_CUH

#include <PLEGMA_kernel_counter.cuh>

#define GET_ID(sid) {(sid) % c_localL[0],				\
		     ((sid)/c_localL[0]) % c_localL[1],			\
		     ((sid)/c_localL[0]/c_localL[1]) % c_localL[2],	\
		     ((sid)/c_localL[0]/c_localL[1]/c_localL[2]) % c_localL[3] }

#define GET_ID_ZYX(sid) {(sid) % c_localL[0],				\
			 ((sid)/c_localL[0]) % c_localL[1],		\
			 ((sid)/c_localL[0]/c_localL[1]) % c_localL[2] }

#define LEXIC_ID(id) LEXIC(id[3],id[2],id[1],id[0],c_localL)
#define LEXIC_3D(i,id)(i==0 ? LEXIC_TZY(id[3],id[2],id[1],c_localL) : \
                      (i==1 ? LEXIC_TZX(id[3],id[2],id[0],c_localL) : \
		      (i==2 ? LEXIC_TYX(id[3],id[1],id[0],c_localL) : \
		              LEXIC_ZYX(id[2],id[1],id[0],c_localL))))
#define LEXIC_NOX(j,id)( j==1 ? LEXIC_TZ(id[3],id[2],c_localL) : \
			 ( j==2 ? LEXIC_TY(id[3],id[1],c_localL) : \
			   LEXIC_ZY(id[2],id[1],c_localL) ) )
#define	LEXIC_NOY(j,id)( j==0 ? LEXIC_TZ(id[3],id[2],c_localL) : \
			 ( j==2 ? LEXIC_TX(id[3],id[0],c_localL) : \
			   LEXIC_ZX(id[2],id[0],c_localL) ) )
#define LEXIC_NOZ(j,id)( j==0 ? LEXIC_TY(id[3],id[1],c_localL) : \
			 ( j==1 ? LEXIC_TX(id[3],id[0],c_localL) : \
			   LEXIC_YX(id[1],id[0],c_localL) ) )
#define LEXIC_NOT(j,id)( j==0 ? LEXIC_ZY(id[2],id[1],c_localL) : \
			 ( j==1 ? LEXIC_ZX(id[2],id[0],c_localL) : \
			   LEXIC_YX(id[1],id[0],c_localL)  ) )
// assuming i!=j
#define LEXIC_2D(i,j,id)( i==0 ? LEXIC_NOX(j,id) :   \
			  ( i==1 ? LEXIC_NOY(j,id) : \
			    ( i==2 ? LEXIC_NOZ(j,id) : LEXIC_NOT(j,id) )))
#define LEXIC_PLUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]+1)%c_localL[0],c_localL) : \
                        (i==1 ? LEXIC(id[3],id[2],(id[1]+1)%c_localL[1],id[0],c_localL) : \
                        (i==2 ? LEXIC(id[3],(id[2]+1)%c_localL[2],id[1],id[0],c_localL) : \
			        LEXIC((id[3]+1)%c_localL[3],id[2],id[1],id[0],c_localL))))
#define LEXIC_MINUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]-1+c_localL[0])%c_localL[0],c_localL) : \
                         (i==1 ? LEXIC(id[3],id[2],(id[1]-1+c_localL[1])%c_localL[1],id[0],c_localL) : \
                         (i==2 ? LEXIC(id[3],(id[2]-1+c_localL[2])%c_localL[2],id[1],id[0],c_localL) : \
			         LEXIC((id[3]-1+c_localL[3])%c_localL[3],id[2],id[1],id[0],c_localL))))

namespace plegma {
  enum get_from { Me, Plus, Minus, PlusPlus, MinusMinus, PlusMinus, MinusPlus };

  struct sidStride {
    size_t sid;
    size_t stride;
    inline __device__ sidStride() = default; 
    inline __device__ sidStride(size_t sid) {
      this->sid = sid;
      this->stride = c_stride;      
    }  
    template<get_from src>
    inline __device__ void setSidStride(size_t sid, const int, short int);
    template<get_from src>
    inline __device__ void setSidStride(size_t sid, const int, short int, short int);
  };
  template<>
  inline __device__ void sidStride::setSidStride<Plus>(size_t sid, const int offset, short int dirPlus) {
    size_t id[4] = GET_ID(sid);
    bool plus_ghost = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1));
    this->sid = plus_ghost ? (c_plusGhost[dirPlus]*offset + LEXIC_3D(dirPlus,id)) : LEXIC_PLUS(dirPlus, id);
    this->stride = plus_ghost ? c_surface[dirPlus] : c_stride;
  }
  template<>
  inline __device__ void sidStride::setSidStride<Minus>(size_t sid, const int offset, short int dirMinus) {
    size_t id[4] = GET_ID(sid);
    bool minus_ghost = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0);
    this->sid = minus_ghost ? (c_minusGhost[dirMinus]*offset + LEXIC_3D(dirMinus,id)) : LEXIC_MINUS(dirMinus, id);
    this->stride = minus_ghost ? c_surface[dirMinus] : c_stride;
  }
  template<>
  inline __device__ void sidStride::setSidStride<PlusPlus>(size_t sid, const int offset, short int dirPlus1, short int dirPlus2) {
    if(dirPlus1 == dirPlus2 && c_dimBreak[dirPlus1]) {
      printf(" !!! ERROR: in PlusPlus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus1_ghost = c_dimBreak[dirPlus1] == true && id[dirPlus1] == (c_localL[dirPlus1]-1);
      if(!plus1_ghost) id[dirPlus1] = (id[dirPlus1] + 1)%c_localL[dirPlus1]; 
      bool plus2_ghost = c_dimBreak[dirPlus2] == true && id[dirPlus2] == (c_localL[dirPlus1]-1);
      if(!plus2_ghost) id[dirPlus2] = (id[dirPlus2] + 1)%c_localL[dirPlus2];

      if(plus1_ghost && plus2_ghost){
	this->sid = c_cornerGhost[dirPlus1][dirPlus2]*offset + LEXIC_2D(dirPlus1,dirPlus2,id);
	this->stride = c_surface2D[dirPlus1][dirPlus2];
      } else if(plus1_ghost) {
	this->sid = c_plusGhost[dirPlus1]*offset + LEXIC_3D(dirPlus1,id);
	this->stride = c_surface[dirPlus1];
      } else if(plus2_ghost) {
	this->sid = c_plusGhost[dirPlus2]*offset + LEXIC_3D(dirPlus2,id);
	this->stride = c_surface[dirPlus2];
      } else {
	this->sid = LEXIC_ID(id);
	this->stride = c_stride;
      }
    }
  }
  template<>
  inline __device__ void sidStride::setSidStride<MinusMinus>(size_t sid, const int offset, short int dirMinus1, short int dirMinus2) {
    if(dirMinus1 == dirMinus2 && c_dimBreak[dirMinus1]) {
      printf(" !!! ERROR: in MinusMinus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool minus1_ghost = c_dimBreak[dirMinus1] == true && id[dirMinus1] == 0;
      if(!minus1_ghost) id[dirMinus1] = (id[dirMinus1] + c_localL[dirMinus1] - 1)%c_localL[dirMinus1]; 
      bool minus2_ghost = c_dimBreak[dirMinus2] == true && id[dirMinus2] == 0;
      if(!minus2_ghost) id[dirMinus2] = (id[dirMinus2] + c_localL[dirMinus2] - 1)%c_localL[dirMinus2];

      if(minus1_ghost && minus2_ghost){
	this->sid = c_cornerGhost[N_DIMS+dirMinus1][N_DIMS+dirMinus2]*offset + LEXIC_2D(dirMinus1,dirMinus2,id);
	this->stride = c_surface2D[dirMinus1][dirMinus2];
      } else if(minus1_ghost) {
	this->sid = c_minusGhost[dirMinus1]*offset + LEXIC_3D(dirMinus1,id);
	this->stride = c_surface[dirMinus1];
      } else if(minus2_ghost) {
	this->sid = c_minusGhost[dirMinus2]*offset + LEXIC_3D(dirMinus2,id);
	this->stride = c_surface[dirMinus2];
      } else {
	this->sid = LEXIC_ID(id);
	this->stride = c_stride;
      }
    }
  }
  template<>
  inline __device__ void sidStride::setSidStride<PlusMinus>(size_t sid, const int offset, short int dirPlus, short int dirMinus) {
    if(dirPlus == dirMinus) {
      this->sid = sid;
      this->stride = c_stride;      
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus_ghost = c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1);
      if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%c_localL[dirPlus]; 
      bool minus_ghost = c_dimBreak[dirMinus] == true && id[dirMinus] == 0;
      if(!minus_ghost) id[dirMinus] = (id[dirMinus] + c_localL[dirMinus] - 1)%c_localL[dirMinus];

      if(plus_ghost && minus_ghost){
	this->sid = c_cornerGhost[dirPlus][N_DIMS+dirMinus]*offset + LEXIC_2D(dirPlus,dirMinus,id);
	this->stride = c_surface2D[dirPlus][dirMinus];
      } else if(plus_ghost) {
	this->sid = c_plusGhost[dirPlus]*offset + LEXIC_3D(dirPlus,id);
	this->stride = c_surface[dirPlus];
      } else if(minus_ghost) {
	this->sid = c_minusGhost[dirMinus]*offset + LEXIC_3D(dirMinus,id);
	this->stride = c_surface[dirMinus];
      } else {
	this->sid = LEXIC_ID(id);
	this->stride = c_stride;
      }
    }
  }
  template<>
  inline __device__ void sidStride::setSidStride<MinusPlus>(size_t sid, const int offset, short int dirMinus, short int dirPlus) {
    this->setSidStride<PlusMinus>(sid, offset, dirPlus, dirMinus);
  }

  template<typename Float>
  struct texture {
    cudaTextureObject_t tex;
    inline __device__ texture() = default; 
    inline __device__ texture(cudaTextureObject_t t) {
      tex = t;
    }
    // Fetch is going to be specialized after
    inline __device__ Float2<Float> fetch(size_t i);
    inline __device__ Float2<Float> get(int i, sidStride &ss) {
      return texture<Float>::fetch(i*ss.stride + ss.sid);
    }
    inline __device__ void set(int i, sidStride &ss, Float2<Float> v);
  };

  // Here we specialize fetch
  template<> inline __device__ Float2<float> texture<float>::fetch(size_t i) {
    return (Float2<float>) tex1Dfetch<float2>(tex,i);  
  }
  template<> inline __device__ Float2<double> texture<double>::fetch(size_t i) {
    int4 v = tex1Dfetch<int4>(tex,i);
    return (Float2<double>) make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
  template<> inline __device__ Float2<counter> texture<counter>::fetch(size_t i) {
    counter::incReads();
    return Float2<counter>();
  }
  
  template<typename Float>
  struct pFloat2 {
    Float2<Float>* p;
    inline __device__ pFloat2(Float* pointer) {
      p = (Float2<Float> *) pointer;
    }
    inline __device__ Float2<Float> get(int i, sidStride &ss) {
      return p[i*ss.stride + ss.sid];
    }
    inline __device__ void set(int i, sidStride &ss, Float2<Float> v){
      p[i*ss.stride + ss.sid] = v;
    }
  };
    
  template<typename T, typename Float>
  struct generic : T {
    using T::T;
    inline __device__ void set(int i, size_t sid, Float2<Float> v) {
      sidStride ss(sid);
      T::set(i,ss,v);
    }
    inline __device__ void set(int i, sidStride &ss, Float2<Float> v) {
      T::set(i,ss,v);
    }
    inline __device__ Float2<Float> get(int i, sidStride &ss) {
      return T::get(i,ss);
    }
    inline __device__ Float2<Float> get(int i, size_t sid) {
      sidStride ss(sid);
      return get(i,ss);
    }
    inline __device__ void get(Float2<Float> *p, int site_size, sidStride &ss) {
#pragma unroll
      for(int i=0; i<site_size; i++) {
	p[i] = get(i,ss);
      }
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(int i, size_t sid, int site_size, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, site_size, dirs ...);
      return get(i, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> *p, size_t sid, int site_size, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, site_size, dirs ...);
      get(p, site_size, ss); 
    }    
  };
  
  template<typename Float>
  using genericTex = generic<texture<Float>,Float>;
  
  template<typename Float>
  using generic2 = generic<pFloat2<Float>,Float>;


  template<typename T, typename Float>
  struct genericGauge : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int mu, short int c1, short int c2, size_t sid, Float2<Float> v) {
      generic<T,Float>::set(((mu*N_COLS + c1)*N_COLS + c2), sid, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], short int mu, size_t sid) {
#pragma unroll
      for(short int c1=0; c1<N_COLS; c1++) {
#pragma unroll
	for(short int c2=0; c2<N_COLS; c2++) {
	  set(mu, c1, c2, sid, G[c1][c2]);
	}    
      }
    }
    inline __device__ Float2<Float> get(short int mu, short int c1, short int c2, sidStride &ss) {
      return generic<T,Float>::get(((mu*N_COLS + c1)*N_COLS + c2), ss);
    }
    inline __device__ Float2<Float> get(short int mu, short int c1, short int c2, size_t sid) {
      sidStride ss(sid);
      return get(mu, c1, c2, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(short int mu, short int c1, short int c2, size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_DIMS*N_COLS*N_COLS, dirs ...);
      return get(mu,c1,c2,ss);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], short int mu, sidStride &ss) {
#pragma unroll
      for(short int c1=0; c1<N_COLS; c1++) {
#pragma unroll
	for(short int c2=0; c2<N_COLS; c2++) {
	  G[c1][c2] = get(mu, c1, c2, ss);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], short int mu, size_t sid) {
      sidStride ss(sid);
      get(G, mu, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], short int mu, size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_DIMS*N_COLS*N_COLS, dirs ...);
      get(G, mu, ss);
    }
  };

  template<typename Float>
  using gaugeTex = genericGauge<texture<Float>, Float >;

  template<typename Float>
  using gauge2 = genericGauge<pFloat2<Float>, Float >;

  template<typename T, typename Float>
  struct genericSu3 : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int c1, short int c2, size_t sid, Float2<Float> v) {
      generic<T,Float>::set((c1*N_COLS + c2), sid, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], size_t sid) {
#pragma unroll
      for(short int c1=0; c1<N_COLS; c1++) {
#pragma unroll
	for(short int c2=0; c2<N_COLS; c2++) {
	  set(c1, c2, sid, G[c1][c2]);
	}    
      }
    }

    inline __device__ Float2<Float> get(short int c1, short int c2, sidStride &ss) {
      return generic<T,Float>::get((c1*N_COLS + c2), ss);
    }
    inline __device__ Float2<Float> get(short int c1, short int c2, size_t sid) {
      sidStride ss(sid);
      return get(c1,c2,ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(short int c1, short int c2, size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_COLS*N_COLS, dirs ...);
      return get(c1,c2,ss);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], sidStride &ss) {
      #pragma unroll
      for(short int c1=0; c1<N_COLS; c1++) {
        #pragma unroll
	for(short int c2=0; c2<N_COLS; c2++) {
	  G[c1][c2] = get(c1, c2, ss);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], size_t sid) {
      sidStride ss(sid);
      get(G, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_DIMS*N_COLS*N_COLS, dirs ...);
      get(G, ss);
    }
  };

  template<typename Float>
  using su3Tex = genericSu3<texture<Float>, Float >;

  template<typename Float>
  using su3_2 = genericSu3<pFloat2<Float>, Float >;
  
  
  template<typename T,typename Float>
  struct genericVector : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int mu, short int c, size_t sid, Float2<Float> v) {
      generic<T,Float>::set((mu*N_COLS + c),sid,v);
    }
    inline __device__ void set(Float2<Float> S[N_SPINS][N_COLS], size_t sid) {
      #pragma unroll
      for(short int mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(short int c=0; c<N_COLS; c++) {
	  set(mu,c,sid,S[mu][c]);
	}    
      }
    }
    inline __device__ Float2<Float> get(short int mu, short int c, sidStride &ss) {
      return generic<T,Float>::get((mu*N_COLS + c),ss);
    }
    inline __device__ Float2<Float> get(short int mu, short int c, size_t sid) {
      sidStride ss(sid);
      return get(mu,c,ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(short int mu, short int c, size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_SPINS*N_COLS, dirs ...);
      return get(mu,c,ss);
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], sidStride &ss) {
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(int c=0; c<N_COLS; c++) {
	  S[mu][c] = get(mu,c,ss);
	}    
      }
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], size_t sid) {
      sidStride ss(sid);
      get(S,ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_SPINS*N_COLS, dirs ...);
      return get(S,ss);
    }
  };

  template<typename Float>
  using vectorTex = genericVector< texture<Float>, Float >;

  template<typename Float>
  using vector2 = genericVector< pFloat2<Float>, Float >;

  template<typename T, typename Float>
    struct genericProp : generic<T,Float>  {
    using generic<T,Float>::generic;
    inline __device__ void set(short int mu, short int nu, short int c1, short int c2, size_t sid, Float2<Float> v) {
      generic<T,Float>::set((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2),sid,v);
    }
    inline __device__ void set(Float2<Float> P[4][4][3][3], size_t sid) {
      #pragma unroll
      for(short int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(short int nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(short int c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(short int c2 = 0 ; c2 < N_COLS ; c2++)
	      set(mu, nu, c1, c2, sid, P[mu][nu][c1][c2]);
    }

    inline __device__ Float2<Float> get(short int mu, short int nu, short int c1, short int c2, sidStride &ss) {
      return generic<T,Float>::get((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2), ss);
    }
    inline __device__ Float2<Float> get(short int mu, short int nu, int c1, int c2, size_t sid) {
      sidStride ss(sid);
      return get(mu, nu, c1, c2, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(short int mu, short int nu, short int c1, short int c2, size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_SPINS*N_SPINS*N_COLS*N_COLS, dirs ...);
      return get(mu,nu,c1,c2,ss);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], sidStride &ss) {
      #pragma unroll
      for(short int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(short int nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(short int c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(short int c2 = 0 ; c2 < N_COLS ; c2++)
	      P[mu][nu][c1][c2] = get(mu, nu, c1, c2, ss);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], size_t sid) {
      sidStride ss(sid);
      get( P, ss );
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], size_t sid, dir_t ... dirs) {
      sidStride ss;
      ss.setSidStride<src>(sid, N_SPINS*N_SPINS*N_COLS*N_COLS, dirs ...);
      return get(P,ss);
    }
  };

  template<typename Float>
  using propTex = genericProp< texture<Float>, Float >;

  template<typename Float>
  using prop2 = genericProp< pFloat2<Float>, Float >;

}
#endif
