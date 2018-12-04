#ifndef PLEGMA_KERNEL_TEXTURE_CUH
#define PLEGMA_KERNEL_TEXTURE_CUH

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
#define LEXIC_PLUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]+1)%c_localL[0],c_localL) : \
                        (i==1 ? LEXIC(id[3],id[2],(id[1]+1)%c_localL[1],id[0],c_localL) : \
                        (i==2 ? LEXIC(id[3],(id[2]+1)%c_localL[2],id[1],id[0],c_localL) : \
			        LEXIC((id[3]+1)%c_localL[3],id[2],id[1],id[0],c_localL))))
#define LEXIC_MINUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]-1+c_localL[0])%c_localL[0],c_localL) : \
                         (i==1 ? LEXIC(id[3],id[2],(id[1]-1+c_localL[1])%c_localL[1],id[0],c_localL) : \
                         (i==2 ? LEXIC(id[3],(id[2]-1+c_localL[2])%c_localL[2],id[1],id[0],c_localL) : \
			         LEXIC((id[3]-1+c_localL[3])%c_localL[3],id[2],id[1],id[0],c_localL))))

namespace plegma {
  enum get_from { Me, Plus, Minus, PlusMinus, MinusPlus, MinusMinus, PlusPlus };

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
  inline __device__ void sidStride::setSidStride<PlusMinus>(size_t sid, const int offset, short int dirPlus, short int dirMinus) {
    size_t id[4] = GET_ID(sid);
    bool plus_ghost = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1));
    if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%c_localL[dirPlus]; 
    bool minus_ghost = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0);
    if(!minus_ghost) id[dirMinus] = (id[dirMinus] + c_localL[dirMinus] - 1)%c_localL[dirMinus];
    
    if(plus_ghost && minus_ghost) printf("!!!!!!!   ERROR: plus and minus ghost together need corner halos\n");
    
    this->sid = plus_ghost ? (c_plusGhost[dirPlus]*offset + LEXIC_3D(dirPlus,id)) :
      ( minus_ghost ? (c_minusGhost[dirMinus]*offset + LEXIC_3D(dirMinus,id)) : LEXIC_ID(id));
    this->stride = plus_ghost ? c_surface[dirPlus] : ( minus_ghost ? c_surface[dirMinus] : c_stride);
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
    inline __device__ Float2<Float> get(int i, size_t sid, int stride) {
      return texture<Float>::fetch(i*stride + sid);
    }
    inline __device__ Float2<Float> get(int i, sidStride &ss) {
      return texture<Float>::fetch(i*ss.stride + ss.sid);
    }
    inline __device__ void set(int i, size_t sid, int stride, Float2<Float> v){;}
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

  template<typename Float>
  struct pFloat2 {
    Float2<Float>* p;
    inline __device__ pFloat2(Float* pointer) {
      p = (Float2<Float> *) pointer;
    }
    inline __device__ Float2<Float> get(int i, size_t sid, int stride) {
      return p[i*stride + sid];
    }
    inline __device__ Float2<Float> get(int i, sidStride &ss) {
      return p[i*ss.stride + ss.sid];
    }
    inline __device__ void set(int i, size_t sid, int stride, Float2<Float> v){
      p[i*stride + sid] = v;
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
      get(p, ss); 
    }    
  };

  template<typename Float>
  struct genericTex : generic<texture<Float>,Float> {using generic<texture<Float>,Float>::generic;};

  template<typename Float>
  struct generic2 : generic<pFloat2<Float>,Float> {using generic<pFloat2<Float>,Float>::generic;};

  template<typename T, typename Float>
  struct genericGauge : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int mu, short int c1, short int c2, size_t sid, Float2<Float> v) {
      sidStride ss(sid);
      T::set(((mu*N_COLS + c1)*N_COLS + c2), ss, v);
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
      return T::get(((mu*N_COLS + c1)*N_COLS + c2), ss);
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
  struct gaugeTex : genericGauge<texture<Float>, Float>{using genericGauge<texture<Float>, Float>::genericGauge;};

  template<typename Float>
  struct gauge2 : genericGauge<pFloat2<Float>,Float >{using genericGauge<pFloat2<Float>,Float >::genericGauge;};


  template<typename T, typename Float>
  struct genericSu3 : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int c1, short int c2, size_t sid, Float2<Float> v) {
      sidStride ss(sid);
      T::set((c1*N_COLS + c2), ss, v);
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
      return T::get((c1*N_COLS + c2), ss);
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
  struct su3Tex : genericSu3<texture<Float>, Float>{using genericSu3<texture<Float>, Float>::genericSu3;};

  template<typename Float>
  struct su3_2 : genericSu3<pFloat2<Float>,Float >{using genericSu3<pFloat2<Float>,Float >::genericSu3;};

  
  template<typename T,typename Float>
  struct genericVector : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int mu, short int c, size_t sid, Float2<Float> v) {
      return T::set((mu*N_COLS + c),sid,v);
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
      return T::get((mu*N_COLS + c),ss);
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
  struct vectorTex : genericVector< texture<Float>, Float > {using genericVector< texture<Float>, Float >::genericVector;};

  template<typename Float>
  struct vector2 : genericVector< pFloat2<Float>, Float > {using genericVector< pFloat2<Float>, Float >::genericVector;};

  template<typename T, typename Float>
    struct genericProp : generic<T,Float>  {
    using generic<T,Float>::generic;
    inline __device__ void set(short int mu, short int nu, short int c1, short int c2, size_t sid, Float2<Float> v) {
      T::set((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2),sid,v);
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
      return T::get((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2), ss);
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
  struct propTex : genericProp< texture<Float>, Float > {using genericProp< texture<Float>, Float >::genericProp;};

  template<typename Float>
  struct prop2 : genericProp< pFloat2<Float>, Float > {using genericProp< pFloat2<Float>, Float >::genericProp;};

}
#endif
