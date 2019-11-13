#ifndef PLEGMA_KERNEL_TEXTURE_CUH
#define PLEGMA_KERNEL_TEXTURE_CUH

#define GET_ID(sid) {(sid) % DGC_localL[0],				\
		     ((sid)/DGC_localL[0]) % DGC_localL[1],			\
		     ((sid)/DGC_localL[0]/DGC_localL[1]) % DGC_localL[2],	\
		     ((sid)/DGC_localL[0]/DGC_localL[1]/DGC_localL[2]) % DGC_localL[3] }

#define GET_ID_ZYX(sid) {(sid) % DGC_localL[0],				\
			 ((sid)/DGC_localL[0]) % DGC_localL[1],		\
			 ((sid)/DGC_localL[0]/DGC_localL[1]) % DGC_localL[2] }

#define LEXIC_ID(id) LEXIC(id[3],id[2],id[1],id[0],DGC_localL)
#define LEXIC_3D(i,id)(i==0 ? LEXIC_TZY(id[3],id[2],id[1],DGC_localL) : \
                      (i==1 ? LEXIC_TZX(id[3],id[2],id[0],DGC_localL) : \
		      (i==2 ? LEXIC_TYX(id[3],id[1],id[0],DGC_localL) : \
		              LEXIC_ZYX(id[2],id[1],id[0],DGC_localL))))
#define LEXIC_NOX(j,id)( j==1 ? LEXIC_TZ(id[3],id[2],DGC_localL) : \
			 ( j==2 ? LEXIC_TY(id[3],id[1],DGC_localL) : \
			   LEXIC_ZY(id[2],id[1],DGC_localL) ) )
#define	LEXIC_NOY(j,id)( j==0 ? LEXIC_TZ(id[3],id[2],DGC_localL) : \
			 ( j==2 ? LEXIC_TX(id[3],id[0],DGC_localL) : \
			   LEXIC_ZX(id[2],id[0],DGC_localL) ) )
#define LEXIC_NOZ(j,id)( j==0 ? LEXIC_TY(id[3],id[1],DGC_localL) : \
			 ( j==1 ? LEXIC_TX(id[3],id[0],DGC_localL) : \
			   LEXIC_YX(id[1],id[0],DGC_localL) ) )
#define LEXIC_NOT(j,id)( j==0 ? LEXIC_ZY(id[2],id[1],DGC_localL) : \
			 ( j==1 ? LEXIC_ZX(id[2],id[0],DGC_localL) : \
			   LEXIC_YX(id[1],id[0],DGC_localL)  ) )

#define LEXIC_ID_3D4D(id,is4D) (is4D ? LEXIC_ID(id) : LEXIC_ZYX(id[2],id[1],id[0],DGC_localL))

// assuming i!=j
#define LEXIC_2D(i,j,id)( i==0 ? LEXIC_NOX(j,id) :   \
			( i==1 ? LEXIC_NOY(j,id) :			\
			( i==2 ? LEXIC_NOZ(j,id) : LEXIC_NOT(j,id) )))
#define LEXIC_PLUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]+1)%DGC_localL[0],DGC_localL) : \
                        (i==1 ? LEXIC(id[3],id[2],(id[1]+1)%DGC_localL[1],id[0],DGC_localL) : \
                        (i==2 ? LEXIC(id[3],(id[2]+1)%DGC_localL[2],id[1],id[0],DGC_localL) : \
			        LEXIC((id[3]+1)%DGC_localL[3],id[2],id[1],id[0],DGC_localL))))
#define LEXIC_MINUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]-1+DGC_localL[0])%DGC_localL[0],DGC_localL) : \
                         (i==1 ? LEXIC(id[3],id[2],(id[1]-1+DGC_localL[1])%DGC_localL[1],id[0],DGC_localL) : \
                         (i==2 ? LEXIC(id[3],(id[2]-1+DGC_localL[2])%DGC_localL[2],id[1],id[0],DGC_localL) : \
			         LEXIC((id[3]-1+DGC_localL[3])%DGC_localL[3],id[2],id[1],id[0],DGC_localL))))

#define LEXIC_3D_PLUS(i,id)(i==0 ? LEXIC_ZYX(id[2],id[1],(id[0]+1)%DGC_localL[0],DGC_localL) : \
			   (i==1 ? LEXIC_ZYX(id[2],(id[1]+1)%DGC_localL[1],id[0],DGC_localL) : \
			           LEXIC_ZYX((id[2]+1)%DGC_localL[2],id[1],id[0],DGC_localL)))
#define LEXIC_3D_MINUS(i,id)(i==0 ? LEXIC_ZYX(id[2],id[1],(id[0]-1+DGC_localL[0])%DGC_localL[0],DGC_localL) : \
			    (i==1 ? LEXIC_ZYX(id[2],(id[1]-1+DGC_localL[1])%DGC_localL[1],id[0],DGC_localL) : \
			            LEXIC_ZYX((id[2]-1+DGC_localL[2])%DGC_localL[2],id[1],id[0],DGC_localL)))

#define LEXIC_3D4D_PLUS(i,id,is4D) (is4D ? LEXIC_PLUS(i,id) : LEXIC_3D_PLUS(i,id))
#define LEXIC_3D4D_MINUS(i,id,is4D) (is4D ? LEXIC_MINUS(i,id) : LEXIC_3D_MINUS(i,id))

namespace plegma {
  enum get_from { Me, Plus, Minus, PlusPlus, MinusMinus, PlusMinus, MinusPlus, PlusNoGhost, MinusNoGhost, PlusOnlyGhost, MinusOnlyGhost};

  struct sidStride {
    const bool is4D;
    bool returnZero;
    size_t sid;
    size_t stride;
    inline __device__ sidStride(const size_t& sid, const bool& is4D = true, const bool& returnZero = false) :
      is4D(is4D), returnZero(returnZero), sid(sid), stride(is4D ? DGC_localVolume : DGC_localVolume3D) { }

    template<get_from src>
    inline __device__ void shift(const int& site_size, const short& dir);
    
    template<get_from src>
    inline __device__ void shift(const int& site_size, const short& dir1, const short& dir2);
    
    inline __device__ void accessSideGhost(const size_t& sid3D, const int& site_size, const short& dir, const ORIENTATION& sign) {
      size_t volume = is4D ? DGC_localVolume : DGC_localVolume3D;
      size_t sideGhost = DGC_sideGhost[dir][sign];
      this->stride = DGC_surface3D[dir];
      if(not is4D) {
	sideGhost /= DGC_localL[DIM_T];
	this->stride /= DGC_localL[DIM_T];
      }
      this->sid = (volume+sideGhost)*site_size + sid3D;
    }
    
    inline __device__ void accessCornerGhost(const size_t& sid2D, const int& site_size, const short& dir1, const short& dir2, const ORIENTATION& sign1, const ORIENTATION& sign2) {
      size_t volume = is4D ? DGC_localVolume : DGC_localVolume3D;
      size_t sideGhostVolume = is4D ? DGC_sideGhostVolume : DGC_sideGhostVolume3D;
      size_t cornerGhost = DGC_cornerGhost[dir1][dir2][sign1][sign2];
      this->stride = DGC_surface2D[dir1][dir2];
      if(not is4D) {
	cornerGhost /= DGC_localL[DIM_T];
	this->stride /= DGC_localL[DIM_T];
      }
      this->sid = (volume+sideGhostVolume+cornerGhost)*site_size + sid2D;
    }
  };


  template<>
  inline __device__ void sidStride::shift<Plus>(const int& site_size, const short& dirPlus) {
    size_t id[4] = GET_ID(sid);
    bool plus_ghost = (DGC_dimBreak[dirPlus] == true && id[dirPlus] == (DGC_localL[dirPlus]-1));
    if(plus_ghost) {
      this->accessSideGhost(LEXIC_3D(dirPlus,id), site_size, dirPlus, DIR_PLUS);
    } else {
      this->sid = LEXIC_3D4D_PLUS(dirPlus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<Minus>(const int& site_size, const short& dirMinus) {
    size_t id[4] = GET_ID(sid);
    bool minus_ghost = (DGC_dimBreak[dirMinus] == true && id[dirMinus] == 0);
    if(minus_ghost) {
      this->accessSideGhost(LEXIC_3D(dirMinus,id), site_size, dirMinus, DIR_MINUS);
    } else {
      this->sid = LEXIC_3D4D_MINUS(dirMinus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusOnlyGhost>(const int& site_size, const short& dirPlus) {
    size_t id[4] = GET_ID(sid);
    bool plus_ghost = (DGC_dimBreak[dirPlus] == true && id[dirPlus] == (DGC_localL[dirPlus]-1));
    if(plus_ghost) {
      this->accessSideGhost(LEXIC_3D(dirPlus,id), site_size, dirPlus, DIR_PLUS);
    } else {
      this->returnZero = true;
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusOnlyGhost>(const int& site_size, const short& dirMinus) {
    size_t id[4] = GET_ID(sid);
    bool minus_ghost = (DGC_dimBreak[dirMinus] == true && id[dirMinus] == 0);
    if(minus_ghost) {
      this->accessSideGhost(LEXIC_3D(dirMinus,id), site_size, dirMinus, DIR_MINUS);
    } else {
      this->returnZero = true;
    }
  }
  
  template<>
  inline __device__ void sidStride::shift<PlusNoGhost>(const int& site_size, const short& dirPlus) {
    size_t id[4] = GET_ID(sid);
    bool plus_ghost = (DGC_dimBreak[dirPlus] == true && id[dirPlus] == (DGC_localL[dirPlus]-1));
    if(plus_ghost) {
      this->returnZero = true;
    } else {
      this->sid = LEXIC_3D4D_PLUS(dirPlus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusNoGhost>(const int& site_size, const short& dirMinus) {
    size_t id[4] = GET_ID(sid);
    bool minus_ghost = (DGC_dimBreak[dirMinus] == true && id[dirMinus] == 0);
    if(not minus_ghost) {
      this->returnZero = true;
    } else {
      this->sid = LEXIC_3D4D_MINUS(dirMinus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusPlus>(const int& site_size, const short& dirPlus1, const short& dirPlus2) {
    if(dirPlus1 == dirPlus2 && DGC_dimBreak[dirPlus1]) {
      printf(" !!! ERROR: in PlusPlus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus1_ghost = DGC_dimBreak[dirPlus1] == true && id[dirPlus1] == (DGC_localL[dirPlus1]-1);
      if(!plus1_ghost) id[dirPlus1] = (id[dirPlus1] + 1)%DGC_localL[dirPlus1]; 
      bool plus2_ghost = DGC_dimBreak[dirPlus2] == true && id[dirPlus2] == (DGC_localL[dirPlus1]-1);
      if(!plus2_ghost) id[dirPlus2] = (id[dirPlus2] + 1)%DGC_localL[dirPlus2];

      if(plus1_ghost && plus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus1,dirPlus2,id), site_size, dirPlus1, dirPlus2, DIR_PLUS, DIR_PLUS);
      } else if(plus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus1,id), site_size, dirPlus1, DIR_PLUS);
      } else if(plus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus2,id), site_size, dirPlus2, DIR_PLUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusMinus>(const int& site_size, const short& dirMinus1, const short& dirMinus2) {
    if(dirMinus1 == dirMinus2 && DGC_dimBreak[dirMinus1]) {
      printf(" !!! ERROR: in MinusMinus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool minus1_ghost = DGC_dimBreak[dirMinus1] == true && id[dirMinus1] == 0;
      if(!minus1_ghost) id[dirMinus1] = (id[dirMinus1] + DGC_localL[dirMinus1] - 1)%DGC_localL[dirMinus1]; 
      bool minus2_ghost = DGC_dimBreak[dirMinus2] == true && id[dirMinus2] == 0;
      if(!minus2_ghost) id[dirMinus2] = (id[dirMinus2] + DGC_localL[dirMinus2] - 1)%DGC_localL[dirMinus2];

      if(minus1_ghost && minus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus1,dirMinus2,id), site_size, dirMinus1, dirMinus2, DIR_MINUS, DIR_MINUS);
      } else if(minus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus1,id), site_size, dirMinus1, DIR_MINUS);
      } else if(minus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus2,id), site_size, dirMinus2, DIR_MINUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusMinus>(const int& site_size, const short& dirPlus, const short& dirMinus) {
    if(dirPlus == dirMinus) {
      return;
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus_ghost = DGC_dimBreak[dirPlus] == true && id[dirPlus] == (DGC_localL[dirPlus]-1);
      if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%DGC_localL[dirPlus]; 
      bool minus_ghost = DGC_dimBreak[dirMinus] == true && id[dirMinus] == 0;
      if(!minus_ghost) id[dirMinus] = (id[dirMinus] + DGC_localL[dirMinus] - 1)%DGC_localL[dirMinus];

      if(plus_ghost && minus_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus,dirMinus,id), site_size, dirPlus, dirMinus, DIR_PLUS, DIR_MINUS);
      } else if(plus_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus,id), site_size, dirPlus, DIR_PLUS);
      } else if(minus_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus,id), site_size, dirMinus, DIR_MINUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusPlus>(const int& site_size, const short& dirMinus, const short& dirPlus) {
    this->shift<PlusMinus>(site_size, dirPlus, dirMinus);
  }

  template<typename Float>
  struct texture {
    cudaTextureObject_t tex;
    inline __host__ __device__ texture() = default; 
    inline __host__ __device__ texture(const cudaTextureObject_t& t) {
      tex = t;
    }
    // Fetch is going to be specialized after
    inline __device__ Float2<Float> fetch(const size_t& i) const;
    inline __device__ Float2<Float> get(const int& i, const sidStride& ss) const {
      return ss.returnZero ? Float2<Float>(0) : texture<Float>::fetch(i*ss.stride + ss.sid);
    }
    inline __device__ void set(const int& i, const sidStride& ss, const Float2<Float>& v);
  };

  // Here we specialize fetch
  template<> inline __device__ Float2<float> texture<float>::fetch(const size_t& i) const {
    return (Float2<float>) tex1Dfetch<float2>(tex,i);  
  }
  template<> inline __device__ Float2<double> texture<double>::fetch(const size_t& i) const {
    int4 v = tex1Dfetch<int4>(tex,i);
    return (Float2<double>) make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
    
  template<typename Float>
  struct pFloat2 {
    Float2<Float>* p;
    inline __host__ __device__ pFloat2() = default; 
    inline __host__ __device__ pFloat2(const Float* pointer) {
      p = (Float2<Float> *) pointer;
    }
    inline __host__ __device__ Float2<Float> get(const int& i, const sidStride& ss) const {
      return ss.returnZero ? Float2<Float>(0) : p[i*ss.stride + ss.sid];
    }
    inline __host__ __device__ void set(const int& i, const sidStride& ss, const Float2<Float>& v) {
      p[i*ss.stride + ss.sid] = v;
    }
  };
    
  template<typename T, typename Float>
  struct generic : T {
    using T::T;
    bool is4D = true;
    bool returnZero = false;
    inline __device__ void set(const int& i, const size_t& sid, const Float2<Float>& v) {
      sidStride ss(sid,this->is4D,this->returnZero);
      T::set(i,ss,v);
    }
    inline __device__ void set(const int& i, const sidStride& ss, const Float2<Float>& v) {
      T::set(i,ss,v);
    }
    inline __device__ Float2<Float> get(const int& i, const sidStride& ss) const {
      return T::get(i,ss);
    }
    inline __device__ Float2<Float> get(const int& i, const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      return T::get(i,ss);
    }
    inline __device__ void get(Float2<Float> *p, const int& site_size, const sidStride& ss) const {
      #pragma unroll
      for(int i=0; i<site_size; i++) {
	p[i] = get(i,ss);
      }
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const int& i, const size_t& sid, const int& site_size, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(site_size, dirs ...);
      return get(i, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> *p, const size_t& sid, const int& site_size, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(site_size, dirs ...);
      get(p, site_size, ss); 
    }    
  };
  
  template<typename Float>
  using genericTex = generic<texture<Float>, Float>;
  
  template<typename Float>
  using generic2 = generic<pFloat2<Float>, Float>;


  template<typename T, typename Float>
  struct genericGauge : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& mu, const short& c1, const short& c2, const size_t& sid, const Float2<Float>& v) {
      generic<T,Float>::set(((mu*N_COLS + c1)*N_COLS + c2), sid, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], const short& mu, const size_t& sid) {
#pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
#pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  set(mu, c1, c2, sid, G[c1][c2]);
	}    
      }
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c1, const short& c2, const sidStride& ss) const {
      return generic<T,Float>::get(((mu*N_COLS + c1)*N_COLS + c2), ss);
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c1, const short& c2, const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      return get(mu, c1, c2, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& mu, const short& c1, const short& c2, const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_DIMS*N_COLS*N_COLS, dirs ...);
      return get(mu,c1,c2,ss);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const short& mu, const sidStride& ss) const {
      #pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
        #pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  G[c1][c2] = get(mu, c1, c2, ss);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const short& mu, const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      get(G, mu, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const short& mu, const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_DIMS*N_COLS*N_COLS, dirs ...);
      get(G, mu, ss);
    }
  };

  template<typename Float>
  using gaugeTex = genericGauge<texture<Float>, Float>;

  template<typename Float>
  using gauge2 = genericGauge<pFloat2<Float>, Float>;

  template<typename T, typename Float>
  struct genericSu3 : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& c1, const short& c2, const size_t& sid, const Float2<Float>& v) {
      generic<T,Float>::set((c1*N_COLS + c2), sid, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], const size_t& sid) {
#pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
#pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  set(c1, c2, sid, G[c1][c2]);
	}    
      }
    }

    inline __device__ Float2<Float> get(const short& c1, const short& c2, const sidStride& ss) const {
      return generic<T,Float>::get((c1*N_COLS + c2), ss);
    }
    inline __device__ Float2<Float> get(const short& c1, const short& c2, const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      return get(c1,c2,ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& c1, const short& c2, const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_COLS*N_COLS, dirs ...);
      return get(c1,c2,ss);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const sidStride& ss) const {
      #pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
        #pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  G[c1][c2] = get(c1, c2, ss);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      get(G, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_COLS*N_COLS, dirs ...);
      get(G, ss);
    }
  };

  template<typename Float>
  using su3Tex = genericSu3<texture<Float>, Float>;

  template<typename Float>
  using su3_2 = genericSu3<pFloat2<Float>, Float>;
  
  
  template<typename T,typename Float>
  struct genericVector : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& mu, const short& c, const size_t& sid, const Float2<Float>& v) {
      generic<T,Float>::set((mu*N_COLS + c),sid,v);
    }
    inline __device__ void set(Float2<Float> S[N_SPINS][N_COLS], const size_t& sid) {
      #pragma unroll
      for(short mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(short c=0; c<N_COLS; c++) {
	  set(mu,c,sid,S[mu][c]);
	}    
      }
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c, const sidStride& ss) const {
      return generic<T,Float>::get((mu*N_COLS + c),ss);
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c, const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      return get(mu,c,ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& mu, const short& c, const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_SPINS*N_COLS, dirs ...);
      return get(mu,c,ss);
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], const sidStride& ss) const {
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(int c=0; c<N_COLS; c++) {
	  S[mu][c] = get(mu,c,ss);
	}    
      }
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      get(S,ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_SPINS*N_COLS, dirs ...);
      get(S,ss);
    }
  };

  template<typename Float>
  using vectorTex = genericVector< texture<Float>, Float>;

  template<typename Float>
  using vector2 = genericVector< pFloat2<Float>, Float>;

  template<typename T, typename Float>
    struct genericProp : generic<T,Float>  {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& mu, const short& nu, const short& c1, const short& c2, const size_t& sid, const Float2<Float>& v) {
      generic<T,Float>::set((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2),sid,v);
    }
    inline __device__ void set(Float2<Float> P[4][4][3][3], const size_t& sid) {
      #pragma unroll
      for(short mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(short nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(short c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(short c2 = 0 ; c2 < N_COLS ; c2++)
	      set(mu, nu, c1, c2, sid, P[mu][nu][c1][c2]);
    }

    inline __device__ Float2<Float> get(const short& mu, const short& nu, const short& c1, const short& c2, const sidStride& ss) const {
      return generic<T,Float>::get((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2), ss);
    }
    inline __device__ Float2<Float> get(const short& mu, const short& nu, const short& c1, const short& c2, const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      return get(mu, nu, c1, c2, ss);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& mu, const short& nu, const short& c1, const short& c2, const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_SPINS*N_SPINS*N_COLS*N_COLS, dirs ...);
      return get(mu,nu,c1,c2,ss);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], const sidStride& ss) const {
      #pragma unroll
      for(short mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(short nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(short c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(short c2 = 0 ; c2 < N_COLS ; c2++)
	      P[mu][nu][c1][c2] = get(mu, nu, c1, c2, ss);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], const size_t& sid) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      get( P, ss );
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], const size_t& sid, const dir_t&... dirs) const {
      sidStride ss(sid,this->is4D,this->returnZero);
      ss.shift<src>(N_SPINS*N_SPINS*N_COLS*N_COLS, dirs ...);
      get(P,ss);
    }
  };

  template<typename Float>
  using propTex = genericProp< texture<Float>, Float>;

  template<typename Float>
  using prop2 = genericProp< pFloat2<Float>, Float>;

}
#endif
