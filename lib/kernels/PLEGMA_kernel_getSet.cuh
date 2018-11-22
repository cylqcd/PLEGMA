#ifndef PLEGMA_KERNEL_TEXTURE_CUH
#define PLEGMA_KERNEL_TEXTURE_CUH

#define GET_ID(sid) {(sid) % c_localL[0],				\
		     ((sid)/c_localL[0]) % c_localL[1],			\
		     ((sid)/c_localL[0]/c_localL[1]) % c_localL[2],	\
		     ((sid)/c_localL[0]/c_localL[1]/c_localL[2]) % c_localL[3] }

#define LEXIC_ID(id) LEXIC(id[3],id[2],id[1],id[0],c_localL)
#define LEXIC_3D(i,id)(i==0 ? LEXIC_TZY(id[3],id[2],id[1],c_localL) : \
                      (i==1 ? LEXIC_TZX(id[3],id[2],id[0],c_localL) : \
		      (i==2 ? LEXIC_TYX(id[3],id[1],id[0],c_localL) : \
		              LEXIC_ZYX(id[2],id[1],id[0],c_localL))))
#define LEXIC_2D(i,j,id)( i==0 ? ( j==1 ? LEXIC_TZ(id[3],id[2],c_localL) : \
				   ( j==2 ? LEXIC_TY(id[3],id[1],c_localL) : \
				     ( j==3 ? LEXIC_ZY(id[2],id[1],c_localL) ) ) ) : \
			  ( i==1 ? ( j==0 ? LEXIC_TZ(id[3],id[2],c_localL) : \
				     ( j==2 ? LEXIC_TX(id[3],id[0],c_localL) : \
				       ( j==3 ? LEXIC_ZX(id[2],id[0],c_localL) ) ) ) : \
			    ( i==2 ? ( j==0 ? LEXIC_TY(id[3],id[1],c_localL) : \
				       ( j==1 ? LEXIC_TX(id[3],id[0],c_localL) : \
					 ( j==3 ? LEXIC_YX(id[1],id[0],c_localL) ) ) ) : \
			      ( ( j==0 ? LEXIC_ZY(id[2],id[1],c_localL) : \
				  ( j==1 ? LEXIC_ZX(id[2],id[0],c_localL) : \
				    ( j==2 ? LEXIC_YX(id[1],id[0],c_localL) ) ) ) ) ) ) )  
#define LEXIC_PLUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]+1)%c_localL[0],c_localL) : \
                        (i==1 ? LEXIC(id[3],id[2],(id[1]+1)%c_localL[1],id[0],c_localL) : \
                        (i==2 ? LEXIC(id[3],(id[2]+1)%c_localL[2],id[1],id[0],c_localL) : \
			        LEXIC((id[3]+1)%c_localL[3],id[2],id[1],id[0],c_localL))))
#define LEXIC_MINUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],(id[0]-1+c_localL[0])%c_localL[0],c_localL) : \
                         (i==1 ? LEXIC(id[3],id[2],(id[1]-1+c_localL[1])%c_localL[1],id[0],c_localL) : \
                         (i==2 ? LEXIC(id[3],(id[2]-1+c_localL[2])%c_localL[2],id[1],id[0],c_localL) : \
			         LEXIC((id[3]-1+c_localL[3])%c_localL[3],id[2],id[1],id[0],c_localL))))

namespace plegma {
  template<typename Float>
  struct texture {
    cudaTextureObject_t tex;
    inline __device__ Float2<Float> fetch(int i);
    inline __device__ Float2<Float> get(int i, int sid, int stride) {
      return texture<Float>::fetch(i*stride + sid);
    }
    inline __device__ void set(int i, int sid, int stride, Float2<Float> v){;}
  };
  
  template<> inline __device__ Float2<float> texture<float>::fetch(int i) {
    return (Float2<float>) tex1Dfetch<float2>(tex,i);  
  }
  
  template<> inline __device__ Float2<double> texture<double>::fetch(int i) {
    int4 v = tex1Dfetch<int4>(tex,i);
    return (Float2<double>) make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
  
  template<typename Float>
  struct pFloat2 {
    Float2<Float>* p;
    inline __device__ pFloat2(Float* pointer) {
      p = (Float2<Float> *) pointer;
    }
    inline __device__ Float2<Float> get(int i, int sid, int stride) {
      return p[i*stride + sid];
    }
    inline __device__ void set(int i, int sid, int stride, Float2<Float> v){
      p[i*stride + sid] = v;
    }
  };
    
  template<typename T, typename Float>
  struct generic : T {
    using T::T;
    inline __device__ Float2<Float> get(int i, int sid, int stride) {
      return T::get(i,sid,stride);
    }
    inline __device__ void set(int i, int sid, Float2<Float> v) {
      T::set(i,sid,c_stride,v);
    }
    inline __device__ Float2<Float> get(int i, int sid) {
      return get(i,sid,c_stride);
    }
    inline __device__ Float2<Float> getPlus(int i, int offset, short int dirPlus, int sid) {
      int id[4] = GET_ID(sid);
      int sidPlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ?
	(c_plusGhost[dirPlus]*offset + LEXIC_3D(dirPlus,id)) : LEXIC_PLUS(dirPlus, id);
      int stridePlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ? c_surface[dirPlus] : c_stride;
      return get(i, sidPlus, stridePlus);
    }
    inline __device__ Float2<Float> getMinus(int i, int offset, short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      int sidMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ?
	(c_minusGhost[dirMinus]*offset + LEXIC_3D(dirMinus,id)) : LEXIC_MINUS(dirMinus, id);
      int strideMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ? c_surface[dirMinus] : c_stride;
      return get(i, sidMinus, strideMinus);
    }
    inline __device__ Float2<Float> getPlusMinus(int i, int offset, short int dirPlus, short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      if(dirPlus == dirMinus) {
	return get(i,sid);
      }
      bool plus_ghost = c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1);
      if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%c_localL[dirPlus]; 
      bool minus_ghost = c_dimBreak[dirMinus] == true && id[dirMinus] == 0;
      if(!minus_ghost) id[dirMinus] = (id[dirMinus] + c_localL[dirMinus] - 1)%c_localL[dirMinus];

      if(plus_ghost && minus_ghost){
	int sidPlusMinus = c_cornerGhost[dirPlus][N_DIMS+dirMinus]*offset + LEXIC_2D(dirPlus,dirMinus,id);
	int stridePlusMinus = c_surface2D[dirPlus][dirMinus];
      } else{
	int sidPlusMinus = plus_ghost ? (c_plusGhost[dirPlus]*offset + LEXIC_3D(dirPlus,id)) :
	  ( minus_ghost ? (c_minusGhost[dirMinus]*offset + LEXIC_3D(dirMinus,id)) : LEXIC_ID(id));	
	int stridePlusMinus = plus_ghost ? c_surface[dirPlus] : ( minus_ghost ? c_surface[dirMinus] : c_stride);
      }
      return get(i,sidPlusMinus,stridePlusMinus);
    }
    inline __device__ Float2<Float> getMinusPlus(int i, int offset, short int dirMinus, short int dirPlus, int sid) {
      return getPlusMinus(i,offset,dirPlus,dirMinus,sid);
    }
  };

  template<typename Float>
  struct genericTex : generic<texture<Float>,Float> {using generic<texture<Float>,Float>::generic;};

  template<typename Float>
  struct generic2 : generic<pFloat2<Float>,Float> {using generic<pFloat2<Float>,Float>::generic;};

  template<typename T, typename Float>
  struct genericGauge : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(short int dir, int a, int b, int sid, Float2<Float> v) {
      T::set(((dir*N_COLS + a)*N_COLS + b), sid, c_stride, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], short int dir, int sid) {
      #pragma unroll
      for(int a=0; a<N_COLS; a++) {
        #pragma unroll
	for(int b=0; b<N_COLS; b++) {
	  set(dir, a, b, sid, G[a][b]);
	}    
      }
    }
    inline __device__ Float2<Float> get(short int dir, int a, int b, int sid, int stride) {
      return T::get(((dir*N_COLS + a)*N_COLS + b), sid, stride);
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
    inline __device__ void getPlusMinus(Float2<Float> G[N_COLS][N_COLS], short int dirLink, short int dirPlus, short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      if(dirPlus == dirMinus) {
	get(G,dirLink,sid);
      }
      bool plus_ghost = c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1);
      if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%c_localL[dirPlus]; 
      bool minus_ghost = c_dimBreak[dirMinus] == true && id[dirMinus] == 0;
      if(!minus_ghost) id[dirMinus] = (id[dirMinus] + c_localL[dirMinus] - 1)%c_localL[dirMinus];
      int offset = N_SPINS*N_COLS*N_COLS;
      if(plus_ghost && minus_ghost){
	int sidPlusMinus = c_cornerGhost[dirPlus][N_DIMS+dirMinus]*offset + LEXIC_2D(dirPlus,dirMinus,id);
	int stridePlusMinus = c_surface2D[dirPlus][dirMinus];
      }
      else{
	int sidPlusMinus = plus_ghost ? (c_plusGhost[dirPlus]*offset + LEXIC_3D(dirPlus,id)) :
	  ( minus_ghost ? (c_minusGhost[dirMinus]*offset + LEXIC_3D(dirMinus,id)) : LEXIC_ID(id));
	int stridePlusMinus = plus_ghost ? c_surface[dirPlus] : ( minus_ghost ? c_surface[dirMinus] : c_stride);
      }
      return get(G,dirLink,sidPlusMinus,stridePlusMinus);
    }
    inline __device__ void getMinusPlus(Float2<Float> G[N_COLS][N_COLS], short int dirLink, short int dirMinus, short int dirPlus, int sid) {
      getPlusMinus(G,dirLink,dirPlus,dirMinus,sid);
    }
  };

  template<typename Float>
  struct gaugeTex : genericGauge<texture<Float>, Float>{using genericGauge<texture<Float>, Float>::genericGauge;};

  template<typename Float>
  struct gauge2 : genericGauge<pFloat2<Float>,Float >{using genericGauge<pFloat2<Float>,Float >::genericGauge;};


  template<typename T, typename Float>
  struct genericSu3 : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(int a, int b, int sid, Float2<Float> v) {
      T::set((a*N_COLS + b), sid, c_stride, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], int sid) {
      #pragma unroll
      for(int a=0; a<N_COLS; a++) {
        #pragma unroll
	for(int b=0; b<N_COLS; b++) {
	  set(a, b, sid, G[a][b]);
	}    
      }
    }

    inline __device__ Float2<Float> get(int a, int b, int sid, int stride) {
      return T::get((a*N_COLS + b), sid, stride);
    }
    inline __device__ Float2<Float> get(int a, int b, int sid) {
      return get(a,b,sid,c_stride);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], int sid, int stride) {
      #pragma unroll
      for(int a=0; a<N_COLS; a++) {
        #pragma unroll
	for(int b=0; b<N_COLS; b++) {
	  G[a][b] = get(a, b, sid, stride);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], int sid) {
      get(G, sid, c_stride);
    }
    inline __device__ void getPlus(Float2<Float> G[N_COLS][N_COLS], short int dirPlus, int sid) {
      int id[4] = GET_ID(sid);
      int sidPlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ?
	(c_plusGhost[dirPlus]*N_DIMS*N_COLS*N_COLS + LEXIC_3D(dirPlus,id)) : LEXIC_PLUS(dirPlus, id);
      int stridePlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ? c_surface[dirPlus] : c_stride;
      get(G, sidPlus, stridePlus);
    }
    inline __device__ void getMinus(Float2<Float> G[N_COLS][N_COLS], short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      int sidMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ?
	(c_minusGhost[dirMinus]*N_DIMS*N_COLS*N_COLS + LEXIC_3D(dirMinus,id)) : LEXIC_MINUS(dirMinus, id);
      int strideMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ? c_surface[dirMinus] : c_stride;
      get(G, sidMinus, strideMinus);
    }
  };

  template<typename Float>
  struct su3Tex : genericSu3<texture<Float>, Float>{using genericSu3<texture<Float>, Float>::genericSu3;};

  template<typename Float>
  struct su3_2 : genericSu3<pFloat2<Float>,Float >{using genericSu3<pFloat2<Float>,Float >::genericSu3;};

  
  template<typename T,typename Float>
  struct genericVector : generic<T,Float> {
    using generic<T,Float>::generic;
    inline void set(int mu, int c, int sid, Float2<Float> v) {
      return T::set((mu*N_COLS + c),sid,c_stride,v);
    }
    inline __device__ void set(Float2<Float> S[N_SPINS][N_COLS], int sid) {
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(int c=0; c<N_COLS; c++) {
	  set(mu,c,sid,S[mu][c]);
	}    
      }
    }

    inline __device__ Float2<Float> get(int mu, int c, int sid, int stride) {
      return T::get((mu*N_COLS + c),sid,stride);
    }
    inline __device__ Float2<Float> get(int mu, int c, int sid) {
      return get(mu,c,sid,c_stride);
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], int sid, int stride) {
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(int c=0; c<N_COLS; c++) {
	  S[mu][c] = get(mu,c,sid,stride);
	}    
      }
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], int sid) {
      get(S,sid,c_stride);
    }
    inline __device__ void getPlus(Float2<Float> S[N_SPINS][N_COLS], short int dirPlus, int sid) {
      int id[4] = GET_ID(sid);
      int sidPlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ?
	(c_plusGhost[dirPlus]*N_SPINS*N_COLS + LEXIC_3D(dirPlus,id)) : LEXIC_PLUS(dirPlus, id);
      int stridePlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ? c_surface[dirPlus] : c_stride;
      get(S,sidPlus,stridePlus);
    }
    inline __device__ void getMinus(Float2<Float> S[N_SPINS][N_COLS], short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      int sidMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ?
	(c_minusGhost[dirMinus]*N_SPINS*N_COLS + LEXIC_3D(dirMinus,id)) : LEXIC_MINUS(dirMinus, id);
      int strideMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ? c_surface[dirMinus] : c_stride;
      get(S,sidMinus,strideMinus);
    }
    inline __device__ void getPlusMinus(Float2<Float> S[N_SPINS][N_COLS], short int dirPlus, short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      if(dirPlus == dirMinus) {
	get(S,sid);
      }
      bool plus_ghost = c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1);
      if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%c_localL[dirPlus]; 
      bool minus_ghost = c_dimBreak[dirMinus] == true && id[dirMinus] == 0;
      if(!minus_ghost) id[dirMinus] = (id[dirMinus] + c_localL[dirMinus] - 1)%c_localL[dirMinus];
      
      int sidPlusMinus = plus_ghost ? (c_plusGhost[dirPlus]*N_SPINS*N_COLS + LEXIC_3D(dirPlus,id)) :
	( minus_ghost ? (c_minusGhost[dirMinus]*N_SPINS*N_COLS + LEXIC_3D(dirMinus,id)) : LEXIC_ID(id));
      
      int stridePlusMinus = plus_ghost ? c_surface[dirPlus] : ( minus_ghost ? c_surface[dirMinus] : c_stride);
      return get(S,sidPlusMinus,stridePlusMinus);
    }
    inline __device__ void getMinusPlus(Float2<Float> S[N_SPINS][N_COLS], short int dirMinus, short int dirPlus, int sid) {
      getPlusMinus(S,dirPlus,dirMinus,sid);
    }
  };

  template<typename Float>
  struct vectorTex : genericVector< texture<Float>, Float > {using genericVector< texture<Float>, Float >::genericVector;};

  template<typename Float>
  struct vector2 : genericVector< pFloat2<Float>, Float > {using genericVector< pFloat2<Float>, Float >::genericVector;};

  template<typename T, typename Float>
    struct genericProp : generic<T,Float>  {
    using generic<T,Float>::generic;
    inline __device__ void set(int mu, int nu, int a, int b, int sid, Float2<Float> v) {
      T::set((((mu*N_SPINS + nu)*N_COLS + a)*N_COLS + b), sid,c_stride,v);
    }
    inline __device__ void set(Float2<Float> P[4][4][3][3], int sid) {
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(int c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(int c2 = 0 ; c2 < N_COLS ; c2++)
	      set(mu, nu, c1, c2, sid, P[mu][nu][c1][c2]);
    }

    inline __device__ Float2<Float> get(int mu, int nu, int a, int b, int sid, int stride) {
      return T::get((((mu*N_SPINS + nu)*N_COLS + a)*N_COLS + b), sid,stride);
    }
    inline __device__ Float2<Float> get(int mu, int nu, int c1, int c2, int sid) {
      return get(mu, nu, c1, c2, sid, c_stride);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], int sid, int stride) {
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(int c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(int c2 = 0 ; c2 < N_COLS ; c2++)
	      P[mu][nu][c1][c2] = get(mu, nu, c1, c2, sid, stride);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], int sid) {
      get( P, sid, c_stride );
    }
    inline __device__ void getPlus(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], short int dirPlus, int sid) {
      int id[4] = GET_ID(sid);
      int sidPlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ?
	(c_plusGhost[dirPlus]*N_SPINS*N_SPINS*N_COLS*N_COLS + LEXIC_3D(dirPlus,id)) : LEXIC_PLUS(dirPlus, id);
      int stridePlus = (c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1)) ? c_surface[dirPlus] : c_stride;
      get( P, sidPlus, stridePlus );
    }
    inline __device__ void getMinus(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      int sidMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ?
	(c_minusGhost[dirMinus]*N_SPINS*N_SPINS*N_COLS*N_COLS + LEXIC_3D(dirMinus,id)) : LEXIC_MINUS(dirMinus, id);
      int strideMinus = (c_dimBreak[dirMinus] == true && id[dirMinus] == 0) ? c_surface[dirMinus] : c_stride;
      get( P, sidMinus, strideMinus );
    }
    inline __device__ void getPlusMinus(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], short int dirPlus, short int dirMinus, int sid) {
      int id[4] = GET_ID(sid);
      if(dirPlus == dirMinus) {
	get(P,sid);
      }
      bool plus_ghost = c_dimBreak[dirPlus] == true && id[dirPlus] == (c_localL[dirPlus]-1);
      if(!plus_ghost) id[dirPlus] = (id[dirPlus] + 1)%c_localL[dirPlus]; 
      bool minus_ghost = c_dimBreak[dirMinus] == true && id[dirMinus] == 0;
      if(!minus_ghost) id[dirMinus] = (id[dirMinus] + c_localL[dirMinus] - 1)%c_localL[dirMinus];
      
      int sidPlusMinus = plus_ghost ? (c_plusGhost[dirPlus]*N_SPINS*N_SPINS*N_COLS*N_COLS + LEXIC_3D(dirPlus,id)) :
	( minus_ghost ? (c_minusGhost[dirMinus]*N_SPINS*N_SPINS*N_COLS*N_COLS + LEXIC_3D(dirMinus,id)) : LEXIC_ID(id));
      
      int stridePlusMinus = plus_ghost ? c_surface[dirPlus] : ( minus_ghost ? c_surface[dirMinus] : c_stride);
      return get(P,sidPlusMinus,stridePlusMinus);
    }
    inline __device__ void getMinusPlus(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], short int dirMinus, short int dirPlus, int sid) {
      getPlusMinus(P,dirPlus,dirMinus,sid);
    }
  };

  template<typename Float>
  struct propTex : genericProp< texture<Float>, Float > {using genericProp< texture<Float>, Float >::genericProp;};

  template<typename Float>
  struct prop2 : genericProp< pFloat2<Float>, Float > {using genericProp< pFloat2<Float>, Float >::genericProp;};

}
#endif
