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
#define LEXIC_NOX(j,id)( j==1 ? LEXIC_TZ(id[3],id[2],DGC_localL) :	\
		       ( j==2 ? LEXIC_TY(id[3],id[1],DGC_localL) :	\
			   LEXIC_ZY(id[2],id[1],DGC_localL) ) )
#define	LEXIC_NOY(j,id)( j==0 ? LEXIC_TZ(id[3],id[2],DGC_localL) :	\
		       ( j==2 ? LEXIC_TX(id[3],id[0],DGC_localL) :	\
			   LEXIC_ZX(id[2],id[0],DGC_localL) ) )
#define LEXIC_NOZ(j,id)( j==0 ? LEXIC_TY(id[3],id[1],DGC_localL) :	\
		       ( j==1 ? LEXIC_TX(id[3],id[0],DGC_localL) :	\
			   LEXIC_YX(id[1],id[0],DGC_localL) ) )
#define LEXIC_NOT(j,id)( j==0 ? LEXIC_ZY(id[2],id[1],DGC_localL) :      \
		       ( j==1 ? LEXIC_ZX(id[2],id[0],DGC_localL) : 	\
			   LEXIC_YX(id[1],id[0],DGC_localL)  ) )

#define LEXIC_ID_3D4D(id,is4D) (is4D ? LEXIC_ID(id) : LEXIC_ZYX(id[2],id[1],id[0],DGC_localL))

// assuming i!=j
#define LEXIC_2D(i,j,id)( i==0 ? LEXIC_NOX(j,id) :			\
			( i==1 ? LEXIC_NOY(j,id) :			\
			( i==2 ? LEXIC_NOZ(j,id) : LEXIC_NOT(j,id) )))
// assuming i!=j!=k
#define LEXIC_1D(i,j,k,id)( i!=0 && j!=0 && k!=0 ? id[0] :		\
			  ( i!=1 && j!=1 && k!=1 ? id[1] :		\
			  ( i!=2 && j!=2 && k!=2 ? id[2] : id[3] )))

#define ID_PLUS(i,id) ((id[i]+1)%DGC_localL[i])
#define ID_MINUS(i,id) ((id[i]+DGC_localL[i]-1)%DGC_localL[i])

#define LEXIC_PLUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],ID_PLUS(i,id),DGC_localL) : \
                        (i==1 ? LEXIC(id[3],id[2],ID_PLUS(i,id),id[0],DGC_localL) : \
                        (i==2 ? LEXIC(id[3],ID_PLUS(i,id),id[1],id[0],DGC_localL) : \
			        LEXIC(ID_PLUS(i,id),id[2],id[1],id[0],DGC_localL))))
#define LEXIC_MINUS(i,id)(i==0 ? LEXIC(id[3],id[2],id[1],ID_MINUS(i,id),DGC_localL) : \
                         (i==1 ? LEXIC(id[3],id[2],ID_MINUS(i,id),id[0],DGC_localL) : \
                         (i==2 ? LEXIC(id[3],ID_MINUS(i,id),id[1],id[0],DGC_localL) : \
			         LEXIC(ID_MINUS(i,id),id[2],id[1],id[0],DGC_localL))))

#define LEXIC_3D_PLUS(i,id)(i==0 ? LEXIC_ZYX(id[2],id[1],ID_PLUS(i,id),DGC_localL) : \
			   (i==1 ? LEXIC_ZYX(id[2],ID_PLUS(i,id),id[0],DGC_localL) : \
			           LEXIC_ZYX(ID_PLUS(i,id),id[1],id[0],DGC_localL)))
#define LEXIC_3D_MINUS(i,id)(i==0 ? LEXIC_ZYX(id[2],id[1],ID_MINUS(i,id),DGC_localL) : \
			    (i==1 ? LEXIC_ZYX(id[2],ID_MINUS(i,id),id[0],DGC_localL) : \
			            LEXIC_ZYX(ID_MINUS(i,id),id[1],id[0],DGC_localL)))

#define LEXIC_3D4D_PLUS(i,id,is4D) (is4D ? LEXIC_PLUS(i,id) : LEXIC_3D_PLUS(i,id))
#define LEXIC_3D4D_MINUS(i,id,is4D) (is4D ? LEXIC_MINUS(i,id) : LEXIC_3D_MINUS(i,id))

#define IS_MINUS_GHOST(i,id) (DGC_dimBreak[i] == true && id[i] == 0)
#define IS_PLUS_GHOST(i,id) (DGC_dimBreak[i] == true && id[i] == (DGC_localL[i]-1))


namespace plegma {
  enum get_from { Me,
		  Plus, Minus,
		  PlusPlus, MinusMinus, PlusMinus, MinusPlus,
		  PlusNoGhost, MinusNoGhost, PlusOnlyGhost, MinusOnlyGhost,
		  PlusPlusPlus, MinusMinusPlus, PlusMinusPlus, MinusPlusPlus,
		  PlusPlusMinus, MinusMinusMinus, PlusMinusMinus, MinusPlusMinus,
  };

  struct sidStride {
    const bool is4D;
    const bool changeValue;
    bool returnZero;
    const int site_size;
    size_t sid;
    size_t stride;
    
    inline __host__ __device__ size_t volume() const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_localVolume : DGC_localVolume3D;
      #else
      return is4D ? HGC_localVolume : HGC_localVolume3D;
      #endif
    }

    inline __host__ __device__ size_t sideGhostVolume() const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_sideGhostVolume : DGC_sideGhostVolume3D;
      #else
      return is4D ? HGC_sideGhostVolume : HGC_sideGhostVolume3D;      
      #endif
    }

    inline __host__ __device__ size_t sideGhostL(const short& dir) const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_surface3D[dir] : (DGC_surface3D[dir]/DGC_localL[DIM_T]);
      #else
      return is4D ? HGC_surface3D[dir] : (HGC_surface3D[dir]/HGC_localL[DIM_T]);
      #endif
    }

    inline __host__ __device__ size_t sideGhostShift(const short& dir, const ORIENTATION& sign) const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_sideGhost[dir][sign] : (DGC_sideGhost[dir][sign]/DGC_localL[DIM_T]);
      #else
      return is4D ? HGC_sideGhost[dir][sign] : (HGC_sideGhost[dir][sign]/HGC_localL[DIM_T]);
      #endif
    }

    inline __host__ __device__ size_t cornerGhostVolume() const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_cornerGhostVolume : DGC_cornerGhostVolume3D;
      #else
      return is4D ? HGC_cornerGhostVolume : HGC_cornerGhostVolume3D;      
      #endif
    }

    inline __host__ __device__ size_t cornerGhostL(const short& dir1, const short& dir2) const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_surface2D[OFF2(dir1,dir2)] : (DGC_surface2D[OFF2(dir1,dir2)]/DGC_localL[DIM_T]);
      #else
      return is4D ? HGC_surface2D[OFF2(dir1,dir2)] : (HGC_surface2D[OFF2(dir1,dir2)]/HGC_localL[DIM_T]);
      #endif
    }

    inline __host__ __device__ size_t vertexGhostL(const short& dir1, const short& dir2, const short& dir3) const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_surface1D[OFF3(dir1,dir2,dir3)] : (DGC_surface1D[OFF3(dir1,dir2,dir3)]/DGC_localL[DIM_T]);
      #else
      return is4D ? HGC_surface1D[OFF3(dir1,dir2,dir3)] : (HGC_surface1D[OFF3(dir1,dir2,dir3)]/HGC_localL[DIM_T]);
      #endif
    }

    inline __host__ __device__ size_t cornerGhostShift(const short& dir1, const short& dir2,
					  const ORIENTATION& sign1, const ORIENTATION& sign2) const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_cornerGhost[OFF2(dir1,dir2)][sign1][sign2] : (DGC_cornerGhost[OFF2(dir1,dir2)][sign1][sign2]/DGC_localL[DIM_T]);
      #else
      return is4D ? HGC_cornerGhost[OFF2(dir1,dir2)][sign1][sign2] : (HGC_cornerGhost[OFF2(dir1,dir2)][sign1][sign2]/HGC_localL[DIM_T]);
      #endif
    }

    inline __host__ __device__ size_t vertexGhostShift(const short& dir1, const short& dir2, const short& dir3,
					  const ORIENTATION& sign1, const ORIENTATION& sign2, const ORIENTATION& sign3) const {
      #ifdef __CUDA_ARCH__
      return is4D ? DGC_vertexGhost[OFF3(dir1,dir2,dir3)][sign1][sign2][sign3] : (DGC_vertexGhost[OFF3(dir1,dir2,dir3)][sign1][sign2][sign3]/DGC_localL[DIM_T]);
      #else
      return is4D ? HGC_vertexGhost[OFF3(dir1,dir2,dir3)][sign1][sign2][sign3] : (HGC_vertexGhost[OFF3(dir1,dir2,dir3)][sign1][sign2][sign3]/HGC_localL[DIM_T]);
      #endif
    }

    inline __host__ __device__ sidStride(const int& site_size, const bool& is4D, const bool& changeValue) :
      is4D(is4D), changeValue(changeValue), returnZero(false), site_size(site_size), sid(0), stride(volume()) { }

    inline __host__ __device__ void setSid(const size_t& sid) {
      this->sid = sid;
      this->stride = volume();
      this->returnZero = false;
    }
    
    template<get_from src>
    inline __device__ void shift(const short& dir);
    
    template<get_from src>
    inline __device__ void shift(const short& dir1, const short& dir2);
    
    template<get_from src>
    inline __device__ void shift(const short& dir1, const short& dir2, const short& dir3);
    
    inline __host__ __device__ void accessSideGhost(const size_t& sid3D, const short& dir, const ORIENTATION& sign) {
      this->stride = sideGhostL(dir);
      this->sid = (volume()+sideGhostShift(dir, sign))*site_size + sid3D;
    }

    inline __host__ __device__ void accessCornerGhost(const size_t& sid2D, const short& dir1, const short& dir2, const ORIENTATION& sign1, const ORIENTATION& sign2) {
      this->stride = cornerGhostL(dir1, dir2);
      this->sid = (volume()+sideGhostVolume()+cornerGhostShift(dir1,dir2,sign1,sign2))*site_size + sid2D;
    }

    inline __host__ __device__ void accessVertexGhost(const size_t& sid1D, const short& dir1, const short& dir2, const short& dir3, const ORIENTATION& sign1, const ORIENTATION& sign2, const ORIENTATION& sign3) {
      this->stride = vertexGhostL(dir1, dir2, dir3);
      this->sid = (volume()+sideGhostVolume()+cornerGhostVolume()+vertexGhostShift(dir1,dir2,dir3,sign1,sign2,sign3))*site_size + sid1D;
    }
};


  template<>
  inline __device__ void sidStride::shift<Plus>(const short& dirPlus) {
    size_t id[4] = GET_ID(sid);
    if(IS_PLUS_GHOST(dirPlus, id)) {
      this->accessSideGhost(LEXIC_3D(dirPlus,id), dirPlus, DIR_PLUS);
    } else {
      this->sid = LEXIC_3D4D_PLUS(dirPlus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<Minus>(const short& dirMinus) {
    size_t id[4] = GET_ID(sid);
    if(IS_MINUS_GHOST(dirMinus, id)) {
      this->accessSideGhost(LEXIC_3D(dirMinus,id), dirMinus, DIR_MINUS);
    } else {
      this->sid = LEXIC_3D4D_MINUS(dirMinus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusOnlyGhost>(const short& dirPlus) {
    size_t id[4] = GET_ID(sid);
    if(IS_PLUS_GHOST(dirPlus, id)) {
      this->accessSideGhost(LEXIC_3D(dirPlus,id), dirPlus, DIR_PLUS);
    } else {
      this->returnZero = true;
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusOnlyGhost>(const short& dirMinus) {
    size_t id[4] = GET_ID(sid);
    if(IS_MINUS_GHOST(dirMinus, id)) {
      this->accessSideGhost(LEXIC_3D(dirMinus,id), dirMinus, DIR_MINUS);
    } else {
      this->returnZero = true;
    }
  }
  
  template<>
  inline __device__ void sidStride::shift<PlusNoGhost>(const short& dirPlus) {
    size_t id[4] = GET_ID(sid);
    if(IS_PLUS_GHOST(dirPlus, id)) {
      this->returnZero = true;
    } else {
      this->sid = LEXIC_3D4D_PLUS(dirPlus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusNoGhost>(const short& dirMinus) {
    size_t id[4] = GET_ID(sid);
    if(IS_MINUS_GHOST(dirMinus, id)) {
      this->returnZero = true;
    } else {
      this->sid = LEXIC_3D4D_MINUS(dirMinus, id, is4D);
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusPlus>(const short& dirPlus1, const short& dirPlus2) {
    if(dirPlus1 == dirPlus2 && DGC_dimBreak[dirPlus1]) {
      printf(" !!! ERROR: in PlusPlus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus1_ghost = IS_PLUS_GHOST(dirPlus1, id);
      if(!plus1_ghost) id[dirPlus1] = ID_PLUS(dirPlus1, id);
      bool plus2_ghost = IS_PLUS_GHOST(dirPlus2, id);
      if(!plus2_ghost) id[dirPlus2] = ID_PLUS(dirPlus2, id);

      if(plus1_ghost && plus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus1,dirPlus2,id), dirPlus1, dirPlus2, DIR_PLUS, DIR_PLUS);
      } else if(plus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus1,id), dirPlus1, DIR_PLUS);
      } else if(plus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus2,id), dirPlus2, DIR_PLUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusMinus>(const short& dirMinus1, const short& dirMinus2) {
    if(dirMinus1 == dirMinus2 && DGC_dimBreak[dirMinus1]) {
      printf(" !!! ERROR: in MinusMinus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool minus1_ghost = IS_MINUS_GHOST(dirMinus1, id);
      if(!minus1_ghost) id[dirMinus1] = ID_MINUS(dirMinus1, id);
      bool minus2_ghost = IS_MINUS_GHOST(dirMinus2, id);
      if(!minus2_ghost) id[dirMinus2] = ID_MINUS(dirMinus2, id);

      if(minus1_ghost && minus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus1,dirMinus2,id), dirMinus1, dirMinus2, DIR_MINUS, DIR_MINUS);
      } else if(minus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus1,id), dirMinus1, DIR_MINUS);
      } else if(minus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus2,id), dirMinus2, DIR_MINUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusMinus>(const short& dirPlus, const short& dirMinus) {
    if(dirPlus == dirMinus) {
      return;
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus_ghost = IS_PLUS_GHOST(dirPlus, id);
      if(!plus_ghost) id[dirPlus] = ID_PLUS(dirPlus, id);
      bool minus_ghost = IS_MINUS_GHOST(dirMinus, id);
      if(!minus_ghost) id[dirMinus] = ID_MINUS(dirMinus, id);

      if(plus_ghost && minus_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus,dirMinus,id), dirPlus, dirMinus, DIR_PLUS, DIR_MINUS);
      } else if(plus_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus,id), dirPlus, DIR_PLUS);
      } else if(minus_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus,id), dirMinus, DIR_MINUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusPlus>(const short& dirMinus, const short& dirPlus) {
    this->shift<PlusMinus>(dirPlus, dirMinus);
  }

  template<>
  inline __device__ void sidStride::shift<PlusPlusPlus>(const short& dirPlus1, const short& dirPlus2, const short& dirPlus3) {
    if(dirPlus1 == dirPlus2 || dirPlus1 == dirPlus3 || dirPlus3 == dirPlus2) {
      printf(" !!! ERROR: in PlusPlusPlus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus1_ghost = IS_PLUS_GHOST(dirPlus1, id);
      if(!plus1_ghost) id[dirPlus1] = ID_PLUS(dirPlus1, id);
      bool plus2_ghost = IS_PLUS_GHOST(dirPlus2, id);
      if(!plus2_ghost) id[dirPlus2] = ID_PLUS(dirPlus2, id);
      bool plus3_ghost = IS_PLUS_GHOST(dirPlus3, id);
      if(!plus3_ghost) id[dirPlus3] = ID_PLUS(dirPlus3, id);

      if(plus1_ghost && plus2_ghost && plus3_ghost){
	this->accessVertexGhost(LEXIC_1D(dirPlus1,dirPlus2,dirPlus3,id), dirPlus1, dirPlus2, dirPlus3, DIR_PLUS, DIR_PLUS, DIR_PLUS);
      } else if(plus1_ghost && plus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus1,dirPlus2,id), dirPlus1, dirPlus2, DIR_PLUS, DIR_PLUS);
      } else if(plus1_ghost && plus3_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus1,dirPlus3,id), dirPlus1, dirPlus3, DIR_PLUS, DIR_PLUS);
      } else if(plus2_ghost && plus3_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus2,dirPlus3,id), dirPlus2, dirPlus3, DIR_PLUS, DIR_PLUS);
      } else if(plus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus1,id), dirPlus1, DIR_PLUS);
      } else if(plus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus2,id), dirPlus2, DIR_PLUS);
      } else if(plus3_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus3,id), dirPlus3, DIR_PLUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusPlusMinus>(const short& dirPlus1, const short& dirPlus2, const short& dirMinus) {
    if(dirPlus1 == dirPlus2 || dirPlus1 == dirMinus || dirMinus == dirPlus2) {
      printf(" !!! ERROR: in PlusPlusMinus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool plus1_ghost = IS_PLUS_GHOST(dirPlus1, id);
      if(!plus1_ghost) id[dirPlus1] = ID_PLUS(dirPlus1, id);
      bool plus2_ghost = IS_PLUS_GHOST(dirPlus2, id);
      if(!plus2_ghost) id[dirPlus2] = ID_PLUS(dirPlus2, id);
      bool minus_ghost = IS_MINUS_GHOST(dirMinus, id);
      if(!minus_ghost) id[dirMinus] = ID_MINUS(dirMinus, id);

      if(plus1_ghost && plus2_ghost && minus_ghost){
	this->accessVertexGhost(LEXIC_1D(dirPlus1,dirPlus2,dirMinus,id), dirPlus1, dirPlus2, dirMinus, DIR_PLUS, DIR_PLUS, DIR_MINUS);
      } else if(plus1_ghost && plus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus1,dirPlus2,id), dirPlus1, dirPlus2, DIR_PLUS, DIR_PLUS);
      } else if(plus1_ghost && minus_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus1,dirMinus,id), dirPlus1, dirMinus, DIR_PLUS, DIR_MINUS);
      } else if(plus2_ghost && minus_ghost){
	this->accessCornerGhost(LEXIC_2D(dirPlus2,dirMinus,id), dirPlus2, dirMinus, DIR_PLUS, DIR_MINUS);
      } else if(plus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus1,id), dirPlus1, DIR_PLUS);
      } else if(plus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus2,id), dirPlus2, DIR_PLUS);
      } else if(minus_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus,id), dirMinus, DIR_MINUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<PlusMinusPlus>(const short& dirPlus1, const short& dirMinus, const short& dirPlus2) {
    this->shift<PlusPlusMinus>(dirPlus1, dirPlus2, dirMinus);
  }
  template<>
  inline __device__ void sidStride::shift<MinusPlusPlus>(const short& dirMinus, const short& dirPlus1, const short& dirPlus2) {
    this->shift<PlusPlusMinus>(dirPlus1, dirPlus2, dirMinus);
  }
  
  template<>
  inline __device__ void sidStride::shift<MinusMinusMinus>(const short& dirMinus1, const short& dirMinus2, const short& dirMinus3) {
    if(dirMinus1 == dirMinus2 || dirMinus1 == dirMinus3 || dirMinus3 == dirMinus2) {
      printf(" !!! ERROR: in MinusMinus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool minus1_ghost = IS_MINUS_GHOST(dirMinus1, id);
      if(!minus1_ghost) id[dirMinus1] = ID_MINUS(dirMinus1, id);
      bool minus2_ghost = IS_MINUS_GHOST(dirMinus2, id);
      if(!minus2_ghost) id[dirMinus2] = ID_MINUS(dirMinus2, id);
      bool minus3_ghost = IS_MINUS_GHOST(dirMinus3, id);
      if(!minus3_ghost) id[dirMinus3] = ID_MINUS(dirMinus3, id);

      if(minus1_ghost && minus2_ghost && minus3_ghost){
	this->accessVertexGhost(LEXIC_1D(dirMinus1,dirMinus2,dirMinus3,id), dirMinus1, dirMinus2, dirMinus3, DIR_MINUS, DIR_MINUS, DIR_MINUS);
      } else if(minus1_ghost && minus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus1,dirMinus2,id), dirMinus1, dirMinus2, DIR_MINUS, DIR_MINUS);
      } else if(minus1_ghost && minus3_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus1,dirMinus3,id), dirMinus1, dirMinus3, DIR_MINUS, DIR_MINUS);
      } else if(minus2_ghost && minus3_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus2,dirMinus3,id), dirMinus2, dirMinus3, DIR_MINUS, DIR_MINUS);
      } else if(minus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus1,id), dirMinus1, DIR_MINUS);
      } else if(minus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus2,id), dirMinus2, DIR_MINUS);
      } else if(minus3_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus3,id), dirMinus3, DIR_MINUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusMinusPlus>(const short& dirMinus1, const short& dirMinus2, const short& dirPlus) {
    if(dirMinus1 == dirMinus2 || dirMinus1 == dirPlus || dirPlus == dirMinus2) {
      printf(" !!! ERROR: in MinusMinus we cannot access the second neighbour !!!");
    } else {
      size_t id[4] = GET_ID(sid);
      bool minus1_ghost = IS_MINUS_GHOST(dirMinus1, id);
      if(!minus1_ghost) id[dirMinus1] = ID_MINUS(dirMinus1, id);
      bool minus2_ghost = IS_MINUS_GHOST(dirMinus2, id);
      if(!minus2_ghost) id[dirMinus2] = ID_MINUS(dirMinus2, id);
      bool plus_ghost = IS_PLUS_GHOST(dirPlus, id);
      if(!plus_ghost) id[dirPlus] = ID_PLUS(dirPlus, id);

      if(minus1_ghost && minus2_ghost && plus_ghost){
	this->accessVertexGhost(LEXIC_1D(dirMinus1,dirMinus2,dirPlus,id), dirMinus1, dirMinus2, dirPlus, DIR_MINUS, DIR_MINUS, DIR_PLUS);
      } else if(minus1_ghost && minus2_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus1,dirMinus2,id), dirMinus1, dirMinus2, DIR_MINUS, DIR_MINUS);
      } else if(minus1_ghost && plus_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus1,dirPlus,id), dirMinus1, dirPlus, DIR_MINUS, DIR_PLUS);
      } else if(minus2_ghost && plus_ghost){
	this->accessCornerGhost(LEXIC_2D(dirMinus2,dirPlus,id), dirMinus2, dirPlus, DIR_MINUS, DIR_PLUS);
      } else if(minus1_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus1,id), dirMinus1, DIR_MINUS);
      } else if(minus2_ghost) {
	this->accessSideGhost(LEXIC_3D(dirMinus2,id), dirMinus2, DIR_MINUS);
      } else if(plus_ghost) {
	this->accessSideGhost(LEXIC_3D(dirPlus,id), dirPlus, DIR_PLUS);
      } else {
	this->sid = LEXIC_ID_3D4D(id,is4D);
      }
    }
  }
  template<>
  inline __device__ void sidStride::shift<MinusPlusMinus>(const short& dirMinus1, const short& dirPlus, const short& dirMinus2) {
    this->shift<MinusMinusPlus>(dirMinus1, dirMinus2, dirPlus);
  }
  template<>
  inline __device__ void sidStride::shift<PlusMinusMinus>(const short& dirPlus, const short& dirMinus1, const short& dirMinus2) {
    this->shift<MinusMinusPlus>(dirMinus1, dirMinus2, dirPlus);
  }

  template<typename Float>
  struct pFloat2 : sidStride {
    Float2<Float>* p;

    __host__ __device__ pFloat2(Float2<Float>* p, int site_size, bool is4D, bool changeValue) :
      sidStride(site_size, is4D, changeValue), p(p) { }
        
    inline __host__ __device__ Float2<Float> get(const int& i) const {
      return this->returnZero ? Float2<Float>(0) : p[i*this->stride + this->sid];
    }
    inline __host__ __device__ void set(const int& i, const Float2<Float>& v) {
      // In case of changeValue we don't change the value. Useful for tuning an in/out vector
      p[i*this->stride + this->sid] = this->changeValue ? v : p[i*this->stride + this->sid];
    }
    inline __host__ __device__ void add(const int& i, const Float2<Float>& v) {
      p[i*this->stride + this->sid] += this->changeValue ? v : 0;
    }
    inline __host__ __device__ Float2<Float>& operator[](const int& i) {
      return p[i*this->stride + this->sid];
    }
  };
  
  template<typename Float>
  struct texture : pFloat2<Float> {
    cudaTextureObject_t tex;

    __host__ __device__ texture(cudaTextureObject_t tex, Float2<Float>* p, int site_size, bool is4D, bool changeValue) :
      pFloat2<Float>(p, site_size, is4D, changeValue), tex(tex) { }

    #ifdef PLEGMA_TEXTURE
    // Fetch is going to be specialized after
    inline __device__ Float2<Float> fetch(const size_t& i) const;
    inline __device__ Float2<Float> get(const int& i) const {
      return this->returnZero ? Float2<Float>(0) : texture<Float>::fetch(i*this->stride + this->sid);
    }
    #endif
  };

  #ifdef PLEGMA_TEXTURE
  // Here we specialize fetch
  template<> inline __device__ Float2<float> texture<float>::fetch(const size_t& i) const {
    return (Float2<float>) tex1Dfetch<float2>(tex,i);  
  }
  template<> inline __device__ Float2<double> texture<double>::fetch(const size_t& i) const {
    int4 v = tex1Dfetch<int4>(tex,i);
    return (Float2<double>) make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
  #endif

  template<typename T, typename Float>
  struct generic : T {
    using T::T;
    inline __device__ void set(const int& i, const size_t& sid, const Float2<Float>& v) {
      sidStride::setSid(sid);
      T::set(i,v);
    }
    inline __device__ Float2<Float> get(const int& i, const size_t& sid) {
      sidStride::setSid(sid);
      return T::get(i);
    }
    inline __device__ void get(Float2<Float> *p) const {
      #pragma unroll
      for(int i=0; i<T::site_size; i++) {
	p[i] = T::get(i);
      }
    }
    inline __device__ void get(Float2<Float> *p, const size_t& sid) {
      sidStride::setSid(sid);
      return get(p);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const int& i, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      return T::get(i);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> *p, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      get(p); 
    }    
  };
  
  template<typename Float>
  using genericTex = generic<texture<Float>, Float>;
  
  template<typename Float>
  using generic2 = generic<pFloat2<Float>, Float>;


  template<typename T, typename Float>
  struct genericGauge : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& mu, const short& c1, const short& c2, const Float2<Float>& v) {
      T::set(((mu*N_COLS + c1)*N_COLS + c2), v);
    }
    inline __device__ void set(const short& mu, const short& c1, const short& c2, const size_t& sid, const Float2<Float>& v) {
      sidStride::setSid(sid);
      set(mu, c1, c2, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], const short& mu, const size_t& sid) {
      sidStride::setSid(sid);
#pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
#pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  set(mu, c1, c2, G[c1][c2]);
	}    
      }
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c1, const short& c2) const {
      return T::get(((mu*N_COLS + c1)*N_COLS + c2));
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c1, const short& c2, const size_t& sid) {
      sidStride::setSid(sid);
      return get(mu, c1, c2);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& mu, const short& c1, const short& c2, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      return get(mu,c1,c2);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const short& mu) const {
      #pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
        #pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  G[c1][c2] = get(mu, c1, c2);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const short& mu, const size_t& sid) {
      sidStride::setSid(sid);
      get(G, mu);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const short& mu, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      get(G, mu);
    }
  };

  template<typename Float>
  using gaugeTex = genericGauge<texture<Float>, Float>;

  template<typename Float>
  using gauge2 = genericGauge<pFloat2<Float>, Float>;

  template<typename T, typename Float>
  struct genericSu3 : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& c1, const short& c2, const Float2<Float>& v) {
      T::set((c1*N_COLS + c2), v);
    }
    inline __device__ void set(const short& c1, const short& c2, const size_t& sid, const Float2<Float>& v) {
      sidStride::setSid(sid);
      set(c1, c2, v);
    }
    inline __device__ void set(Float2<Float> G[N_COLS][N_COLS], const size_t& sid) {
      sidStride::setSid(sid);
      #pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
        #pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  set(c1, c2, G[c1][c2]);
	}    
      }
    }

    inline __device__ Float2<Float> get(const short& c1, const short& c2) const {
      return T::get((c1*N_COLS + c2));
    }
    inline __device__ Float2<Float> get(const short& c1, const short& c2, const size_t& sid) {
      sidStride::setSid(sid);
      return get(c1,c2);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& c1, const short& c2, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      return get(c1,c2);
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS]) const {
      #pragma unroll
      for(short c1=0; c1<N_COLS; c1++) {
        #pragma unroll
	for(short c2=0; c2<N_COLS; c2++) {
	  G[c1][c2] = get(c1, c2);
	}    
      }
    }
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const size_t& sid) {
      sidStride::setSid(sid);
      get(G);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> G[N_COLS][N_COLS], const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      get(G);
    }
  };

  template<typename Float>
  using su3Tex = genericSu3<texture<Float>, Float>;

  template<typename Float>
  using su3_2 = genericSu3<pFloat2<Float>, Float>;
  
  
  template<typename T,typename Float>
  struct genericVector : generic<T,Float> {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& mu, const short& c, const Float2<Float>& v) {
      T::set((mu*N_COLS + c),v);
    }
    inline __device__ void set(const short& mu, const short& c, const size_t& sid, const Float2<Float>& v) {
      sidStride::setSid(sid);
      set(mu,c,v);
    }
    inline __device__ void set(Float2<Float> S[N_SPINS][N_COLS], const size_t& sid) {
      sidStride::setSid(sid);
      #pragma unroll
      for(short mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(short c=0; c<N_COLS; c++) {
	  set(mu,c,S[mu][c]);
	}    
      }
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c) const {
      return T::get(mu*N_COLS + c);
    }
    inline __device__ Float2<Float> get(const short& mu, const short& c, const size_t& sid) {
      sidStride::setSid(sid);
      return get(mu,c);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& mu, const short& c, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      return get(mu,c);
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS]) const {
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++) {
        #pragma unroll
	for(int c=0; c<N_COLS; c++) {
	  S[mu][c] = get(mu,c);
	}    
      }
    }
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], const size_t& sid) {
      sidStride::setSid(sid);
      get(S);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> S[N_SPINS][N_COLS], const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      get(S);
    }
  };

  template<typename Float>
  using vectorTex = genericVector< texture<Float>, Float>;

  template<typename Float>
  using vector2 = genericVector< pFloat2<Float>, Float>;

  template<typename T, typename Float>
    struct genericProp : generic<T,Float>  {
    using generic<T,Float>::generic;
    inline __device__ void set(const short& mu, const short& nu, const short& c1, const short& c2, const Float2<Float>& v) {
      T::set((((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2),v);
    }
    inline __device__ void set(const short& mu, const short& nu, const short& c1, const short& c2, const size_t& sid, const Float2<Float>& v) {
      sidStride::setSid(sid);
      set(mu, nu, c1, c2,v);
    }
    inline __device__ void set(Float2<Float> P[4][4][3][3], const size_t& sid) {
      sidStride::setSid(sid);
      #pragma unroll
      for(short mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(short nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(short c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(short c2 = 0 ; c2 < N_COLS ; c2++)
	      set(mu, nu, c1, c2, P[mu][nu][c1][c2]);
    }

    inline __device__ Float2<Float> get(const short& mu, const short& nu, const short& c1, const short& c2) const {
      return T::get(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2);
    }
    inline __device__ Float2<Float> get(const short& mu, const short& nu, const short& c1, const short& c2, const size_t& sid) {
      sidStride::setSid(sid);
      return get(mu, nu, c1, c2);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ Float2<Float> get(const short& mu, const short& nu, const short& c1, const short& c2, const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      return get(mu,nu,c1,c2);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS]) const {
      #pragma unroll
      for(short mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(short nu = 0 ; nu < N_SPINS ; nu++)
          #pragma unroll
	  for(short c1 = 0 ; c1 < N_COLS ; c1++)
            #pragma unroll
	    for(short c2 = 0 ; c2 < N_COLS ; c2++)
	      P[mu][nu][c1][c2] = get(mu, nu, c1, c2);
    }
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], const size_t& sid) {
      sidStride::setSid(sid);
      get(P);
    }
    template<get_from src, typename ...dir_t>
    inline __device__ void get(Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], const size_t& sid, const dir_t&... dirs) {
      sidStride::setSid(sid);
      sidStride::shift<src>(dirs ...);
      get(P);
    }
  };

  template<typename Float>
  using propTex = genericProp< texture<Float>, Float>;

  template<typename Float>
  using prop2 = genericProp< pFloat2<Float>, Float>;

  template<template<typename> class T, template<typename> class Tfield, typename Float>
  static inline T<Float> toField2(const Tfield<Float>& field) {
    return T<Float>((Float2<Float>*) field.D_elem(), field.Field_length(), field.is4D(), true);
  }

  template<template<typename> class T, template<typename> class Tfield, typename Float>
  static inline std::shared_ptr<T<Float>> toTexture(const Tfield<Float>& field) {
    return std::shared_ptr<T<Float>>(new T<Float>(field.createTexObject(), (Float2<Float>*) field.D_elem(), field.Field_length(), field.is4D(), true), [&](T<Float>* ptr){field.destroyTexObject(ptr->tex); delete ptr;});
  }
    

}
#endif
