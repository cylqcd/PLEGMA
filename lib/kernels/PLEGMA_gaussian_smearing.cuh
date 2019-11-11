#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
template<typename FloatOut, typename FloatIn, typename FloatGauge, bool noGhost=false, bool onlyGhost=false>
__global__ void gaussian_smearing_kernel(Float2<FloatOut>* out,
					 vectorTex<FloatIn> vecInTex,
					 gaugeTex<FloatGauge> gaugeTex,
					 FloatOut alpha, int it=-1){
  static_assert(not (noGhost and onlyGhost), "Not possible to have both: noGhost and onlyGhost");
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  if(it >= 0) {
    Float2<FloatIn> S[N_SPINS][N_COLS];
    it = it - DGC_procPosition[3] * DGC_localL[3];
    // Only the correct timeslice has to work, the others just copy
    if(it != sid/DGC_localVolume3D) {
      if(not onlyGhost) {
	vecInTex.get(S,sid);
        #pragma unroll
	for(int mu=0; mu<N_SPINS; mu++) 
          #pragma unroll
	  for(int c=0; c<N_COLS; c++)
	    out[(mu*N_COLS + c)*DGC_localVolume + sid] = S[mu][c];
      }
      return;
    }
  }
  
  if(onlyGhost) {
    // Checking if we are on the border
    bool exit = true;
    size_t id[4] = GET_ID(sid);
    for(int dir = 0; dir < N_DIMS-1; dir++)
      exit &= (not DGC_dimBreak[dir]) || (id[dir] != (DGC_localL[dir]-1) && id[dir] != 0);
    if(exit) return;
  }
  
  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn> S[N_SPINS][N_COLS];
  Float2<FloatOut> tmp[N_SPINS][N_COLS];

  #pragma unroll
  for(int mu=0; mu<N_SPINS; mu++) 
    #pragma unroll
    for(int c=0; c<N_COLS; c++)
      tmp[mu][c] = 0.;

  // we don't smear the time -> N_DIMS-1
  #pragma unroll
  for(int dir = 0; dir < N_DIMS-1; dir++) {
    vecInTex.get<onlyGhost ? PlusOnlyGhost : (noGhost ? PlusNoGhost : Plus)>(S, sid, dir);
    if(isNotZeroV(S)) {
      gaugeTex.get(G, dir, sid);
      mul_G_V<FloatOut,FloatGauge,FloatIn,ACC_PLUS>(tmp,G,S);
    }
    vecInTex.get<onlyGhost ? MinusOnlyGhost : (noGhost ? MinusNoGhost : Minus)>(S, sid, dir);
    if(isNotZeroV(S)) {
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag_V<FloatOut,FloatGauge,FloatIn,ACC_PLUS>(tmp,G,S);
    }
  }

  double normalize;
  normalize = 1./(1. + 6. * alpha);

  if(not onlyGhost) {
    vecInTex.get(S,sid);
    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++) 
      #pragma unroll
      for(int c=0; c<N_COLS; c++)
	out[(mu*N_COLS + c)*DGC_localVolume + sid] = normalize * (S[mu][c] + alpha * tmp[mu][c]);
  } else if(isNotZeroV(tmp)) {
    normalize *= alpha;
    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++) 
      #pragma unroll
      for(int c=0; c<N_COLS; c++)
	out[(mu*N_COLS + c)*DGC_localVolume + sid] += normalize * tmp[mu][c];
  }
}

template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing(FloatOut* out, vectorTex<FloatIn> vecInTex, 
			      gaugeTex<FloatGauge> gaugeTex, FloatOut alpha, int it=-1){
  ProfileStruct ps(HGC_localVolume);
  ps.tune_globally = true;
  tuneAndRun(ps, "gaussian_smearing_kernel", gaussian_smearing_kernel<FloatOut,FloatIn,FloatGauge,false,false>, (Float2<FloatOut> *) out, vecInTex, gaugeTex, alpha, it);
}

template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing_no_ghost(FloatOut* out, vectorTex<FloatIn> vecInTex, 
				       gaugeTex<FloatGauge> gaugeTex, FloatOut alpha, int it=-1){
  ProfileStruct ps(HGC_localVolume);
  ps.tune_globally = true;
  tuneAndRun(ps, "gaussian_smearing_no_ghost_kernel", gaussian_smearing_kernel<FloatOut,FloatIn,FloatGauge,true,false>, (Float2<FloatOut> *) out, vecInTex, gaugeTex, alpha, it);
}

template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing_only_ghost(FloatOut* out, vectorTex<FloatIn> vecInTex, 
					 gaugeTex<FloatGauge> gaugeTex, FloatOut alpha, int it=-1){
  ProfileStruct ps(HGC_localVolume);
  ps.tune_globally = true;
  tuneAndRun(ps, "gaussian_smearing_only_ghost_kernel", gaussian_smearing_kernel<FloatOut,FloatIn,FloatGauge,false,true>, (Float2<FloatOut> *) out, vecInTex, gaugeTex, alpha, it);
}
