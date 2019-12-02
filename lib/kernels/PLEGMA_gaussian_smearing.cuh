#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
template<typename FloatOut, typename FloatIn, typename FloatGauge, bool noGhost=false>
__global__ void gaussian_smearing_kernel(vectorTex<FloatOut>out,
					 vectorTex<FloatIn> vecInTex,
					 gaugeTex<FloatGauge> gaugeTex,
					 FloatOut alpha){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= out.volume()) return;

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
    vecInTex.get<noGhost ? PlusNoGhost : Plus>(S, sid, dir);
    if(isNotZeroV(S)) {
      gaugeTex.get(G, dir, sid);
      mul_G_V<FloatOut,FloatGauge,FloatIn,ACC_PLUS>(tmp,G,S);
    }
    vecInTex.get<noGhost ? MinusNoGhost : Minus>(S, sid, dir);
    if(isNotZeroV(S)) {
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag_V<FloatOut,FloatGauge,FloatIn,ACC_PLUS>(tmp,G,S);
    }
  }

  FloatOut normalize = 1/(1 + 6 * alpha);

  out.setSid(sid);
  vecInTex.get(S,sid);
  #pragma unroll
  for(int mu=0; mu<N_SPINS; mu++) 
    #pragma unroll
    for(int c=0; c<N_COLS; c++)
      out[mu*N_COLS + c] = normalize * (S[mu][c] + alpha * tmp[mu][c]);
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
__global__ void gaussian_smearing_only_ghost_kernel(vectorTex<FloatOut>out,
						    vectorTex<FloatIn> vecInTex,
						    gaugeTex<FloatGauge> gaugeTex,
						    FloatOut alpha, short dir){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= 2*out.sideGhostL(dir)) return;

  ORIENTATION sign = sid>=out.sideGhostL(dir) ? DIR_MINUS : DIR_PLUS;
  size_t id[4];
  sid = sid % out.sideGhostL(dir);
  #pragma unroll
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir) {
      id[i] = sign==DIR_PLUS ? (DGC_localL[dir]-1):0;
    } else {
      id[i] = sid % DGC_localL[i];
      sid /= DGC_localL[i];
    }
  }
  sid = LEXIC_ID(id);
  
  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn> S[N_SPINS][N_COLS];
  Float2<FloatOut> tmp[N_SPINS][N_COLS];

  #pragma unroll
  for(int mu=0; mu<N_SPINS; mu++) 
    #pragma unroll
    for(int c=0; c<N_COLS; c++)
      tmp[mu][c] = 0.;

  if(sign==DIR_PLUS) {
    vecInTex.get<PlusOnlyGhost>(S, sid, dir);
    if(isNotZeroV(S)) {
      gaugeTex.get(G, dir, sid);
      mul_G_V<FloatOut,FloatGauge,FloatIn,ACC_PLUS>(tmp,G,S);
    }
  } else {
    vecInTex.get<MinusOnlyGhost>(S, sid, dir);
    if(isNotZeroV(S)) {
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag_V<FloatOut,FloatGauge,FloatIn,ACC_PLUS>(tmp,G,S);
    }
  }

  FloatOut normalize = alpha/(1 + 6 * alpha);

  out.setSid(sid);
  if(isNotZeroV(tmp)) {
    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++) 
      #pragma unroll
      for(int c=0; c<N_COLS; c++)
	out[mu*N_COLS + c] += normalize * tmp[mu][c];
  }
}


template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing(vectorTex<FloatOut>& out, vectorTex<FloatIn>& vecInTex, 
			      gaugeTex<FloatGauge>& gaugeTex, FloatOut alpha){
  ProfileStruct ps(out.volume());
  tuneAndRun(ps, "gaussian_smearing_kernel", gaussian_smearing_kernel<FloatOut,FloatIn,FloatGauge,false>,
	     out, vecInTex, gaugeTex, alpha);
}

template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing_no_ghost(vectorTex<FloatOut>& out, vectorTex<FloatIn>& vecInTex, 
				       gaugeTex<FloatGauge>& gaugeTex, FloatOut alpha){
  ProfileStruct ps(out.volume());
  tuneAndRun(ps, "gaussian_smearing_no_ghost_kernel", gaussian_smearing_kernel<FloatOut,FloatIn,FloatGauge,true>,
	     out, vecInTex, gaugeTex, alpha);
}

template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing_only_ghost(vectorTex<FloatOut>& out, vectorTex<FloatIn>& vecInTex, 
					 gaugeTex<FloatGauge>& gaugeTex, FloatOut alpha){
  for(int dir = 0; dir < N_DIMS-1; dir++) {
    if(HGC_dimBreak[dir]) {
      ProfileStruct ps(out.sideGhostL(dir)*2);
      tuneAndRun(ps, "gaussian_smearing_only_ghost_kernel", gaussian_smearing_only_ghost_kernel<FloatOut,FloatIn,FloatGauge>,
		 out, vecInTex, gaugeTex, alpha, dir);
    }
  }
}
