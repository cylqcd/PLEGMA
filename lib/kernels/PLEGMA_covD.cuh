#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>
using namespace plegma;
template<typename FloatOut, typename FloatIn, typename FloatGauge>
__global__ void covD_kernel(FloatOut* out,
			    vectorTex<FloatIn> vTex, 
			    gaugeTex<FloatGauge> gTex, int dirOr){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;
  
  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn> Sin[N_SPINS][N_COLS];
  Float2<FloatOut> Sout[N_SPINS][N_COLS];
  vector2<FloatOut> out2(out);
  int dir = dirOr%4;
  
  if(dirOr < 4){
    gTex.get(G,dir,sid);
    vTex.get<Plus>(Sin,sid,dir);
    mul_G_V(Sout,G,Sin);
    out2.set(Sout,sid);
  }
  else{
    gTex.get<Minus>(G,dir,sid,dir);
    vTex.get<Minus>(Sin,sid,dir);
    mul_Gdag_V(Sout,G,Sin);
    out2.set(Sout,sid);
  }
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covD_k(FloatOut *out, vectorTex<FloatIn> v, gaugeTex<FloatGauge> g, int dirOr){
  ProfileStruct ps(HGC_localVolume);
  tuneAndRun(ps, "covD_kernel", covD_kernel<FloatOut,FloatIn,FloatGauge>, out, v, g, dirOr);
  checkCudaError();
}
