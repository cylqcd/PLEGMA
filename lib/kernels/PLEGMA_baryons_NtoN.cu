#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_projectors.cuh>

static const __device__ short int NtoN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float NtoN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_NtoN_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
  Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  texProp1.get(prop1,vid);
  texProp2.get(prop2,vid);
#pragma unroll
  for(int mu = 0 ; mu < 4 ; mu++)
#pragma unroll
    for(int mup = 0 ; mup < 4 ; mup++) {
      Float2<FloatA> tmp[2] = {0., 0.};
#pragma unroll
      for(int idx = 0 ; idx < 16 ; idx++){
	int nu = NtoN_indices[idx][0];
	int lu = NtoN_indices[idx][1];
	int lup = NtoN_indices[idx][2];
	int nup = NtoN_indices[idx][3];
#pragma unroll
	for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	  int a = eps[cc1][0];
	  int b = eps[cc1][1];
	  int c = eps[cc1][2];
#pragma unroll
	  for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	    int a1 = eps[cc2][0];
	    int b1 = eps[cc2][1];
	    int c1 = eps[cc2][2];
	    FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * NtoN_values[idx];
	    tmp[0] = tmp[0] + factor * prop2[lu][lup][b][b1] *
	      (prop1[nu][nup][a][a1] * prop1[mu][mup][c][c1] - prop1[nu][mup][a][c1] * prop1[mu][nup][c][a1]);
	    tmp[1] = tmp[1] + factor * prop1[lu][lup][b][b1] *
	      (prop2[nu][nup][a][a1] * prop2[mu][mup][c][c1] - prop2[nu][mup][a][c1] * prop2[mu][nup][c][a1]);
	  }
	}
      }
#pragma unroll
      for(int i = 0 ; i < 2 ; i++){
	accum[(i*N_SPINS + mu)*N_SPINS+mup] = tmp[i];
      }
    }
}

template __device__ void contract_NtoN_kernel<float,float,float>(propTex<float> texProp1, propTex<float> texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_NtoN_kernel<double,double,double>(propTex<double> texProp1, propTex<double> texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatC, typename FloatA, typename FloatB>
struct ArgsNucleonSeqSource{
  FloatC* vec;
  cudaTextureObject_t prop1;
  cudaTextureObject_t prop2;
  WHICHPROJECTOR proj;
  WHICHPARTICLE particle;
  int timeslice;
};

template<typename Float>
inline __device__ Float2<Float> fetch(cudaTextureObject_t tex,size_t i);
template<> inline __device__ Float2<float> fetch<float>(cudaTextureObject_t tex,size_t i) {
    return (Float2<float>) tex1Dfetch<float2>(tex,i);
}
template<> inline __device__ Float2<double> fetch<double>(cudaTextureObject_t tex,size_t i) {
    int4 v = tex1Dfetch<int4>(tex,i);
    return (Float2<double>) make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
template<typename Float>
inline __device__ Float2<Float> get(cudaTextureObject_t tex, int i, int stride, int sid){
  return fetch<Float>(i*stride + sid);
}
template<typename Float>
inline __device__ void get(cudaTextureObject_t tex, Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS], int stride, int sid){
#pragma unroll
  for(short int mu = 0 ; mu < N_SPINS; mu++)
#pragma	unroll
    for(short int nu = 0 ; nu < N_SPINS; nu++)
#pragma unroll
      for(short int c1 = 0 ; c1 < N_COLS ; c1++)
#pragma unroll
	for(short int c2 = 0 ; c2 < N_COLS ; c2++)
	  P[mu][nu][c1][c2] = get(tex,(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2),stride,sid);
}


template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2>
__global__ void contractNucleonSeqSource(ArgsNucleonSeqSource<FloatC,FloatA,FloatB> args){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= c_stride/c_local[3]) return;
  Float2<FloatC> *vec2 = (Float2<FloatC> *) args.vec;
  
  const Float2<float> (*pr)[8];
  const short int (*prInd)[8][2];
  if(args.particle == PROTON){
    pr = (Float2<float> (*)[8]) projTmP;
    prInd = projIndTmP;
  }
  else if(args.particle == NEUTRON){
    pr = (Float2<float> (*)[8]) projTmM;
    prInd = projIndTmM;    
  }
  else{
    printf("Error: You can use only PROTON or NEUTRON\n");
    asm("trap;"); 
  }

  Float2<Float> P[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<Float> P2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<Float> spinor[N_SPINS][N_COLS];
  get(prop1,P,c_stride/c_localL[3],sid);
  if(isTwoPropDiff) get(prop2,P,c_stride/c_localL[3],sid);

  
  if(!isTwoPropDiff){
    for(int cc1 = 0 ; cc1 < 6 ; cc1++){
      c1 = eps[cc1][0];
      c2 = eps[cc1][1];
      c3 = eps[cc1][2];
      for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	c1p = eps[cc2][0];
	c2p = eps[cc2][1];
	c3p = eps[cc2][2];
	if(c3p == c_c2){
	  for(int idx = 0 ; idx < 16 ; idx++){
	    mu = NTN_indices[idx][0];
	    nu = NTN_indices[idx][1];
	    ku = NTN_indices[idx][2];
	    lu = NTN_indices[idx][3];
	    if(lu == c_nu){
	      for(int nz = 0; nz < 8; nz++){
		int b = prInd[args.proj][nz][0];
		int a = prInd[args.proj][nz][1];
		Float2<FloatC> factor = (-1)*c_sgn_eps[cc1]*c_sgn_eps[cc2]*c_NTN_values[idx]*pr[args.proj][nz];
		spinor[nu][c3] = spinor[nu][c3] + factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];
		spinor[nu][c3] = spinor[nu][c3] + factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
	      }}}}}}}
  else{ 

  }
  
}
