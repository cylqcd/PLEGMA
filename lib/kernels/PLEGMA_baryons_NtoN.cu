#include <PLEGMA_kernel_utils.cuh>

static const __device__ short int NtoN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float NtoN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_NtoN_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
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

template __device__ void contract_NtoN_kernel<float,float,float>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_NtoN_kernel<double,double,double>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);


