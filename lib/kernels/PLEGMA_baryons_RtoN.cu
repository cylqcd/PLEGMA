#include "PLEGMA_kernel_utils.cuh"

static const __device__ short int RtoN_indices[64][6] = {0,3,0,1,0,2,0,3,0,1,1,3,0,3,0,1,2,0,0,3,0,1,3,1,0,3,1,0,0,2,0,3,1,0,1,3,0,3,1,0,2,0,0,3,1,0,3,1,0,3,2,3,0,2,0,3,2,3,1,3,0,3,2,3,2,0,0,3,2,3,3,1,0,3,3,2,0,2,0,3,3,2,1,3,0,3,3,2,2,0,0,3,3,2,3,1,1,2,0,1,0,2,1,2,0,1,1,3,1,2,0,1,2,0,1,2,0,1,3,1,1,2,1,0,0,2,1,2,1,0,1,3,1,2,1,0,2,0,1,2,1,0,3,1,1,2,2,3,0,2,1,2,2,3,1,3,1,2,2,3,2,0,1,2,2,3,3,1,1,2,3,2,0,2,1,2,3,2,1,3,1,2,3,2,2,0,1,2,3,2,3,1,2,1,0,1,0,2,2,1,0,1,1,3,2,1,0,1,2,0,2,1,0,1,3,1,2,1,1,0,0,2,2,1,1,0,1,3,2,1,1,0,2,0,2,1,1,0,3,1,2,1,2,3,0,2,2,1,2,3,1,3,2,1,2,3,2,0,2,1,2,3,3,1,2,1,3,2,0,2,2,1,3,2,1,3,2,1,3,2,2,0,2,1,3,2,3,1,3,0,0,1,0,2,3,0,0,1,1,3,3,0,0,1,2,0,3,0,0,1,3,1,3,0,1,0,0,2,3,0,1,0,1,3,3,0,1,0,2,0,3,0,1,0,3,1,3,0,2,3,0,2,3,0,2,3,1,3,3,0,2,3,2,0,3,0,2,3,3,1,3,0,3,2,0,2,3,0,3,2,1,3,3,0,3,2,2,0,3,0,3,2,3,1};
static const __device__ float RtoN_values[64] = {-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1};

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_RtoN_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
  Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  texProp1.get(prop1,vid);
  texProp2.get(prop2,vid);
#pragma unroll
  for(int mup = 0 ; mup < 4 ; mup++){
    Float2<FloatA> tmp[2*N_SPINS];
    for(int i = 0 ; i < 2*N_SPINS ; i++)
      tmp[i]=0;
#pragma unroll
    for(int idx = 0 ; idx < 64 ; idx++){
      int nu = RtoN_indices[idx][0];
      int lu = RtoN_indices[idx][1];
      int lup = RtoN_indices[idx][2];
      int nup = RtoN_indices[idx][3];
      int mu = RtoN_indices[idx][4];
      int rho = RtoN_indices[idx][5];
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
	  FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * RtoN_values[idx];
	  tmp[mu] = tmp[mu] + factor*prop2[lu][lup][b][b1]*
	    (prop1[nu][nup][a][a1] * prop1[rho][mup][c][c1] - prop1[nu][mup][a][c1] * prop1[rho][nup][c][a1]);
	  tmp[N_SPINS+mu] = tmp[N_SPINS+mu] + factor*prop1[lu][lup][b][b1]*
	    (prop2[nu][nup][a][a1] * prop2[rho][mup][c][c1] - prop2[nu][mup][a][c1] * prop2[rho][nup][c][a1]);
	}
      }
    }
#pragma unroll
    for(int i = 0 ; i < 2 ; i++){
#pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++){
	accum[(i*N_SPINS + mu)*N_SPINS+mup] = tmp[i*N_SPINS + mu];
      }
    }
  }
}

template __device__ void contract_RtoN_kernel<float,float,float>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_RtoN_kernel<double,double,double>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
