#include <PLEGMA_kernel_utils.cuh>

static const __device__ short int deltas_indices[3][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float deltas_values[3][16] =  {1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

// ISO1/2 Deltap and delta0, 11,22,33
template<typename FloatA, typename FloatB, typename FloatC, int gamma>
__device__ void contract_deltas_iso1o2_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
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
	int nu = deltas_indices[gamma][idx][0];
	int lu = deltas_indices[gamma][idx][1];
	int lup = deltas_indices[gamma][idx][2];
	int nup = deltas_indices[gamma][idx][3];
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
	    FloatC factor = (1./3.) * sgn_eps[cc1] * sgn_eps[cc2] * deltas_values[gamma][idx];
	    tmp[0] = tmp[0] + factor*(-4.*prop1[nu][mup][a][c1]*prop2[lu][lup][b][b1]*prop1[mu][nup][c][a1]
				      +2.*prop1[nu][lup][a][b1]*prop2[lu][mup][b][c1]*prop1[mu][nup][c][a1]
				      +2.*prop1[nu][mup][a][c1]*prop1[lu][nup][b][a1]*prop2[mu][lup][c][b1]
				      -2.*prop1[nu][nup][a][a1]*prop1[lu][mup][b][c1]*prop2[mu][lup][c][b1]
				      -2.*prop1[nu][nup][a][a1]*prop2[lu][mup][b][c1]*prop1[mu][lup][c][b1]
				      -1.*prop1[nu][lup][a][b1]*prop1[lu][nup][b][a1]*prop2[mu][mup][c][c1]
				      +1.*prop1[nu][nup][a][a1]*prop1[lu][lup][b][b1]*prop2[mu][mup][c][c1]
				      +4.*prop1[nu][nup][a][a1]*prop2[lu][lup][b][b1]*prop1[mu][mup][c][c1] );
	    tmp[1] = tmp[1] + factor*(-4.*prop2[nu][mup][a][c1]*prop1[lu][lup][b][b1]*prop2[mu][nup][c][a1]
				      +2.*prop2[nu][lup][a][b1]*prop1[lu][mup][b][c1]*prop2[mu][nup][c][a1]
				      +2.*prop2[nu][mup][a][c1]*prop2[lu][nup][b][a1]*prop1[mu][lup][c][b1]
				      -2.*prop2[nu][nup][a][a1]*prop2[lu][mup][b][c1]*prop1[mu][lup][c][b1]
				      -2.*prop2[nu][nup][a][a1]*prop1[lu][mup][b][c1]*prop2[mu][lup][c][b1]
				      -1.*prop2[nu][lup][a][b1]*prop2[lu][nup][b][a1]*prop1[mu][mup][c][c1]
				      +1.*prop2[nu][nup][a][a1]*prop2[lu][lup][b][b1]*prop1[mu][mup][c][c1]
				      +4.*prop2[nu][nup][a][a1]*prop1[lu][lup][b][b1]*prop2[mu][mup][c][c1] );
	  }
	}
      }
#pragma unroll
      for(int i = 0 ; i < 2 ; i++){
	accum[(i*N_SPINS + mu)*N_SPINS+mup] = tmp[i];
      }
    }
}

// ISO3/2 Deltapp and deltam, 11,22,33
template<typename FloatA, typename FloatB, typename FloatC, int gamma>
__device__ void contract_deltas_iso3o2_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
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
	int nu = deltas_indices[gamma][idx][0];
	int lu = deltas_indices[gamma][idx][1];
	int lup = deltas_indices[gamma][idx][2];
	int nup = deltas_indices[gamma][idx][3];
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
	    FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * deltas_values[gamma][idx];
	    tmp[0] = tmp[0] + factor*(prop1[nu][lup][a][b1]*prop1[lu][mup][b][c1]*prop1[mu][nup][c][a1]
				      -prop1[nu][mup][a][c1]*prop1[lu][lup][b][b1]*prop1[mu][nup][c][a1]
				      +prop1[nu][mup][a][c1]*prop1[lu][nup][b][a1]*prop1[mu][lup][c][b1]
				      -prop1[nu][nup][a][a1]*prop1[lu][mup][b][c1]*prop1[mu][lup][c][b1]
				      -prop1[nu][lup][a][b1]*prop1[lu][nup][b][a1]*prop1[mu][mup][c][c1]
				      +prop1[nu][nup][a][a1]*prop1[lu][lup][b][b1]*prop1[mu][mup][c][c1] );
	    tmp[1] = tmp[1] + factor*(prop2[nu][lup][a][b1]*prop2[lu][mup][b][c1]*prop2[mu][nup][c][a1]
				      -prop2[nu][mup][a][c1]*prop2[lu][lup][b][b1]*prop2[mu][nup][c][a1]
				      +prop2[nu][mup][a][c1]*prop2[lu][nup][b][a1]*prop2[mu][lup][c][b1]
				      -prop2[nu][nup][a][a1]*prop2[lu][mup][b][c1]*prop2[mu][lup][c][b1]
				      -prop2[nu][lup][a][b1]*prop2[lu][nup][b][a1]*prop2[mu][mup][c][c1]
				      +prop2[nu][nup][a][a1]*prop2[lu][lup][b][b1]*prop2[mu][mup][c][c1] );
	  }
	}
      }
#pragma unroll
      for(int i = 0 ; i < 2 ; i++){
	accum[(i*N_SPINS + mu)*N_SPINS+mup] = tmp[i];
      }
    }
}

template __device__ void contract_deltas_iso1o2_kernel<float,float,float,0>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso1o2_kernel<float,float,float,1>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso1o2_kernel<float,float,float,2>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso3o2_kernel<float,float,float,0>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso3o2_kernel<float,float,float,1>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso3o2_kernel<float,float,float,2>(propTex<float>& texProp1, propTex<float>& texProp2, Float2<float> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso1o2_kernel<double,double,double,0>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso1o2_kernel<double,double,double,1>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso1o2_kernel<double,double,double,2>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso3o2_kernel<double,double,double,0>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso3o2_kernel<double,double,double,1>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
template __device__ void contract_deltas_iso3o2_kernel<double,double,double,2>(propTex<double>& texProp1, propTex<double>& texProp2, Float2<double> accum[2*N_SPINS*N_SPINS], int vid);
