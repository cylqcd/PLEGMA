#include <PLEGMA_kernel_utils.cuh>

static const __device__ short int NTN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float NTN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};
static const __device__ short int NTR_indices[64][6] = {0,1,0,3,0,2,0,1,0,3,1,3,0,1,0,3,2,0,0,1,0,3,3,1,0,1,1,2,0,2,0,1,1,2,1,3,0,1,1,2,2,0,0,1,1,2,3,1,0,1,2,1,0,2,0,1,2,1,1,3,0,1,2,1,2,0,0,1,2,1,3,1,0,1,3,0,0,2,0,1,3,0,1,3,0,1,3,0,2,0,0,1,3,0,3,1,1,0,0,3,0,2,1,0,0,3,1,3,1,0,0,3,2,0,1,0,0,3,3,1,1,0,1,2,0,2,1,0,1,2,1,3,1,0,1,2,2,0,1,0,1,2,3,1,1,0,2,1,0,2,1,0,2,1,1,3,1,0,2,1,2,0,1,0,2,1,3,1,1,0,3,0,0,2,1,0,3,0,1,3,1,0,3,0,2,0,1,0,3,0,3,1,2,3,0,3,0,2,2,3,0,3,1,3,2,3,0,3,2,0,2,3,0,3,3,1,2,3,1,2,0,2,2,3,1,2,1,3,2,3,1,2,2,0,2,3,1,2,3,1,2,3,2,1,0,2,2,3,2,1,1,3,2,3,2,1,2,0,2,3,2,1,3,1,2,3,3,0,0,2,2,3,3,0,1,3,2,3,3,0,2,0,2,3,3,0,3,1,3,2,0,3,0,2,3,2,0,3,1,3,3,2,0,3,2,0,3,2,0,3,3,1,3,2,1,2,0,2,3,2,1,2,1,3,3,2,1,2,2,0,3,2,1,2,3,1,3,2,2,1,0,2,3,2,2,1,1,3,3,2,2,1,2,0,3,2,2,1,3,1,3,2,3,0,0,2,3,2,3,0,1,3,3,2,3,0,2,0,3,2,3,0,3,1};
static const __device__ float NTR_values[64] = {1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1};
static const __device__ short int RTN_indices[64][6] = {0,3,0,1,0,2,0,3,0,1,1,3,0,3,0,1,2,0,0,3,0,1,3,1,0,3,1,0,0,2,0,3,1,0,1,3,0,3,1,0,2,0,0,3,1,0,3,1,0,3,2,3,0,2,0,3,2,3,1,3,0,3,2,3,2,0,0,3,2,3,3,1,0,3,3,2,0,2,0,3,3,2,1,3,0,3,3,2,2,0,0,3,3,2,3,1,1,2,0,1,0,2,1,2,0,1,1,3,1,2,0,1,2,0,1,2,0,1,3,1,1,2,1,0,0,2,1,2,1,0,1,3,1,2,1,0,2,0,1,2,1,0,3,1,1,2,2,3,0,2,1,2,2,3,1,3,1,2,2,3,2,0,1,2,2,3,3,1,1,2,3,2,0,2,1,2,3,2,1,3,1,2,3,2,2,0,1,2,3,2,3,1,2,1,0,1,0,2,2,1,0,1,1,3,2,1,0,1,2,0,2,1,0,1,3,1,2,1,1,0,0,2,2,1,1,0,1,3,2,1,1,0,2,0,2,1,1,0,3,1,2,1,2,3,0,2,2,1,2,3,1,3,2,1,2,3,2,0,2,1,2,3,3,1,2,1,3,2,0,2,2,1,3,2,1,3,2,1,3,2,2,0,2,1,3,2,3,1,3,0,0,1,0,2,3,0,0,1,1,3,3,0,0,1,2,0,3,0,0,1,3,1,3,0,1,0,0,2,3,0,1,0,1,3,3,0,1,0,2,0,3,0,1,0,3,1,3,0,2,3,0,2,3,0,2,3,1,3,3,0,2,3,2,0,3,0,2,3,3,1,3,0,3,2,0,2,3,0,3,2,1,3,3,0,3,2,2,0,3,0,3,2,3,1};
static const __device__ float RTN_values[64] = {-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1};
static const __device__ short int RTR_indices[256][8] = {0,3,0,3,0,2,0,2,0,3,0,3,0,2,1,3,0,3,0,3,0,2,2,0,0,3,0,3,0,2,3,1,0,3,0,3,1,3,0,2,0,3,0,3,1,3,1,3,0,3,0,3,1,3,2,0,0,3,0,3,1,3,3,1,0,3,0,3,2,0,0,2,0,3,0,3,2,0,1,3,0,3,0,3,2,0,2,0,0,3,0,3,2,0,3,1,0,3,0,3,3,1,0,2,0,3,0,3,3,1,1,3,0,3,0,3,3,1,2,0,0,3,0,3,3,1,3,1,0,3,1,2,0,2,0,2,0,3,1,2,0,2,1,3,0,3,1,2,0,2,2,0,0,3,1,2,0,2,3,1,0,3,1,2,1,3,0,2,0,3,1,2,1,3,1,3,0,3,1,2,1,3,2,0,0,3,1,2,1,3,3,1,0,3,1,2,2,0,0,2,0,3,1,2,2,0,1,3,0,3,1,2,2,0,2,0,0,3,1,2,2,0,3,1,0,3,1,2,3,1,0,2,0,3,1,2,3,1,1,3,0,3,1,2,3,1,2,0,0,3,1,2,3,1,3,1,0,3,2,1,0,2,0,2,0,3,2,1,0,2,1,3,0,3,2,1,0,2,2,0,0,3,2,1,0,2,3,1,0,3,2,1,1,3,0,2,0,3,2,1,1,3,1,3,0,3,2,1,1,3,2,0,0,3,2,1,1,3,3,1,0,3,2,1,2,0,0,2,0,3,2,1,2,0,1,3,0,3,2,1,2,0,2,0,0,3,2,1,2,0,3,1,0,3,2,1,3,1,0,2,0,3,2,1,3,1,1,3,0,3,2,1,3,1,2,0,0,3,2,1,3,1,3,1,0,3,3,0,0,2,0,2,0,3,3,0,0,2,1,3,0,3,3,0,0,2,2,0,0,3,3,0,0,2,3,1,0,3,3,0,1,3,0,2,0,3,3,0,1,3,1,3,0,3,3,0,1,3,2,0,0,3,3,0,1,3,3,1,0,3,3,0,2,0,0,2,0,3,3,0,2,0,1,3,0,3,3,0,2,0,2,0,0,3,3,0,2,0,3,1,0,3,3,0,3,1,0,2,0,3,3,0,3,1,1,3,0,3,3,0,3,1,2,0,0,3,3,0,3,1,3,1,1,2,0,3,0,2,0,2,1,2,0,3,0,2,1,3,1,2,0,3,0,2,2,0,1,2,0,3,0,2,3,1,1,2,0,3,1,3,0,2,1,2,0,3,1,3,1,3,1,2,0,3,1,3,2,0,1,2,0,3,1,3,3,1,1,2,0,3,2,0,0,2,1,2,0,3,2,0,1,3,1,2,0,3,2,0,2,0,1,2,0,3,2,0,3,1,1,2,0,3,3,1,0,2,1,2,0,3,3,1,1,3,1,2,0,3,3,1,2,0,1,2,0,3,3,1,3,1,1,2,1,2,0,2,0,2,1,2,1,2,0,2,1,3,1,2,1,2,0,2,2,0,1,2,1,2,0,2,3,1,1,2,1,2,1,3,0,2,1,2,1,2,1,3,1,3,1,2,1,2,1,3,2,0,1,2,1,2,1,3,3,1,1,2,1,2,2,0,0,2,1,2,1,2,2,0,1,3,1,2,1,2,2,0,2,0,1,2,1,2,2,0,3,1,1,2,1,2,3,1,0,2,1,2,1,2,3,1,1,3,1,2,1,2,3,1,2,0,1,2,1,2,3,1,3,1,1,2,2,1,0,2,0,2,1,2,2,1,0,2,1,3,1,2,2,1,0,2,2,0,1,2,2,1,0,2,3,1,1,2,2,1,1,3,0,2,1,2,2,1,1,3,1,3,1,2,2,1,1,3,2,0,1,2,2,1,1,3,3,1,1,2,2,1,2,0,0,2,1,2,2,1,2,0,1,3,1,2,2,1,2,0,2,0,1,2,2,1,2,0,3,1,1,2,2,1,3,1,0,2,1,2,2,1,3,1,1,3,1,2,2,1,3,1,2,0,1,2,2,1,3,1,3,1,1,2,3,0,0,2,0,2,1,2,3,0,0,2,1,3,1,2,3,0,0,2,2,0,1,2,3,0,0,2,3,1,1,2,3,0,1,3,0,2,1,2,3,0,1,3,1,3,1,2,3,0,1,3,2,0,1,2,3,0,1,3,3,1,1,2,3,0,2,0,0,2,1,2,3,0,2,0,1,3,1,2,3,0,2,0,2,0,1,2,3,0,2,0,3,1,1,2,3,0,3,1,0,2,1,2,3,0,3,1,1,3,1,2,3,0,3,1,2,0,1,2,3,0,3,1,3,1,2,1,0,3,0,2,0,2,2,1,0,3,0,2,1,3,2,1,0,3,0,2,2,0,2,1,0,3,0,2,3,1,2,1,0,3,1,3,0,2,2,1,0,3,1,3,1,3,2,1,0,3,1,3,2,0,2,1,0,3,1,3,3,1,2,1,0,3,2,0,0,2,2,1,0,3,2,0,1,3,2,1,0,3,2,0,2,0,2,1,0,3,2,0,3,1,2,1,0,3,3,1,0,2,2,1,0,3,3,1,1,3,2,1,0,3,3,1,2,0,2,1,0,3,3,1,3,1,2,1,1,2,0,2,0,2,2,1,1,2,0,2,1,3,2,1,1,2,0,2,2,0,2,1,1,2,0,2,3,1,2,1,1,2,1,3,0,2,2,1,1,2,1,3,1,3,2,1,1,2,1,3,2,0,2,1,1,2,1,3,3,1,2,1,1,2,2,0,0,2,2,1,1,2,2,0,1,3,2,1,1,2,2,0,2,0,2,1,1,2,2,0,3,1,2,1,1,2,3,1,0,2,2,1,1,2,3,1,1,3,2,1,1,2,3,1,2,0,2,1,1,2,3,1,3,1,2,1,2,1,0,2,0,2,2,1,2,1,0,2,1,3,2,1,2,1,0,2,2,0,2,1,2,1,0,2,3,1,2,1,2,1,1,3,0,2,2,1,2,1,1,3,1,3,2,1,2,1,1,3,2,0,2,1,2,1,1,3,3,1,2,1,2,1,2,0,0,2,2,1,2,1,2,0,1,3,2,1,2,1,2,0,2,0,2,1,2,1,2,0,3,1,2,1,2,1,3,1,0,2,2,1,2,1,3,1,1,3,2,1,2,1,3,1,2,0,2,1,2,1,3,1,3,1,2,1,3,0,0,2,0,2,2,1,3,0,0,2,1,3,2,1,3,0,0,2,2,0,2,1,3,0,0,2,3,1,2,1,3,0,1,3,0,2,2,1,3,0,1,3,1,3,2,1,3,0,1,3,2,0,2,1,3,0,1,3,3,1,2,1,3,0,2,0,0,2,2,1,3,0,2,0,1,3,2,1,3,0,2,0,2,0,2,1,3,0,2,0,3,1,2,1,3,0,3,1,0,2,2,1,3,0,3,1,1,3,2,1,3,0,3,1,2,0,2,1,3,0,3,1,3,1,3,0,0,3,0,2,0,2,3,0,0,3,0,2,1,3,3,0,0,3,0,2,2,0,3,0,0,3,0,2,3,1,3,0,0,3,1,3,0,2,3,0,0,3,1,3,1,3,3,0,0,3,1,3,2,0,3,0,0,3,1,3,3,1,3,0,0,3,2,0,0,2,3,0,0,3,2,0,1,3,3,0,0,3,2,0,2,0,3,0,0,3,2,0,3,1,3,0,0,3,3,1,0,2,3,0,0,3,3,1,1,3,3,0,0,3,3,1,2,0,3,0,0,3,3,1,3,1,3,0,1,2,0,2,0,2,3,0,1,2,0,2,1,3,3,0,1,2,0,2,2,0,3,0,1,2,0,2,3,1,3,0,1,2,1,3,0,2,3,0,1,2,1,3,1,3,3,0,1,2,1,3,2,0,3,0,1,2,1,3,3,1,3,0,1,2,2,0,0,2,3,0,1,2,2,0,1,3,3,0,1,2,2,0,2,0,3,0,1,2,2,0,3,1,3,0,1,2,3,1,0,2,3,0,1,2,3,1,1,3,3,0,1,2,3,1,2,0,3,0,1,2,3,1,3,1,3,0,2,1,0,2,0,2,3,0,2,1,0,2,1,3,3,0,2,1,0,2,2,0,3,0,2,1,0,2,3,1,3,0,2,1,1,3,0,2,3,0,2,1,1,3,1,3,3,0,2,1,1,3,2,0,3,0,2,1,1,3,3,1,3,0,2,1,2,0,0,2,3,0,2,1,2,0,1,3,3,0,2,1,2,0,2,0,3,0,2,1,2,0,3,1,3,0,2,1,3,1,0,2,3,0,2,1,3,1,1,3,3,0,2,1,3,1,2,0,3,0,2,1,3,1,3,1,3,0,3,0,0,2,0,2,3,0,3,0,0,2,1,3,3,0,3,0,0,2,2,0,3,0,3,0,0,2,3,1,3,0,3,0,1,3,0,2,3,0,3,0,1,3,1,3,3,0,3,0,1,3,2,0,3,0,3,0,1,3,3,1,3,0,3,0,2,0,0,2,3,0,3,0,2,0,1,3,3,0,3,0,2,0,2,0,3,0,3,0,2,0,3,1,3,0,3,0,3,1,0,2,3,0,3,0,3,1,1,3,3,0,3,0,3,1,2,0,3,0,3,0,3,1,3,1};
static const __device__ float RTR_values[256] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const __device__ short int Delta_indices[3][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float Delta_values[3][16] = {1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__device__ void contract_NtoN_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (vid < c_threads/c_localL[3]){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    for(int mu = 0 ; mu < 4 ; mu++)
      for(int mup = 0 ; mup < 4 ; mup++) {
	Float2<FloatA> tmp[2] = {0., 0.};
        #pragma unroll
	for(int idx = 0 ; idx < 16 ; idx++){
	  int nu = NTN_indices[idx][0];
	  int lu = NTN_indices[idx][1];
	  int lup = NTN_indices[idx][2];
	  int nup = NTN_indices[idx][3];
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
	      FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * NTN_values[idx];
	      tmp[0] = tmp[0] + factor * prop2[lu][lup][b][b1] *
		(prop1[nu][nup][a][a1] * prop1[mu][mup][c][c1] - prop1[nu][mup][a][c1] * prop1[mu][nup][c][a1]);
	      tmp[1] = tmp[1] + factor * prop1[lu][lup][b][b1] *
		(prop2[nu][nup][a][a1] * prop2[mu][mup][c][c1] - prop2[nu][mup][a][c1] * prop2[mu][nup][c][a1]);
	    }
	  }
	}
	for(int i = 0 ; i < 2 ; i++){
	  accum[(i*N_SPINS + mu)*N_SPINS+mup] = tmp[i];
	}
      }
  }
}

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__device__ void contract_NtoR_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (vid < c_threads/c_localL[3]){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    #pragma unroll
    for(int mu = 0 ; mu < 4 ; mu++){
      Float2<FloatA> tmp[2*N_SPINS];
      #pragma unroll
      for(int idx = 0 ; idx < 64 ; idx++){
	int nu = NTR_indices[idx][0];
	int lu = NTR_indices[idx][1];
	int lup = NTR_indices[idx][2];
	int nup = NTR_indices[idx][3];
	int mup = NTR_indices[idx][4];
	int rho = NTR_indices[idx][5];
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
	    FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * NTR_values[idx];
	    tmp[mup] = tmp[mup] - factor*prop2[lu][lup][b][b1]*
	      (prop1[nu][nup][a][a1] * prop1[mu][rho][c][c1] - prop1[nu][rho][a][c1] * prop1[mu][nup][c][a1]);
	    tmp[N_SPINS+mup] = tmp[N_SPINS+mup] - factor*prop1[lu][lup][b][b1]*
	      (prop2[nu][nup][a][a1] * prop2[mu][rho][c][c1] - prop2[nu][rho][a][c1] * prop2[mu][nup][c][a1]);
	  }
	}
      }
      #pragma unroll
      for(int i = 0 ; i < 2 ; i++){
        #pragma unroll
	for(int mup = 0 ; mup < N_SPINS ; mup++){
	  accum[(i*N_SPINS + mu)*N_SPINS+mup] = tmp[i*N_SPINS+mup];
	}
      }
    }
  }
}
 
template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__device__ void contract_RtoN_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (vid < c_threads/c_localL[3]){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    for(int mup = 0 ; mup < 4 ; mup++){
      Float2<FloatA> tmp[2*N_SPINS];
      for(int idx = 0 ; idx < 64 ; idx++){
	int nu = RTN_indices[idx][0];
	int lu = RTN_indices[idx][1];
	int lup = RTN_indices[idx][2];
	int nup = RTN_indices[idx][3];
	int mu = RTN_indices[idx][4];
	int rho = RTN_indices[idx][5];
	for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	  int a = eps[cc1][0];
	  int b = eps[cc1][1];
	  int c = eps[cc1][2];
	  for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	    int a1 = eps[cc2][0];
	    int b1 = eps[cc2][1];
	    int c1 = eps[cc2][2];
	    FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * RTN_values[idx];
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
}
 
template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__device__ void contract_RtoR_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid) {
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (vid < c_threads/c_localL[3]){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    for(int idx = 0 ; idx < 64 ; idx++){
      int nu = RTR_indices[idx][0];
      int lu = RTR_indices[idx][1];
      int lup = RTR_indices[idx][2];
      int nup = RTR_indices[idx][3];
      int mu = RTR_indices[idx][4];
      int rho = RTR_indices[idx][5];
      int mup = RTR_indices[idx][6];
      int rhop = RTR_indices[idx][7];
      for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	int a = eps[cc1][0];
	int b = eps[cc1][1];
	int c = eps[cc1][2];
	for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	    int a1 = eps[cc2][0];
	    int b1 = eps[cc2][1];
	    int c1 = eps[cc2][2];
	    FloatC factor = sgn_eps[cc1] * sgn_eps[cc2] * RTR_values[idx];
	    accum[mu*N_SPINS+mup] = accum[mu*N_SPINS+mup] - factor*prop2[lu][lup][b][b1]*
	      (prop1[nu][nup][a][a1] * prop1[rho][rhop][c][c1] - prop1[nu][rhop][a][c1] * prop1[rho][nup][c][a1]);
	    accum[(N_SPINS+mu)*N_SPINS+mup] = accum[(N_SPINS+mu)*N_SPINS+mup] - factor*prop1[lu][lup][b][b1]*
	      (prop2[nu][nup][a][a1] * prop2[rho][rhop][c][c1] - prop2[nu][rhop][a][c1] * prop2[rho][nup][c][a1]);
	}
      }
    }
  }
}
 
template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__global__ void contract_baryons_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, FloatC* block,
					int it, int x0, int y0, int z0, BARYONS_TYPE ip){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*c_stride_spatial;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;

  Float2<FloatC> accum[2*N_SPINS*N_SPINS];
  
  switch(ip){
  case NtoN:
    contract_NtoN_kernel<FloatA,FloatB,FloatC,runFT>(texProp1, texProp2, accum, vid);
    break;
  case NtoR:
      contract_NtoN_kernel<FloatA,FloatB,FloatC,runFT>(texProp1, texProp2, accum, vid);
      break;
  case RtoN:
    contract_NtoN_kernel<FloatA,FloatB,FloatC,runFT>(texProp1, texProp2, accum, vid);
    break;
  case RtoR:
    contract_NtoN_kernel<FloatA,FloatB,FloatC,runFT>(texProp1, texProp2, accum, vid);
    break;
  }
    
  if(runFT) {
    int source_pos[3] = {x0, y0, z0}; 
    __shared__ Float2<FloatC> shared_cache[2*N_SPINS*N_SPINS*THREADS_PER_BLOCK];
    fourier_transform_3D(block2, accum, shared_cache, 2*N_SPINS*N_SPINS, sid, source_pos);
  } else {
    for(int ip = 0 ; ip < 2*N_SPINS*N_SPINS ; ip++){
      block2[sid*2*N_SPINS*N_SPINS + ip] = accum[ip];
    }
  }
}

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
static void contract_baryons(propTex<FloatA> texProp1, propTex<FloatB> texProp2, PLEGMA_Correlator<FloatC> &corr, int it){

  int SpVol = GK_localVolume/GK_localL[3];

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume

  FloatC *h_partial_block = NULL;
  FloatC *d_partial_block = NULL;

  int n_flavors=2;
  int site_size=N_SPINS*N_SPINS*2;
  size_t volume;
  size_t size;
  size_t alloc_size;
  if(runFT==true){
    volume = GK_Nmoms;
    size = n_flavors*site_size*volume;
    alloc_size = size * gridDim.x;
  } else {
    volume = SpVol;
    size = n_flavors*site_size*volume;
    alloc_size = size; 
  }
  h_partial_block = (FloatC*)malloc(alloc_size*sizeof(FloatC));
  if(h_partial_block == NULL) errorQuda("contract_baryons_kernel: Cannot allocate host block.\n");
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(FloatC) );
  checkCudaError();

  int isource = corr.getIdSource();
  for(int ip=0; ip<N_BARYONS; ip++) {
    cudaFuncSetCacheConfig(contract_baryons_kernel<FloatA,FloatB,FloatC,runFT>, cudaFuncCachePreferShared);
    contract_baryons_kernel<FloatA,FloatB,FloatC,runFT><<<gridDim,blockDim>>>( texProp1, texProp2, d_partial_block, it,
									       GK_sourcePosition[isource][0],
									       GK_sourcePosition[isource][1],
									       GK_sourcePosition[isource][2], (BARYONS_TYPE) ip);
    checkCudaError();
    
    cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
    checkCudaError();
    
    if(runFT==true){
      FloatC *reduction =(FloatC*) calloc(size,sizeof(FloatC));
      for(size_t i = 0 ; i < size/2; i++)
	for(int j = 0 ; j < gridDim.x; j++) {
	  reduction[i*2+0] += h_partial_block[(i*gridDim.x + j)*2+0];
	  reduction[i*2+1] += h_partial_block[(i*gridDim.x + j)*2+1];
	}
      MPI_Allreduce(reduction, h_partial_block, size, MPI_Type(reduction), MPI_SUM, GK_spaceComm);
      free(reduction);
    }
    
    FloatC *corr_ip = corr.getCorr() + ip*GK_localL[3]*size;
    for(size_t v = 0 ; v < volume; v++)
      for(int f = 0 ; f < n_flavors; f++)
	for(int i = 0 ; i < site_size; i++)
	  corr_ip[((f*GK_localL[3] + it)*volume +v)*site_size+i] = h_partial_block[(v*n_flavors+f)*site_size+i];
    
  }
  free(h_partial_block);
  cudaFree(d_partial_block);
  checkCudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_baryons(propTex<FloatA> texProp1, propTex<FloatB> texProp2,
			     PLEGMA_Correlator<FloatC> &corr, int it){
  if (corr.getCorrSpace()==POSITION_SPACE){
    contract_baryons<FloatA,FloatB,FloatC,false>(texProp1,texProp2,corr,it);
  }
  else if(corr.getCorrSpace()==MOMENTUM_SPACE) {
    contract_baryons<FloatA,FloatB,FloatC,true>(texProp1,texProp2,corr,it);
  }
  else
    errorQuda("run_contract_baryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();
}
