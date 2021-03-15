#include <PLEGMA_kernel_utils.cuh>
#include <../../include/PLEGMA_gammas.h>

using namespace plegma;

template<typename FloatOut, typename FloatV, typename FloatP, unsigned int N_GAMMAS_SCATT>
__global__ void V3_kernel( vectorTex<FloatV> vectorPhi, KernelArr<GAMMAS_SCATT> listGammas,
			   propTex<FloatP> propS, Float2<FloatOut> *block2,
			   int it, int time_step, int maxT, int4 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;

  register Float2<FloatOut> accum[N_GAMMAS_SCATT*N_SPINS*N_COLS];
  for(int i = 0 ; i <N_GAMMAS_SCATT*N_SPINS*N_COLS  ; i++){
    accum[i] = 0.;
  }
  
  if (sid3D < DGC_localVolume3D){
    Float2<FloatP> s[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatV> phi[N_SPINS][N_COLS];
    propS.get(s,vid);
    vectorPhi.get(phi,vid);

    
    const Float2<float> (*g)[4];
    const short int (*gammasIdx)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma_scatt;
    gammasIdx = gammaInd_scatt;

    #pragma unroll
    for(int i_g = 0 ; i_g < N_GAMMAS_SCATT; i_g++){
      int gId=listGammas.array[i_g];
      
      #pragma unroll //for loop over nonzero entries
      for(int nz_e = 0 ; nz_e < 4 ; nz_e++){
	int alpha0=gammasIdx[gId][nz_e][0];
	int alpha1=gammasIdx[gId][nz_e][1];
	Float2<FloatOut> factor=g[gId][nz_e];
	
        #pragma unroll
	for(int beta = 0 ; beta < N_SPINS ; beta++){
          #pragma unroll
	  for(int a = 0 ; a < N_COLS ; a++){
            #pragma unroll
	    for(int b = 0 ; b < N_COLS ; b++){
	      accum[(i_g*N_SPINS + beta)*N_COLS+b] =
		accum[(i_g*N_SPINS + beta)*N_COLS+b]
		+ conj(phi[alpha0][a])*factor*s[alpha1][beta][a][b];
	    }
	  }
	}
      }
    }
  }
      
  extern __shared__ int ext_shared_cache[];
  Float2<FloatOut> *shared_cache = (Float2<FloatOut> *) ext_shared_cache;
  int source_pos[3] = {source.x, source.y, source.z};

  const unsigned int OUT_DOF= N_GAMMAS_SCATT;
  const unsigned int IN_DOF= N_SPINS*N_COLS;

  #pragma unroll
  for(int i_gs = 0 ; i_gs < OUT_DOF; i_gs++)
    fourier_transform_3D(block2+i_gs*IN_DOF*grid3D, accum+i_gs*IN_DOF, shared_cache, IN_DOF, sid3D, source_pos, moms, (OUT_DOF-1)*IN_DOF, -1, time_step, tid); 
}

