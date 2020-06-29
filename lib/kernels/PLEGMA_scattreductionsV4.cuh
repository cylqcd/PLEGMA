#include <PLEGMA_kernel_utils.cuh>
#include <../../include/PLEGMA_gammas.h>

using namespace plegma;

template<typename FloatOut, typename FloatV, typename FloatP, unsigned int N_GAMMAS_SCATT>
__global__ void V4_kernel( vectorTex<FloatV> vectorPhi, KernelArr<GAMMAS_SCATT> listGammas,
			   propTex<FloatP> propS1, propTex<FloatP> propS2, Float2<FloatOut> *block2,
			   int it, int time_step, int maxT, int4 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;

  register Float2<FloatOut> accum[N_GAMMAS_SCATT*N_SPINS*N_SPINS*N_SPINS*N_COLS];
  for(int i = 0 ; i <N_GAMMAS_SCATT*N_SPINS*N_SPINS*N_SPINS*N_COLS  ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC_localVolume3D){
    Float2<FloatP> s1[N_SPINS][N_SPINS][N_COLS][N_COLS], s2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatV> phi[N_SPINS][N_COLS];
    propS1.get(s1,vid);
    propS2.get(s2,vid);
    vectorPhi.get(phi,vid);

    const unsigned short N_S1C=N_SPINS*N_COLS;
    const unsigned short N_S2C=N_SPINS*N_SPINS*N_COLS;
    const unsigned short N_S3C=N_SPINS*N_SPINS*N_SPINS*N_COLS;
    
    
    const Float2<float> (*g)[4];
    const short (*gammasIdx)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma_scatt;
    gammasIdx = gammaInd_scatt;

    #pragma unroll 
    for( unsigned short alfa1=0; alfa1<N_SPINS; alfa1++){
      #pragma unroll 
      for( unsigned short alfa2=0; alfa2<N_SPINS; alfa2++ ){
        #pragma unroll 
      	for(unsigned short n_g=0; n_g<N_GAMMAS_SCATT; n_g++ ){
	  int gId=listGammas.array[n_g];
          #pragma unroll 
	  for(int nz_e = 0 ; nz_e < 4 ; nz_e++){
	    int beta0=gammasIdx[gId][nz_e][0];
	    int beta1=gammasIdx[gId][nz_e][1];
	    Float2<FloatOut> factor_gamma=g[gId][nz_e];
	    #pragma unroll
	    for( unsigned short alfa0=0; alfa0<N_SPINS; alfa0++){
	      #pragma unroll
	      for( unsigned short eps1_nz=0; eps1_nz<6; eps1_nz++ ){
		unsigned short a=plegma::eps[eps1_nz][0];
		unsigned short b=plegma::eps[eps1_nz][1];
		unsigned short c=plegma::eps[eps1_nz][2];
		int eps1_sgn=plegma::sgn_eps[eps1_nz];
		#pragma unroll
		for( unsigned short eps2_nz=0; eps2_nz<6; eps2_nz++ ){
		  unsigned short l=plegma::eps[eps2_nz][0];
		  unsigned short m=plegma::eps[eps2_nz][1];
		  unsigned short n=plegma::eps[eps2_nz][2];
		  int eps2_sgn=plegma::sgn_eps[eps2_nz];
                  Float2<FloatOut> factor=eps1_sgn*eps2_sgn*factor_gamma;
		  accum[ n_g*N_S3C + alfa0*N_S2C + alfa1*N_S1C + alfa2*N_COLS + l] =
		    accum[ n_g*N_S3C + alfa0*N_S2C + alfa1*N_S1C + alfa2*N_COLS + l] + factor*phi[alfa0][a]*s1[beta1][alfa1][b][m]*s2[beta0][alfa2][c][n];
		}
	      }
	    }
	  }
	}
      }
    }
  }

  extern __shared__ int ext_shared_cache[];
  Float2<FloatOut> *shared_cache = (Float2<FloatOut> *) ext_shared_cache;
  int source_pos[3] = {source.x, source.y, source.z};

  const unsigned int OUT_DOF= N_GAMMAS_SCATT*N_SPINS*N_SPINS;
  const unsigned int IN_DOF= N_SPINS*N_COLS;

  #pragma unroll
  for(int i_gs = 0 ; i_gs < OUT_DOF; i_gs++)
    fourier_transform_3D(block2+i_gs*IN_DOF*grid3D, accum+i_gs*IN_DOF, shared_cache, IN_DOF, sid3D, source_pos, moms, (OUT_DOF-1)*IN_DOF, -1, time_step, tid);

}
