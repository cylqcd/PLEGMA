#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas.cuh>

using namespace plegma;

template<typename FloatOut, typename FloatV, typename FloatP, unsigned int N_GAMMAS>
__global__ void V2_kernel( FloatV *Phi, KernelArr<GAMMAS> listGammas,
			   FloatP *S1, FloatP *S2, Float2<FloatOut> *block2,
			   int it, int time_step, int3 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int vid = sid3D + (it+tid)*DGC_localVolume3D;
  int site_size = N_GAMMAS*N_SPINS*N_SPINS*N_SPINS*N_COLS;


  register Float2<FloatOut> accum[N_GAMMAS*N_SPINS*N_SPINS*N_SPINS*N_COLS];
  for(int i = 0 ; i <N_GAMMAS*N_SPINS*N_SPINS*N_SPINS*N_COLS  ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC_localVolume3D){
    prop2<FloatP> propS1(S1), propS2(S2);
    vector2<FloatV> vectorPhi(Phi);
    
    Float2<FloatP> s1[N_SPINS][N_SPINS][N_COLS][N_COLS], s2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatV> phi[N_SPINS][N_COLS];
    propS1.get(s1,vid);
    propS1.get(s2,vid);
    vectorPhi.get(phi,vid);

    const unsigned short N_S1C=N_SPINS*N_COLS;
    const unsigned short N_S2C=N_SPINS*N_SPINS*N_COLS;
    const unsigned short N_S3C=N_SPINS*N_SPINS*N_SPINS*N_COLS;
    
    
    const Float2<float> (*g)[4];
    const short (*gammasIdx)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gammasIdx = gammaInd;

    #pragma unroll 
    for( unsigned short alfa1=0; alfa1<N_SPINS; alfa1++){
      #pragma unroll 
      for( unsigned short alfa2=0; alfa2<N_SPINS; alfa2++ ){
        #pragma unroll 
	for( unsigned short alfa0=0; alfa0<N_SPINS; alfa0++ ){
          #pragma unroll 
	  for(unsigned short n_g=0; n_g<N_GAMMAS; n_g++ ){
	    int gId=listGammas.array[n_g];
            #pragma unroll 
	    for(int nz_e = 0 ; nz_e < 4 ; nz_e++){
	      int beta0=gammasIdx[gId][nz_e][0];
	      int beta1=gammasIdx[gId][nz_e][1];
	      Float2<FloatOut> factor=g[gId][nz_e];
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
		  accum[ n_g*N_S3C + alfa0*N_S2C + alfa1*N_S1C + alfa2*N_COLS + n ] =
		    accum[ n_g*N_S3C + alfa0*N_S2C + alfa1*N_S1C + alfa2*N_COLS + n ] + eps1_sgn*eps2_sgn*phi[beta0][c]*factor*s1[beta1][alfa0][b][m]*s2[alfa1][alfa2][a][l];
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
  fourier_transform_3D(block2, accum, shared_cache, site_size, sid3D, source_pos, moms, 0, -1, time_step, tid);
 
}


template<typename FloatOut, typename FloatV, typename FloatP>
void V2_kernel_wrapper( ProfileStruct &ps, Float2<FloatOut> *block2,
			int it, int time_step, int3 source, tex_mom_list moms,
			KernelArr<GAMMAS> &listGammas, FloatV *Phi, FloatP *S1, FloatP *S2){
  dim3 grid = ps.tp.grid;
  grid.x = (grid.x/time_step)*MIN(HGC_localL[3]-it, time_step);

  switch(listGammas.size){
  case(1): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)1><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(2): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)2><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(3): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)3><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(4): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)4><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(5): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)5><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(6): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)6><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(7): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)7><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(8): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)8><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(9): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)9><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(10): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)10><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(11): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)11><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(12): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)12><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(13): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)13><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(14): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)14><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(15): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)15><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(16): V2_kernel<FloatOut,FloatV,FloatP,(unsigned int)16><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  }
}
