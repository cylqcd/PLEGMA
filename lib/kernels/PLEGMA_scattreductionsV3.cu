#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas.cuh>

using namespace plegma;

template<typename FloatOut, typename FloatV, typename FloatP, unsigned int N_GAMMAS>
__global__ void V3_kernel( FloatV *Phi, KernelArr<GAMMAS> listGammas,
			   FloatP *S, Float2<FloatOut> *block2,
			   int it, int time_step, int3 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int vid = sid3D + (it+tid)*DGC_localVolume3D;
  int site_size = N_SPINS*N_COLS;

  register Float2<FloatOut> accum[N_GAMMAS*N_SPINS*N_COLS];
  for(int i = 0 ; i <N_GAMMAS*N_SPINS*N_COLS  ; i++){
    accum[i] = 0.;
  }
  
  if (sid3D < DGC_localVolume3D){
    prop2<FloatP> propS(S);
    vector2<FloatV> vectorPhi(Phi);
    
    Float2<FloatP> s[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatV> phi[N_SPINS][N_COLS];
    propS.get(s,vid);
    vectorPhi.get(phi,vid);

    
    const Float2<float> (*g)[4];
    const short int (*gammasIdx)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gammasIdx = gammaInd;

    #pragma unroll
    for(int i_g = 0 ; i_g < N_GAMMAS; i_g++){
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
    
  #pragma unroll
  for(int i_g = 0 ; i_g < N_GAMMAS; i_g++)
    fourier_transform_3D(block2+i_g*site_size*grid3D, accum+i_g*site_size, shared_cache, site_size, sid3D, source_pos, moms, (N_GAMMAS-1)*site_size, -1, time_step, tid); //+
 
}

template<typename FloatOut, typename FloatV, typename FloatP>
void V3_kernel_wrapper( ProfileStruct &ps, Float2<FloatOut> *block2,
			int it, int time_step, int3 source, tex_mom_list moms,
			KernelArr<GAMMAS> &listGammas, FloatV *Phi, FloatP *S){
  dim3 grid = ps.tp.grid;
  grid.x = (grid.x/time_step)*MIN(HGC_localL[3]-it, time_step);

  switch(listGammas.size){
  case(1): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)1><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(2): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)2><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(3): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)3><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(4): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)4><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  /* case(5): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)5><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(6): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)6><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(7): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)7><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(8): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)8><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(9): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)9><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(10): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)10><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(11): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)11><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(12): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)12><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(13): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)13><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(14): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)14><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(15): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)15><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  /* case(16): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)16><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break; */
  }
  
}
