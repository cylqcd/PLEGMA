#include <PLEGMA_kernel_utils.cuh>
#include <../../include/PLEGMA_gammas.h>

using namespace plegma;

template<typename FloatOut, typename FloatV, typename FloatP, unsigned int N_GAMMAS_SCATT, bool CONJ_V>
__global__ void V5_kernel( vectorTex<FloatV> vectorPhi1, vectorTex<FloatP> vectorPhi2,
			   Float2<FloatOut> *block2,
			   int it, int time_step, int maxT, int4 source, tex_mom_list moms){
 
  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  register Float2<FloatOut> accum[N_SPINS*N_SPINS*N_COLS];
  for(int i = 0 ; i <N_SPINS*N_SPINS*N_COLS  ; i++){
    accum[i] = 0.;
  }
  
  if (sid3D < DGC_localVolume3D){
    Float2<FloatV> phi1[N_SPINS][N_COLS];
    Float2<FloatV> phi2[N_SPINS][N_COLS];
    vectorPhi1.get(phi1,vid);
    vectorPhi2.get(phi2,vid);

    

    //loops
    if(CONJ_V){

       #pragma unroll
       for (int alpha=0; alpha<N_SPINS; ++alpha){

         #pragma unroll 
         for (int beta=0; beta<N_SPINS; ++beta){

           #pragma unroll
           for( unsigned short eps1_nz=0; eps1_nz<6; eps1_nz++ ){
             unsigned short m=plegma::eps[eps1_nz][0];
             unsigned short a=plegma::eps[eps1_nz][1];
             unsigned short b=plegma::eps[eps1_nz][2];
             int eps1_sgn=plegma::sgn_eps[eps1_nz];
             Float2<FloatOut> factor=eps1_sgn;	      
             accum[(alpha*N_SPINS+beta)*N_COLS+m] = accum[(alpha*N_SPINS+beta)*N_COLS+m] 
		    + conj(phi1[alpha][a])*conj(phi2[beta][b])*factor;
	   }
         }
      }
    }
    else {
       #pragma unroll
       for(int alpha=0; alpha<N_SPINS; ++alpha){
        
         #pragma unroll
         for(int beta=0; beta<N_SPINS; ++beta){
         
           for( unsigned short m=0; m<N_COLS; m++ ){
         /*  #pragma unroll
           for( unsigned short eps1_nz=0; eps1_nz<6; eps1_nz++ ){
             unsigned short m=plegma::eps[eps1_nz][0];
             unsigned short a=plegma::eps[eps1_nz][1];
             unsigned short b=plegma::eps[eps1_nz][2];
             int eps1_sgn=plegma::sgn_eps[eps1_nz];
             Float2<FloatOut> factor=eps1_sgn;*/
	     accum[(alpha*N_SPINS+beta)*N_COLS+m] = accum[(alpha*N_SPINS+beta)*N_COLS+m]+phi1[0][0];
//                    + (phi1[alpha][a])*(phi2[beta][b])*factor;

           }       
         }
       }
    }
    printf("V5 gridDim %d blockIdx.x %d blockDim.x %d threadIdx.x %d blockIdx.x %d sid3D %d grid3D %d t %d vid %d it %d time step %d\n", gridDim.x,blockIdx.x, blockDim.x, threadIdx.x, blockIdx.x, sid3D, grid3D, t, vid, it, time_step); 
  }
  extern __shared__ int ext_shared_cache[];
  Float2<FloatOut> *shared_cache = (Float2<FloatOut> *) ext_shared_cache;
  int source_pos[3] = {source.x, source.y, source.z};
  
  const unsigned int OUT_DOF= N_SPINS;
  const unsigned int IN_DOF= N_SPINS*N_COLS;

  #pragma unroll
  for(int i_gs = 0 ; i_gs < OUT_DOF; i_gs++)
    fourier_transform_3D(block2+i_gs*IN_DOF*grid3D, accum+i_gs*IN_DOF, shared_cache, IN_DOF, sid3D, source_pos, moms, (OUT_DOF-1)*IN_DOF, -1, time_step, tid);
}

