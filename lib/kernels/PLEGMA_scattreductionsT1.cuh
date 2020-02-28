#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas_scatt.cuh>

using namespace plegma;

template<typename FloatOut, typename FloatP, unsigned int N_GAMMAS_SCATT_I, unsigned int N_GAMMAS_SCATT_F>
__global__ void T1_kernel( KernelArr<GAMMAS_SCATT> listGammas_i, KernelArr<GAMMAS_SCATT> listGammas_f,
			   FloatP *S1, FloatP *S2, FloatP *S3, Float2<FloatOut> *block2,
			   int it, int time_step, int maxT, int4 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  //int site_size = N_GAMMAS_SCATT_I*N_GAMMAS_SCATT_F*N_SPINS*N_SPINS;

  register Float2<FloatOut> accum[N_GAMMAS_SCATT_I*N_GAMMAS_SCATT_F*N_SPINS*N_SPINS];
  for(int i = 0 ; i <N_GAMMAS_SCATT_I*N_GAMMAS_SCATT_F*N_SPINS*N_SPINS ; i++){
    accum[i] = 0.;
  }


  if (sid3D < DGC_localVolume3D){
    prop2<FloatP> propS1(S1), propS2(S2), propS3(S3);
    
    Float2<FloatP> s1[N_SPINS][N_SPINS][N_COLS][N_COLS], s2[N_SPINS][N_SPINS][N_COLS][N_COLS], s3[N_SPINS][N_SPINS][N_COLS][N_COLS];
    propS1.get(s1,vid);
    propS2.get(s2,vid);
    propS3.get(s3,vid);
    
    const Float2<float> (*gi)[4];
    const short (*gammas_i_Idx)[4][2];
    gi = (Float2<float> (*)[4]) plegma::gamma_scatt;
    gammas_i_Idx = gammaInd_scatt;

    const Float2<float> (*gf)[4];
    const short (*gammas_f_Idx)[4][2];
    gf = (Float2<float> (*)[4]) plegma::gamma_scatt;
    gammas_f_Idx = gammaInd_scatt;

    #pragma unroll 
    for(unsigned short n_gf=0; n_gf<N_GAMMAS_SCATT_F; n_gf++ ){
    
      int g_f_Id=listGammas_f.array[n_gf];

      #pragma unroll 
      for(unsigned short n_gi=0; n_gi<N_GAMMAS_SCATT_I; n_gi++ ){
	          
        int g_i_Id=listGammas_i.array[n_gi];

        #pragma unroll 
        for( unsigned short alpha=0; alpha<N_SPINS; alpha++){
              
          #pragma unroll 
          for( unsigned short beta=0; beta<N_SPINS; beta++ ){
            Float2<FloatOut> tmp=0.0;


            #pragma unroll 
            for(int nz_ef = 0 ; nz_ef < 4 ; nz_ef++){
              int beta0=gammas_f_Idx[g_f_Id][nz_ef][0];
              int beta1=gammas_f_Idx[g_f_Id][nz_ef][1];
              Float2<FloatOut> factor_f=gf[g_f_Id][nz_ef];
             
              #pragma unroll 
              for(int nz_ei = 0 ; nz_ei < 4 ; nz_ei++){
                int alpha0=gammas_i_Idx[g_i_Id][nz_ei][0];
                int alpha1=gammas_i_Idx[g_i_Id][nz_ei][1];
                Float2<FloatOut> factor_i=gi[g_i_Id][nz_ei];
        
                
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
                    Float2<FloatOut> factor=eps1_sgn*eps2_sgn*factor_i*factor_f;
                    tmp = tmp + factor*s1[alpha][alpha0][c][l]*s2[beta0][alpha1][b][m]*s3[beta1][beta][a][n];
                  }//color source
                }//color sink
              }//mult_gamma_source
            }//mult gamma_sink
            accum[ (((n_gi*N_GAMMAS_SCATT_F + n_gf)*N_SPINS) + alpha)*N_SPINS + beta ] = tmp ;
	  }//beta
	}//alpha
      }//gamma_source
    }//gamma_sink
  }

  extern __shared__ int ext_shared_cache[];
  Float2<FloatOut> *shared_cache = (Float2<FloatOut> *) ext_shared_cache;
  int source_pos[3] = {source.x, source.y, source.z}; 

  const unsigned int OUT_DOF= N_GAMMAS_SCATT_I*N_GAMMAS_SCATT_F*N_SPINS;
  const unsigned int IN_DOF= N_SPINS;

  #pragma unroll
  for(int i_gs = 0 ; i_gs < OUT_DOF; i_gs++)
    fourier_transform_3D(block2+i_gs*IN_DOF*grid3D, accum+i_gs*IN_DOF, shared_cache, IN_DOF, sid3D, source_pos, moms, (OUT_DOF-1)*IN_DOF, -1, time_step, tid);

}

