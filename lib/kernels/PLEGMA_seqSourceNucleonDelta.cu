#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_projectors.cuh>
#include <PLEGMA_Vector.h>
#include <PLEGMA_seqSourceNucleon.cuh>


using namespace plegma;
template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2, int c_gamma>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
#ifdef PLEGMA_SCATTERING_CONTRACTIONS
  const Float2<float> (*gi)[4];

  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid >= vec.volume()) return;
  const Float2<float> (*pr)[8];
  const short int (*prInd)[8][2];
  if(particle == PROTON){
    pr = (Float2<float> (*)[8]) projTmP;
    prInd = projIndTmP;
  }
  else if(particle == NEUTRON){
    pr = (Float2<float> (*)[8]) projTmM;
    prInd = projIndTmM;    
  }
  else{
    printf("Error: You can use only PROTON or NEUTRON\n");
    asm("trap;"); 
  }

#ifdef PLEGMA_SCATTERING_CONTRACTIONS
  const Float2<float> (*prscatt);
  const short int (*prIndscatt)[2];
  prscatt = (Float2<float> (*))projScatt;
  prIndscatt = projIndScatt;
#endif
  
  const short (*gammas_i_Idx)[4][2];
  gi = (Float2<float> (*)[4]) plegma::gamma_scatt;
  gammas_i_Idx = gammaInd_scatt;

  const Float2<float> (*gf)[4];
  const short (*gammas_f_Idx)[4][2];
  gf = (Float2<float> (*)[4]) plegma::gamma_scatt;
  gammas_f_Idx = gammaInd_scatt;

  Float2<FloatA> P[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatB> P2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatC> spinor[N_SPINS][N_COLS];
  for(short mu = 0; mu < N_SPINS; mu++)
    for(short cc = 0; cc < N_COLS; cc++)
      spinor[mu][cc] = 0.;
  
  prop1.get(P,sid);
  if(isTwoPropDiff) prop2.get(P2, sid);

  #pragma unroll
  for(short cc1 = 0 ; cc1 < 6 ; cc1++){
    short c1 = eps[cc1][0];
    short c2 = eps[cc1][1];
    short c3 = eps[cc1][2];
    #pragma unroll
    for(short cc2 = 0 ; cc2 < 6 ; cc2++){
      short c1p = eps[cc2][0];
      short c2p = eps[cc2][1];
      short c3p = eps[cc2][2];
      if(c3p == c_c2){
        #pragma unroll
        for(short idx_source = 0; idx_source < N_SPINS ; idx_source++){
          #pragma unroll
          for (short idx_sink = 0 ; idx_sink < N_SPINS ; idx_sink++){
            short mu = gammas_f_Idx[gamma_sink][idx_sink][0];//sink
	    short nu = gammas_f_Idx[gamma_sink][idx_sink][1];//sink
	    short ku = gammas_i_Idx[gamma_source][idx_source][0];//source 
	    short lu = gammas_i_Idx[gamma_source][idx_source][1];//source
            if (proj<8){
              #pragma unroll
              for(short nz = 0; nz < 8; nz++){
	        int b = prInd[proj][nz][0];
	        int a = prInd[proj][nz][1];
                const auto gf_val = gf[gamma_sink][idx_sink];
                const auto gi_val = gi[gamma_source][idx_source];

                Float2<FloatC> factor(
                   static_cast<FloatC>(gf_val.x * gi_val.x - gf_val.y * gi_val.y),
                   static_cast<FloatC>(gf_val.x * gi_val.y + gf_val.y * gi_val.x)
                );
                factor=factor*-1*sgn_eps[cc1]*sgn_eps[cc2]*pr[proj][nz]; 

	        if(!isTwoPropDiff){
	          if(ku == c_nu){
                    #pragma unroll
                    for(short gu = 0 ; gu < 4 ; gu++){
                      if (c_gamma==0){
                        if ( nu == gu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][lu][c2][c2p];
                      }
                      else if (c_gamma==1){
                        if ( nu == gu ) spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[a][b][c2][c2p];
                      }
                      else if (c_gamma==2){
                        if ( a == gu )  spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[nu][b][c2][c2p];
                      }
                      else if (c_gamma==6){
	                if ( nu == gu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][lu][c2][c2p];
		        if ( nu == gu ) spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[a][b][c2][c2p];
                        if ( a == gu )  spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[nu][b][c2][c2p];
                      }
                      else if (c_gamma==7){
                        if ( nu == gu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][lu][c2][c2p];
                        if ( nu == gu ) spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[a][b][c2][c2p];
//                        if ( a == gu )  spinor[gu][c3] += factor * P[mu][ku][c1][c1p] * P[nu][b][c2][c2p];
                      }

                    }
	          }
	        }
	        else{
                  #pragma unroll
	          for(short gu = 0 ; gu < 4 ; gu++){
                    if (c_gamma==0){
                      if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[a][lu][c2][c2p];
                    }
                    else if (c_gamma==1){
                      if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                    }
                    else if (c_gamma==2){
                      if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                    }
                    else if (c_gamma==3){
                      if( a == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[mu][b][c2][c2p];
                    }
                    else if (c_gamma==4){
                      if( mu == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][b][c2][c2p];
                    }
                    else if (c_gamma==5){
                      if( mu == gu &&  b == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][lu][c2][c2p];
                    }
                    else if (c_gamma==6){
                      if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[a][lu][c2][c2p];
                      if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                      if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                      if( a == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[mu][b][c2][c2p];
                      if( mu == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][b][c2][c2p];   	
                      if( mu == gu &&  b == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][lu][c2][c2p];
	            }
                    else if (c_gamma==7){
                      if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[a][lu][c2][c2p];
                      if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                      if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                      if( a == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[mu][b][c2][c2p];

                    }
                  }
	        }
              }
            }
	    else{
              const auto gf_val = gf[gamma_sink][idx_sink];
              const auto gi_val = gi[gamma_source][idx_source];


              int b = prIndscatt[proj][0];
              int a = prIndscatt[proj][1];
              Float2<FloatC> factor(
                   static_cast<FloatC>(gf_val.x * gi_val.x - gf_val.y * gi_val.y),
                   static_cast<FloatC>(gf_val.x * gi_val.y + gf_val.y * gi_val.x)
              );
              factor=factor*-1*sgn_eps[cc1]*sgn_eps[cc2];
              factor=factor*prscatt[proj];
              if(!isTwoPropDiff){
                if(ku == c_nu){
                  for(short gu = 0 ; gu < N_SPINS ; gu++){
                    if (c_gamma==0){
                      if ( nu == gu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][lu][c2][c2p];
                    }
                    else if (c_gamma==1){
                      if ( nu == gu ) spinor[gu][c3] += factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
                    }
                    else if (c_gamma==2){
                      if ( a == gu )  spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[nu][b][c2][c2p];
                    }
                    else if (c_gamma==6){
                      if ( nu == gu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][lu][c2][c2p];
                      if ( nu == gu ) spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[a][b][c2][c2p];
                      if ( a == gu )  spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[nu][b][c2][c2p];
                    }
                    else if (c_gamma==7){
                      if ( nu == gu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][lu][c2][c2p];
                      if ( nu == gu ) spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[a][b][c2][c2p];
//                      if ( a == gu )  spinor[gu][c3] += factor * P[mu][lu][c1][c1p] * P[nu][b][c2][c2p];
                    }
                  }
                }
              }
              else{
	        #pragma unroll
                for(short gu = 0 ; gu < N_SPINS ; gu++){
                  if (c_gamma==0){
                    if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[a][lu][c2][c2p];
                  }
                  else if (c_gamma==1){
                    if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                  }
                  else if (c_gamma==2){
                    if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                  }
                  else if (c_gamma==3){
                    if( a == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[mu][b][c2][c2p];
                  }
                  else if (c_gamma==4){
                    if( mu == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][b][c2][c2p];
                  }
                  else if (c_gamma==5){
                    if( mu == gu &&  b == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][lu][c2][c2p];
                  }
                  else if (c_gamma==6){
                    if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[a][lu][c2][c2p];
                    if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                    if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                    if( a == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[mu][b][c2][c2p];
                    if( mu == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][b][c2][c2p];
                    if( mu == gu &&  b == c_nu ) spinor[gu][c3] += factor * P2[a][ku][c1][c1p] * P[nu][lu][c2][c2p];
                  }
                  else if (c_gamma==7){
                    if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[a][lu][c2][c2p];
                    if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                    if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                    if( a == gu && lu == c_nu ) spinor[gu][c3] += factor * P2[nu][ku][c1][c1p] * P[mu][b][c2][c2p];

                  }

                }
	      }
            }
          }
        }
      }
    }
  }
  vec.set(spinor, sid);
#endif
#endif
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_gamma){
  switch(c_gamma){
  //case(0):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,0>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  //case(1):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,1>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  //case(2):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,2>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  //case(3):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,3>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  //case(4):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,4>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  //case(5):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,5>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  case(6):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,6>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  //case(7):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,7>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle);break;
  }
}
template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_c2, int c_gamma){
  switch(c_c2){
  case(0):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,0>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_gamma);break;
  case(1):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,1>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_gamma);break;
  case(2):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,2>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_gamma);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma){
  switch(c_nu){
  case(0):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,0>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_c2, c_gamma);break;
  case(1):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,1>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_c2, c_gamma);break;
  case(2):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,2>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_c2, c_gamma);break;
  case(3):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,3>(vec,prop1,prop2,gamma_source, gamma_sink,proj,particle,c_c2, c_gamma);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB>
__global__ void contractNucleonDeltaSeqSource_kernel(vector2<FloatC> vec, propTex<FloatA> prop1, propTex<FloatB> prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, bool isTwoPropDiff, int c_nu, int c_c2, int c_gamma){
  if(isTwoPropDiff) contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,true>(vec,prop1,prop2,gamma_source,gamma_sink,proj,particle,c_nu,c_c2, c_gamma);
  else contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,false>(vec,prop1,prop2,gamma_source,gamma_sink,proj,particle,c_nu,c_c2, c_gamma);
}

template<typename FloatC, typename FloatA, typename FloatB>
static void contractNucleonDeltaSeqSource(vector2<FloatC> &vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, bool isTwoPropDiff, int c_nu, int c_c2, int c_gamma){
  ProfileStruct ps(vec.volume());
  tuneAndRun(ps,"contractNucleonDeltaSeqSource", contractNucleonDeltaSeqSource_kernel<FloatC,FloatA,FloatB>, vec, prop1, prop2, gamma_source, gamma_sink, proj, particle, isTwoPropDiff, c_nu, c_c2, c_gamma);
  checkQudaError();
}

template<typename FloatC, typename FloatA>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1,  GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  propTex<FloatA>& prop2 = prop1;
  contractNucleonDeltaSeqSource(vec, prop1,prop2, gamma_source, gamma_sink, proj,particle,  false, c_nu, c_c2, c_gamma);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  contractNucleonDeltaSeqSource(vec, prop1, prop2, gamma_source, gamma_sink, proj,particle,  true, c_nu, c_c2, c_gamma);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template void contractNucleonDeltaSeqSource<float,float>(vector2<float> vec, propTex<float>& prop1, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);
template void contractNucleonDeltaSeqSource<double,double>(vector2<double> vec, propTex<double>& prop1, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);


template void contractNucleonDeltaSeqSource<float,float,float>(vector2<float> vec, propTex<float>& prop1, propTex<float>& prop2, GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink,  WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);
template void contractNucleonDeltaSeqSource<double,double,double>(vector2<double> vec, propTex<double>& prop1, propTex<double>& prop2,  GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);
