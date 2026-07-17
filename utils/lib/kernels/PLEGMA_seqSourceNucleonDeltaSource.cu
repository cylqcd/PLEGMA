#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_projectors.cuh>
#include <PLEGMA_Vector.h>
#include <PLEGMA_seqSourceNucleon.cuh>

//NtoN_indices should be the results of a multiplication of Cg5 at the source and Cg5 at the sink
//Pr result of the muliplication with the projector

#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
static const __device__ short int NtoDelta_indices[4][16][4]={0,0,0,1,//Cgx*Cg5
                                                              0,0,1,0,
                                                              0,0,2,3,
                                                              0,0,3,2,
                                                              1,1,0,1,
                                                              1,1,1,0,
                                                              1,1,2,3,
                                                              1,1,3,2,
                                                              2,2,0,1,
                                                              2,2,1,0,
                                                              2,2,2,3,
                                                              2,2,3,2,
                                                              3,3,0,1,
                                                              3,3,1,0,
                                                              3,3,2,3,
                                                              3,3,3,2,
                                                              0,0,0,1,//Cgy*Cg5
                                                              0,0,1,0,
                                                              0,0,2,3,
                                                              0,0,3,2,
                                                              1,1,0,1,
                                                              1,1,1,0,
                                                              1,1,2,3,
                                                              1,1,3,2,
                                                              2,2,0,1,
                                                              2,2,1,0,
                                                              2,2,2,3,
                                                              2,2,3,2,
                                                              3,3,0,1,
                                                              3,3,1,0,
                                                              3,3,2,3,
                                                              3,3,3,2,
                                                              0,1,0,1,//Cgz*Cg5
                                                              0,1,1,0,
                                                              0,1,2,3,
                                                              0,1,3,2,
                                                              1,0,0,1,
                                                              1,0,1,0,
                                                              1,0,2,3,
                                                              1,0,3,2,
                                                              2,3,0,1,
                                                              2,3,1,0,
                                                              2,3,2,3,
                                                              2,3,3,2,
                                                              3,2,0,1,
                                                              3,2,1,0,
                                                              3,2,2,3,
                                                              3,2,3,2,
                                                              0,3,0,1,//Cgz*Cg5
                                                              0,3,1,0,
                                                              0,3,2,3,
                                                              0,3,3,2,
                                                              1,2,0,1,
                                                              1,2,1,0,
                                                              1,2,2,3,
                                                              1,2,3,2,
                                                              2,1,0,1,
                                                              2,1,1,0,
                                                              2,1,2,3,
                                                              2,1,3,2,
                                                              3,0,0,1,
                                                              3,0,1,0,
                                                              3,0,2,3,
                                                              3,0,3,2};

  



static const __device__ float NtoDelta_values[4][16][2] =     {{{0,1},{0,-1},{0,1},{0,-1},//Cgx*Cg5
                                                            {0,-1},{0,1},{0,-1},{0,1},
                                                            {0,-1},{0,1},{0,-1},{0,1},
                                                            {0,1},{0,-1},{0,1},{0,-1}},
                                                           {{-1,0},{1,0},{-1,0},{1,0},//Cgy*Cg5
                                                            {-1,0},{1,0},{-1,0},{1,0},
                                                            {1,0},{-1,0},{1,0},{-1,0},
                                                            {1,0},{-1,0},{1,0},{-1,0}},
                                                           {{0,-1},{0,1},{0,-1},{0,1},//Cgz*Cg5
                                                            {0,-1},{0,1},{0,-1},{0,1},
                                                            {0,1},{0,-1},{0,1},{0,-1},
                                                            {0,1},{0,-1},{0,1},{0,-1}},
                                                           {{0,-1},{0,1},{0,-1},{0,1},//Cgt*Cg5
                                                            {0,-1},{0,1},{0,-1},{0,1},
                                                            {0,1},{0,-1},{0,1},{0,-1},
                                                            {0,1},{0,-1},{0,1},{0,-1}}};

#endif

using namespace plegma;
template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2, int c_gamma>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
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
      if(c3p == c_c2)
          #pragma unroll
	  for(short idx = 0 ; idx < 16 ; idx++){
	    short mu = NtoDelta_indices[c_gamma][idx][0];
	    short nu = NtoDelta_indices[c_gamma][idx][1];
	    short ku = NtoDelta_indices[c_gamma][idx][2];
	    short lu = NtoDelta_indices[c_gamma][idx][3];
            if (proj<8){
            #pragma unroll
            for(short nz = 0; nz < 8; nz++){
	      int b = prInd[proj][nz][0];
	      int a = prInd[proj][nz][1];
	      Float2<FloatC> factor(-1*sgn_eps[cc1]*sgn_eps[cc2]*NtoDelta_values[c_gamma][idx][0],-1*sgn_eps[cc1]*sgn_eps[cc2]*NtoDelta_values[c_gamma][idx][1]);
              factor=factor*pr[proj][nz];
	      if(!isTwoPropDiff){
               for(short gu = 0 ; gu < 4 ; gu++){
	            if ( mu == gu &&  lu == c_nu ) spinor[gu][c3] += factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];
		    if ( mu == gu &&  lu == c_nu ) spinor[gu][c3] += factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
                    if ( nu == gu &&  b  == c_nu )  spinor[gu][c3] += factor * P[a][ku][c1][c1p] * P[mu][lu][c2][c2p];
                 	        }
	      }
	      else{
#pragma unroll
	        for(short gu = 0 ; gu < 4 ; gu++){
                  if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][ku][c2][c2p];
                  if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                  if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                  if( a == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][b][c2][c2p];


                  if( a  == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][b][c1][c1p] * P[mu][lu][c2][c2p];   	
                  if( mu == gu &&  lu == c_nu ) spinor[gu][c3] += factor * P2[nu][b][c1][c1p] * P[a][ku][c2][c2p];
	        }
	      }
            }
            }
	    else{
#ifdef PLEGMA_SCATTERING_CONTRACTIONS
            int b = prIndscatt[proj][0];
            int a = prIndscatt[proj][1];
            Float2<FloatC> factor ((-1)*sgn_eps[cc1]*sgn_eps[cc2]*NtoDelta_values[c_gamma][idx][0],(-1)*sgn_eps[cc1]*sgn_eps[cc2]*NtoDelta_values[c_gamma][idx][0]);
            factor=factor*prscatt[proj];
            if(!isTwoPropDiff){
              if(lu == c_nu){
                spinor[nu][c3] += factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];
                spinor[nu][c3] += factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
                spinor[nu][c3] += factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];

              }
            }
            else
#pragma unroll
              for(short gu = 0 ; gu < 4 ; gu++){
                if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][ku][c2][c2p];
                if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                if( a == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][b][c2][c2p];
              }
#endif
	    } 

      }}}
  vec.set(spinor, sid);
#endif
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_gamma){
  switch(c_gamma){
  case(0):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,0>(vec,prop1,prop2,proj,particle);break;
  case(1):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,1>(vec,prop1,prop2,proj,particle);break;
  case(2):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,2>(vec,prop1,prop2,proj,particle);break;
  case(3):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,c_c2,3>(vec,prop1,prop2,proj,particle);break;
  }
}
template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_c2, int c_gamma){
  switch(c_c2){
  case(0):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,0>(vec,prop1,prop2,proj,particle,c_gamma);break;
  case(1):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,1>(vec,prop1,prop2,proj,particle,c_gamma);break;
  case(2):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,2>(vec,prop1,prop2,proj,particle,c_gamma);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff>
__device__ void contractNucleonDeltaSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma){
  switch(c_nu){
  case(0):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,0>(vec,prop1,prop2,proj,particle,c_c2, c_gamma);break;
  case(1):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,1>(vec,prop1,prop2,proj,particle,c_c2, c_gamma);break;
  case(2):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,2>(vec,prop1,prop2,proj,particle,c_c2, c_gamma);break;
  case(3):contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,3>(vec,prop1,prop2,proj,particle,c_c2, c_gamma);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB>
__global__ void contractNucleonDeltaSeqSource_kernel(vector2<FloatC> vec, propTex<FloatA> prop1, propTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, bool isTwoPropDiff, int c_nu, int c_c2, int c_gamma){
  if(isTwoPropDiff) contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,true>(vec,prop1,prop2,proj,particle,c_nu,c_c2, c_gamma);
  else contractNucleonDeltaSeqSource<FloatC,FloatA,FloatB,false>(vec,prop1,prop2,proj,particle,c_nu,c_c2, c_gamma);
}

template<typename FloatC, typename FloatA, typename FloatB>
static void contractNucleonDeltaSeqSource(vector2<FloatC> &vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, bool isTwoPropDiff, int c_nu, int c_c2, int c_gamma){
  ProfileStruct ps(vec.volume());
  tuneAndRun(ps,"contractNucleonDeltaSeqSource", contractNucleonDeltaSeqSource_kernel<FloatC,FloatA,FloatB>, vec, prop1, prop2, proj, particle, isTwoPropDiff, c_nu, c_c2, c_gamma);
  checkQudaError();
}

template<typename FloatC, typename FloatA>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  propTex<FloatA>& prop2 = prop1;
  contractNucleonDeltaSeqSource(vec, prop1,prop2, proj,particle,  false, c_nu, c_c2, c_gamma);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  contractNucleonDeltaSeqSource(vec, prop1, prop2, proj,particle,  true, c_nu, c_c2, c_gamma);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template void contractNucleonDeltaSeqSource<float,float>(vector2<float> vec, propTex<float>& prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);
template void contractNucleonDeltaSeqSource<double,double>(vector2<double> vec, propTex<double>& prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);


template void contractNucleonDeltaSeqSource<float,float,float>(vector2<float> vec, propTex<float>& prop1, propTex<float>& prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);
template void contractNucleonDeltaSeqSource<double,double,double>(vector2<double> vec, propTex<double>& prop1, propTex<double>& prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_gamma);
