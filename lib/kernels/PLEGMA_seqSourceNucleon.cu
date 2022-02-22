#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_projectors.cuh>
#include <PLEGMA_Vector.h>
#include <PLEGMA_seqSourceNucleon.cuh>

#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
static const __device__ short int NtoN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float NtoN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};
#endif

using namespace plegma;
template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2>
__device__ void contractNucleonSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle){
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

  const Float2<float> (*prscatt);
  const short int (*prIndscatt)[2];
  prscatt = (Float2<float> (*))projScatt;
  prIndscatt = projIndScatt;


  Float2<FloatA> P[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatB> P2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatC> spinor[N_SPINS][N_COLS];
  for(short mu = 0; mu < N_SPINS; mu++)
    for(short cc = 0; cc < N_COLS; cc++)
      spinor[mu][cc] = 0.;
  
  prop1.get(P,sid);
  if(isTwoPropDiff) prop2.get(P2, sid);

  //#pragma unroll
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
//        #pragma unroll
	for(short idx = 0 ; idx < 16 ; idx++){
	  short mu = NtoN_indices[idx][0];
	  short nu = NtoN_indices[idx][1];
	  short ku = NtoN_indices[idx][2];
	  short lu = NtoN_indices[idx][3];
          if (proj<8){
//          #pragma unroll
          for(short nz = 0; nz < 8; nz++){
	    int b = prInd[proj][nz][0];
	    int a = prInd[proj][nz][1];
	    Float2<FloatC> factor = (-1)*sgn_eps[cc1]*sgn_eps[cc2]*NtoN_values[idx]*pr[proj][nz];
	    if(!isTwoPropDiff){
	      if(lu == c_nu){
	        spinor[nu][c3] += factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];
		spinor[nu][c3] += factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
	      }
	    }
	    else
//#pragma unroll
	      for(short gu = 0 ; gu < 4 ; gu++){
                if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][ku][c2][c2p];
                if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                if( a == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][b][c2][c2p];
	      }
	  }
	  }
	  else{
          int b = prIndscatt[proj][0];
          int a = prIndscatt[proj][1];
          Float2<FloatC> factor = (-1)*sgn_eps[cc1]*sgn_eps[cc2]*NtoN_values[idx]*prscatt[proj];
          if(!isTwoPropDiff){
            if(lu == c_nu){
              spinor[nu][c3] += factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];
              spinor[nu][c3] += factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
            }
          }
          else
//#pragma unroll
            for(short gu = 0 ; gu < 4 ; gu++){
              if( mu == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][ku][c2][c2p];
              if( mu == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
              if( a == gu && b == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
              if( a == gu && ku == c_nu ) spinor[gu][c3] += factor * P2[nu][lu][c1][c1p] * P[mu][b][c2][c2p];
            }
	  } 

      }}}
  vec.set(spinor, sid);
#endif
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu>
__device__ void contractNucleonSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_c2){
  switch(c_c2){
  case(0):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,0>(vec,prop1,prop2,proj,particle);break;
  case(1):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,1>(vec,prop1,prop2,proj,particle);break;
  case(2):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,2>(vec,prop1,prop2,proj,particle);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff>
__device__ void contractNucleonSeqSource(vector2<FloatC>& vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2){
  switch(c_nu){
  case(0):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,0>(vec,prop1,prop2,proj,particle,c_c2);break;
  case(1):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,1>(vec,prop1,prop2,proj,particle,c_c2);break;
  case(2):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,2>(vec,prop1,prop2,proj,particle,c_c2);break;
  case(3):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,3>(vec,prop1,prop2,proj,particle,c_c2);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB>
__global__ void contractNucleonSeqSource_kernel(vector2<FloatC> vec, propTex<FloatA> prop1, propTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, bool isTwoPropDiff, int c_nu, int c_c2){
  if(isTwoPropDiff) contractNucleonSeqSource<FloatC,FloatA,FloatB,true>(vec,prop1,prop2,proj,particle,c_nu,c_c2);
  else contractNucleonSeqSource<FloatC,FloatA,FloatB,false>(vec,prop1,prop2,proj,particle,c_nu,c_c2);
}

template<typename FloatC, typename FloatA, typename FloatB>
static void contractNucleonSeqSource(vector2<FloatC> &vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, bool isTwoPropDiff, int c_nu, int c_c2){
  ProfileStruct ps(vec.volume());
  tuneAndRun(ps,"contractNucleonSeqSource", contractNucleonSeqSource_kernel<FloatC,FloatA,FloatB>, vec, prop1, prop2, proj, particle, isTwoPropDiff, c_nu, c_c2);
  checkCudaError();
}

template<typename FloatC, typename FloatA>
void contractNucleonSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  propTex<FloatA>& prop2 = prop1;
  contractNucleonSeqSource(vec, prop1,prop2, proj,particle,  false, c_nu, c_c2);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  contractNucleonSeqSource(vec, prop1, prop2, proj,particle,  true, c_nu, c_c2);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template void contractNucleonSeqSource<float,float>(vector2<float> vec, propTex<float>& prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);
template void contractNucleonSeqSource<double,double>(vector2<double> vec, propTex<double>& prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);


template void contractNucleonSeqSource<float,float,float>(vector2<float> vec, propTex<float>& prop1, propTex<float>& prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);
template void contractNucleonSeqSource<double,double,double>(vector2<double> vec, propTex<double>& prop1, propTex<double>& prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);
