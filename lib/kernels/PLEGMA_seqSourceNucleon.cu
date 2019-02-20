#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_projectors.cuh>

#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
static const __device__ short int NtoN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float NtoN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};
#endif

using namespace plegma;
template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu, int c_c2>
__device__ void contractNucleonSeqSource(FloatC* vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t space_stride = DGC_localVolume3D;
  sidStride ss(sid,space_stride);
  if(sid >= space_stride) return;
  Float2<FloatC> *vec2 = (Float2<FloatC> *) vec;
  
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

  Float2<FloatA> P[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatB> P2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatA> *pP = (Float2<FloatA> *) P;
  Float2<FloatB> *pP2 = (Float2<FloatB> *) P2;
  Float2<FloatC> spinor[N_SPINS][N_COLS];
  for(short mu = 0; mu < N_SPINS; mu++)
    for(short cc = 0; cc < N_COLS; cc++)
      spinor[mu][cc] = 0.;
  
  prop1.get(pP,N_SPINS*N_SPINS*N_COLS*N_COLS,ss);
  if(isTwoPropDiff) prop2.get(pP2, N_SPINS*N_SPINS*N_COLS*N_COLS,ss);

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
	  short mu = NtoN_indices[idx][0];
	  short nu = NtoN_indices[idx][1];
	  short ku = NtoN_indices[idx][2];
	  short lu = NtoN_indices[idx][3];
#pragma unroll
	  for(short nz = 0; nz < 8; nz++){
	    int b = prInd[proj][nz][0];
	    int a = prInd[proj][nz][1];
	    Float2<FloatC> factor = (-1)*sgn_eps[cc1]*sgn_eps[cc2]*NtoN_values[idx]*pr[proj][nz];
	    if(!isTwoPropDiff){
	      if(lu == c_nu){
		spinor[nu][c3] = spinor[nu][c3] + factor * P[mu][b][c1][c1p] * P[a][ku][c2][c2p];
		spinor[nu][c3] = spinor[nu][c3] + factor * P[mu][ku][c1][c1p] * P[a][b][c2][c2p];
	      }
	    }
	    else
#pragma unroll
	      for(short gu = 0 ; gu < 4 ; gu++){
                if( mu == gu && b == c_nu ) spinor[gu][c3] = spinor[gu][c3] + factor * P2[nu][lu][c1][c1p] * P[a][ku][c2][c2p];
                if( mu == gu && ku == c_nu ) spinor[gu][c3] = spinor[gu][c3] + factor * P2[nu][lu][c1][c1p] * P[a][b][c2][c2p];
                if( a == gu && b == c_nu ) spinor[gu][c3] = spinor[gu][c3] + factor * P2[nu][lu][c1][c1p] * P[mu][ku][c2][c2p];
                if( a == gu && ku == c_nu ) spinor[gu][c3] = spinor[gu][c3] + factor * P2[nu][lu][c1][c1p] * P[mu][b][c2][c2p];
	      }
	  }   
	}}}
#pragma unroll
  for(short mu = 0 ; mu < 4 ; mu++)
#pragma unroll
    for(short ic = 0 ; ic < 3 ; ic++)
      vec2[(mu*N_COLS + ic)*DGC_localVolume + timeslice*space_stride + sid] = spinor[mu][ic];
#endif
}


template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff, int c_nu>
__device__ void contractNucleonSeqSource(FloatC* vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_c2){
  switch(c_c2){
  case(0):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,0>(vec,prop1,prop2,proj,particle,timeslice);break;
  case(1):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,1>(vec,prop1,prop2,proj,particle,timeslice);break;
  case(2):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,c_nu,2>(vec,prop1,prop2,proj,particle,timeslice);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB, bool isTwoPropDiff>
__device__ void contractNucleonSeqSource(FloatC* vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2){
  switch(c_nu){
  case(0):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,0>(vec,prop1,prop2,proj,particle,timeslice,c_c2);break;
  case(1):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,1>(vec,prop1,prop2,proj,particle,timeslice,c_c2);break;
  case(2):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,2>(vec,prop1,prop2,proj,particle,timeslice,c_c2);break;
  case(3):contractNucleonSeqSource<FloatC,FloatA,FloatB,isTwoPropDiff,3>(vec,prop1,prop2,proj,particle,timeslice,c_c2);break;
  }
}

template<typename FloatC, typename FloatA, typename FloatB>
__global__ void contractNucleonSeqSource_kernel(FloatC* vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, bool isTwoPropDiff, int c_nu, int c_c2){
  if(isTwoPropDiff) contractNucleonSeqSource<FloatC,FloatA,FloatB,true>(vec,prop1,prop2,proj,particle,timeslice,c_nu,c_c2);
  else contractNucleonSeqSource<FloatC,FloatA,FloatB,false>(vec,prop1,prop2,proj,particle,timeslice,c_nu,c_c2);
}

template<typename FloatC, typename FloatA, typename FloatB>
static void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, bool isTwoPropDiff, int c_nu, int c_c2){
  int SpVol = HGC_localVolume/HGC_localL[3];
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume
  contractNucleonSeqSource_kernel<<<gridDim,blockDim>>>(vec.D_elem(), prop1, prop2, proj, particle, timeslice, isTwoPropDiff, c_nu, c_c2);
  checkCudaError();
}

template<typename FloatC, typename FloatA>
void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1,WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  genericTex<FloatA> prop2 = prop1;
  contractNucleonSeqSource(vec, prop1,prop2, proj,particle, timeslice, false, c_nu, c_c2);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  contractNucleonSeqSource(vec, prop1, prop2, proj,particle, timeslice, true, c_nu, c_c2);
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK");
#endif
}

template void contractNucleonSeqSource<float,float>(PLEGMA_Vector<float> &vec, genericTex<float> prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
template void contractNucleonSeqSource<double,double>(PLEGMA_Vector<double> &vec, genericTex<double> prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);


template void contractNucleonSeqSource<float,float,float>(PLEGMA_Vector<float> &vec, genericTex<float> prop1, genericTex<float> prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
template void contractNucleonSeqSource<double,double,double>(PLEGMA_Vector<double> &vec, genericTex<double> prop1, genericTex<double> prop2,WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
