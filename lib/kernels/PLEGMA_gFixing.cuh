#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_SU3_projection.cuh>
// Output is the gauge transformation
// Input is the gauge field
// This will be done in an iterative manner
template<typename Float>
static __global__ void gTransformLandau_kernel(Float* g, Float* A, Float *rnd,Float overelaxPar, int eo){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int xx[4]=GET_ID(sid);
  if (sid >= DGC_localVolume) return;
  su3_2<Float> Rg(g);
  gauge2<Float> RA(A);
  Float2<Float> gS[N_COLS][N_COLS] , AS[N_COLS][N_COLS], tmp[N_COLS][N_COLS];
  zero_G(gS);
#pragma unroll
  for(int mu = 0; mu < N_DIMS; mu++){
    RA.get(AS,mu,sid);
    G_plus_aG(gS,AS,1.);
    RA.get<Minus>(AS,mu,sid,mu);
    Gdag(AS);
    G_plus_aG(gS,AS,1.);
  }
  su3Projection(tmp,gS);
  if(rnd[sid] < overelaxPar){
    mul_G_G(gS,tmp,tmp);
    Gdag(gS);
  }
  else{
    Gdag(gS,tmp);
  }
  if( ((xx[0]+xx[1]+xx[2]+xx[3])&1) == eo)
    Rg.set(gS,sid);
}

template<typename Float>
static __global__ void gTransformMulALandau_kernel(Float* g, Float* A, int eo){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int xx[4]=GET_ID(sid);
  if (sid >= DGC_localVolume) return;
  int meo=(xx[0]+xx[1]+xx[2]+xx[3])&1;
  su3_2<Float> Rg(g);
  gauge2<Float> RA(A);
  Float2<Float> gS[N_COLS][N_COLS] , AS[N_COLS][N_COLS], tmp[N_COLS][N_COLS];
  if(meo==eo){
#pragma unroll
    for(int mu = 0 ; mu < N_DIMS; mu++){
      RA.get(AS,mu,sid);
      Rg.get(gS,sid);
      mul_G_G(tmp,gS,AS);
      RA.set(tmp,mu,sid);
    }
  }
  else{
#pragma unroll
    for(int mu = 0 ; mu < N_DIMS; mu++){
      RA.get(AS,mu,sid);
      Rg.get<Plus>(gS,sid,mu);
      mul_G_Gdag(tmp,AS,gS);
      RA.set(tmp,mu,sid);
    }
  }
}

template<typename Float>
static void gFixingLandau_k(PLEGMA_Gauge<Float> &u_gFixed, PLEGMA_Gauge<Float> &u,
			  Float overelaxPar, Float tolerance,
			  int maxIter, int seedOverRelax){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if(typeid(Float) != typeid(double)){
    PLEGMA_error("Gauge fixing does not support single precision");
  }
  Float prec=1.,tr,trold=0.0;
  int iter=0;
  PLEGMA_Su3field<Float> g;
  PLEGMA_Su3field<Float> tmp;
  PLEGMA_Field<Float> rnd(BOTH,SCALAR);
  rnd.randInit(seedOverRelax);
  rnd.random(Uniform); //create uniform distribution of random numbers between [0,1)
  u_gFixed.copy(u);
  while(prec>tolerance && iter<maxIter){
    for(int eo=0; eo < 2; eo++){
      u_gFixed.communicateGhost(4);u_gFixed.communicateGhost(5);u_gFixed.communicateGhost(6);u_gFixed.communicateGhost(7);
      gTransformLandau_kernel<Float><<<gridDim,blockDim>>>(g.D_elem(), u_gFixed.D_elem(), rnd.D_elem(),overelaxPar,eo);
      checkCudaError();
      g.communicateGhost(0);g.communicateGhost(1);g.communicateGhost(2);g.communicateGhost(3);
      gTransformMulALandau_kernel<Float><<<gridDim,blockDim>>>(g.D_elem(), u_gFixed.D_elem(),eo);
      checkCudaError();
      rnd.random(Uniform);
    }

    tr=0.;
    for(int mu = 0; mu < N_DIMS; mu++){
      tmp.absorbDir_device(u_gFixed,mu);
      tr+=tmp.sumRtraceU();
    }
    prec = abs(trold-tr)/abs(tr);
    if(HGC_verbosity>1) PLEGMA_printf("Landau Gauge Fixing iter=%d, prec=%+e and trace=%+e\n",iter,prec,tr);
    trold=tr;
    iter++; 
    
  }
  if(iter >= maxIter) PLEGMA_warning("The maximum number of iterations have been reached in the gauge fixing precision (%+e > %+e)\n", prec,tolerance);
}
