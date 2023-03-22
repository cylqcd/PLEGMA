#include "PLEGMA_kernel_utils.cuh"
#include "PLEGMA_SU3_projection.cuh"

template<typename Float>
static __global__ void gluonField_kernel(gauge2<Float> Rout, gauge2<Float> Rin){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid>Rout.volume()) return;
  
  Float2<Float> Gout[N_COLS][N_COLS], Gin[N_COLS][N_COLS];
  #pragma unroll
  for(int mu = 0; mu < N_DIMS; mu++){
    Rin.get(Gin,mu,sid);
    Float tr = (Gin[0][0].y + Gin[1][1].y + Gin[2][2].y)/3.;
    #pragma unroll
    for(int c1 = 0; c1 < N_COLS; c1++)
      #pragma unroll
      for(int c2 = 0; c2 < N_COLS; c2++){
	Gout[c1][c2].x = (Gin[c2][c1].y + Gin[c1][c2].y)/2.;
	Gout[c1][c2].y = (Gin[c2][c1].x - Gin[c1][c2].x)/2.;
      }
    Gout[0][0].x -= tr; Gout[1][1].x -= tr; Gout[2][2].x -= tr;
    Rout.set(Gout,mu,sid);
  }
}


template<typename Float>
static void gluonField_k(gauge2<Float> Rout, gauge2<Float> Rin){
  ProfileStruct ps(Rout.volume());
  tuneAndRun(ps, "gluonField_kernel", gluonField_kernel<Float>, Rout, Rin);
}

// Output is the gauge transformation
// Input is the gauge field
// This will be done in an iterative manner
template<typename Float>
static __global__ void gTransformLandau_kernel(su3_2<Float> Rg, gauge2<Float> RA, Float *rnd,Float overelaxPar, int eo){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int xx[4]=GET_ID(sid);
  if (sid >= Rg.volume()) return;

  Float2<Float> gS[N_COLS][N_COLS] , AS[N_COLS][N_COLS], tmp[N_COLS][N_COLS];
  zero_G(gS);
#pragma unroll
  for(int mu = 0; mu < N_DIMS; mu++){
    RA.get(AS,mu,sid);
    G_plus_aG(gS,AS,1.);
    RA.template get<Minus>(AS,mu,sid,mu);
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
static __global__ void gTransformMulALandau_kernel(su3_2<Float> Rg, gauge2<Float> RA, int eo){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int xx[4]=GET_ID(sid);
  if (sid >= Rg.volume()) return;
  int meo=(xx[0]+xx[1]+xx[2]+xx[3])&1;
  
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
      Rg.template get<Plus>(gS,sid,mu);
      mul_G_Gdag(tmp,AS,gS);
      RA.set(tmp,mu,sid);
    }
  }
}

template<typename Float>
static void gFixingLandau_k(PLEGMA_Gauge<Float> &u_gFixed, PLEGMA_Gauge<Float> &u,
			  Float overelaxPar, Float tolerance,
			  int maxIter, int seedOverRelax){
  assert(u_gFixed.checkVolume(u));
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (u_gFixed.Total_length() + blockDim.x -1)/blockDim.x , 1 , 1);
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
      u_gFixed.communicateGhost(-1, DIR_MINUS);
      gTransformLandau_kernel<Float><<<gridDim,blockDim>>>(toField2<su3_2>(g), toField2<gauge2>(u_gFixed), rnd.D_elem(),overelaxPar,eo);
      checkQudaError();
      g.communicateGhost(-1, DIR_PLUS);
      gTransformMulALandau_kernel<Float><<<gridDim,blockDim>>>(toField2<su3_2>(g), toField2<gauge2>(u_gFixed),eo);
      checkQudaError();
      rnd.random(Uniform);
    }

    tr=0.;
    for(int mu = 0; mu < N_DIMS; mu++){
      tmp.absorbDir_device(u_gFixed,mu);
      tr+=tmp.sumRtraceU();
    }
    prec = abs(trold-tr)/abs(tr);
    if(HGC.verbosity>1) PLEGMA_printf("Landau Gauge Fixing iter=%d, prec=%+e and trace=%+e\n",iter,prec,tr);
    trold=tr;
    iter++; 
    
  }
  if(iter >= maxIter) PLEGMA_warning("The maximum number of iterations have been reached in the gauge fixing precision (%+e > %+e)\n", prec,tolerance);
}
