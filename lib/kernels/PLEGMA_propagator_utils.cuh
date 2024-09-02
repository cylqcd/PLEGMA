#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<LEFTRIGHT LF,typename Float>
static __global__ void apply_gamma_prop_kernel(prop2<Float> prop, GAMMAS r){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> Sin[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<Float> Sout[N_SPINS][N_SPINS][N_COLS][N_COLS];
  if (sid >= prop.volume()) return;
  prop.get(Sin,sid);
  gammaProp<LF>(Sout,Sin,r);
  prop.set(Sout,sid);
}

template<typename Float>
static void apply_gamma_prop(LEFTRIGHT LR, PLEGMA_Propagator<Float>& InOut, GAMMAS r){
  auto prop = toField2<prop2>(InOut);
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (prop.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  switch(LR){
  case(LEFT):
    apply_gamma_prop_kernel<LEFT><<<gridDim,blockDim>>>(prop, r);
    break;
  case(RIGHT):
    apply_gamma_prop_kernel<RIGHT><<<gridDim,blockDim>>>(prop, r);
    break;
  }
  checkCudaError();
}


template<typename Float>
static __global__ void apply_gamma5_propagator_kernel(prop2<Float> prop){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= prop.volume()) return;

  prop.setSid(sid);
  #pragma unroll
  for(int nu = 0 ; nu < N_SPINS ; nu++)
    #pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++)
      #pragma unroll
      for(int c2 = 0 ; c2 < N_COLS ; c2++) {
	Float2<Float> spinor[4];
	// inline shuffling
        #pragma unroll
	for(int mu = 0 ; mu < N_SPINS ; mu++)
	  spinor[(mu+2)%4] = prop.get(mu, nu, c1, c2);
	// replacing
        #pragma unroll
	for(int mu = 0 ; mu < N_SPINS ; mu++)
	  prop.set(mu, nu, c1, c2, spinor[mu]);
      }
}

template<typename Float>
void apply_gamma5_propagator(PLEGMA_Propagator<Float>& inOut){
  auto prop = toField2<prop2>(inOut);
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (prop.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_gamma5_propagator_kernel<<<gridDim,blockDim>>>(prop);
}

template<typename Float>
static __global__ void apply_boundaries_kernel(Float *inOut, int t0){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;
  int t = (sid/DGC_localL[0]/DGC_localL[1]/DGC_localL[2]) % DGC_localL[3];
  t += DGC_procPosition[3] * DGC_localL[3];

  if( t < t0 ) {
#pragma unroll
    for(int i = 0 ; i < N_SPINS*N_SPINS*N_COLS*N_COLS ; i++) {
      inOut[(i*DGC_localVolume + sid)*2 + 0] *= -1.;
      inOut[(i*DGC_localVolume + sid)*2 + 1] *= -1.;
    }
  }
}

template<typename Float>
void apply_boundaries(Float *inOut, int t0){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_boundaries_kernel<<<gridDim,blockDim>>>(inOut,t0);
  checkCudaError();
}


template<typename Float>
static __global__ void rotateToPhysicalBase_kernel(Float *inOut, int sign){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  Float2<Float> P[4][4], PT[4][4], sign_imag_unit(sign*1., IMAG),
    *inOut2 = (Float2<Float> *) inOut;

  #pragma unroll
  for(int c1 = 0 ; c1 < N_COLS ; c1++)
    #pragma unroll
    for(int c2 = 0 ; c2 < N_COLS ; c2++){

      // copying
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS; nu++)
	  P[mu][nu] = inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*DGC_localVolume + sid];

      // shuffling
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS; nu++)
	  PT[mu][nu] = 0.5 * ( P[mu][nu] - P[(mu+2)%4][(nu+2)%4] + sign_imag_unit * ( P[mu][(nu+2)%4] + P[(mu+2)%4][nu] ));

      // replacing
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS; nu++)
	  inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*DGC_localVolume + sid] = PT[mu][nu];
    }

}

template<typename Float>
void rotateToPhysicalBase(Float* inOut, int sign){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  rotateToPhysicalBase_kernel<Float><<<gridDim,blockDim>>>((Float*) inOut,sign);
  checkCudaError();
}

template<typename Float>
struct Spin { Float2<Float> vals[N_SPINS]; };

template<typename Float>
__global__ void build_exact_propagator_kernel( prop2<Float> prop,
					       vectorTex<Float> texVec,
					       Spin<Float> spin,
					       Float2<Float> eval){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= prop.volume()) return;
  
  Float2<Float> inOut[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<Float> vec[N_SPINS][N_COLS];
  prop.get(inOut,sid);
  texVec.get(vec,sid);
  
  #pragma unroll
  for(int alpha = 0 ; alpha < N_SPINS ; alpha++)
    #pragma unroll
    for(int beta = 0 ; beta < N_SPINS ; beta++)
      #pragma unroll
      for(int b = 0 ; b < N_COLS ; b++) {
	Float2<Float> val = vec[alpha][b] * spin.vals[beta];
	inOut[alpha][beta][b][0] += val/eval;
	inOut[alpha][beta][b][1] += val/conj(eval);
      }
  prop.set(inOut,sid);
}

template<typename Float>
void build_exact_propagator(PLEGMA_Propagator<Float>& out, Float *spinVals, Float *eigVals, Float* eigVecs, int nvecs, size_t vec_size, bool dev_ptr){
  auto prop = toField2<prop2>(out);

  ProfileStruct ps(HGC_localVolume);

  const int ils = dev_ptr ? 1 : 4;
  std::vector<PLEGMA_Vector<Float>*> vec;
  size_t vec_bytes = vec_size*sizeof(Float);
  cudaStream_t stream[ils];
  for(int k=0; k<ils; k++) {
    vec.push_back(new PLEGMA_Vector<Float>(dev_ptr ? NONE : DEVICE));
    cudaStreamCreate(stream+k);
    if(dev_ptr) {
      vec[k]->D_elem(eigVecs+k*vec_size);
    } else {
      cudaMemcpyAsync(vec[k]->D_elem(), eigVecs+k*vec_size, vec_bytes, cudaMemcpyHostToDevice, stream[k]);
    }
  }

  Spin<Float> spin;
  
  for(int i=0; i<nvecs; i++) {
    for(int mu=0; mu<4; mu++) {
      spin.vals[mu] = {spinVals[mu*nvecs*2+i*2+0], spinVals[mu*nvecs*2+i*2+1]};
    }
    Float2<Float> eval = {eigVals[i*2+0], eigVals[i*2+1]};
    int k=i%ils;
    if(not dev_ptr) {
      cudaStreamSynchronize(stream[k]);
      checkCudaError();
    }
    auto vecT = toTexture<vectorTex>(*vec[k]);
    if(i==0) {
      auto kernel = tuner(ps, "build_exact_propagator_kernel",  build_exact_propagator_kernel<Float>,
			  prop, *vecT, spin, eval);
      if(not kernel->tuned()) {
	kernel->tune();
      }
      out.zero_device();      
    }
    run( ps, "build_exact_propagator_kernel", build_exact_propagator_kernel<Float>,
	 prop, *vecT, spin, eval);
    
    // Start copying next vector to use
    if(i+ils<nvecs) {
      if(dev_ptr) {
	vec[k]->D_elem(eigVecs+(i+ils)*vec_size);
      } else {
	cudaMemcpyAsync(vec[k]->D_elem(), eigVecs+(i+ils)*vec_size, vec_bytes, cudaMemcpyHostToDevice, stream[k]);
      }
    }
  }
  
  for(int k=0; k<ils; k++) {
    delete vec[k];
    cudaStreamDestroy(stream[k]);
  }
}
