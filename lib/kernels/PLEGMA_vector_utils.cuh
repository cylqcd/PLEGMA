#include <PLEGMA_BLAS.h>
#include <PLEGMA_Random.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas.cuh>
#include <PLEGMA_kernel_tuner.cuh>
#include <PLEGMA_Thrust.h>
using namespace plegma;
using namespace quda;

template<LEFTRIGHT LF,typename Float>
static __global__ void apply_gamma_vector_kernel(Float *inOut, GAMMAS r){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  vector2<Float> vec(inOut);
  Float2<Float> Sin[N_SPINS][N_COLS];
  Float2<Float> Sout[N_SPINS][N_COLS]; 
  if (sid >= DGC_localVolume) return;
  vec.get(Sin,sid);
  gammaV<LF>(Sout,Sin,r);
  vec.set(Sout,sid);
}

template<typename Float>
static void apply_gamma_vector(LEFTRIGHT LR,Float *inOut,GAMMAS r){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  switch(LR){
  case(LEFT):
    apply_gamma_vector_kernel<LEFT><<<gridDim,blockDim>>>((Float*) inOut, r);
    break;
  case(RIGHT):
    apply_gamma_vector_kernel<RIGHT><<<gridDim,blockDim>>>((Float*) inOut, r);
    break;
  }
  checkCudaError();
}

template<LEFTRIGHT LF,typename Float>
static __global__ void apply_gamma_scatt_vector_kernel(Float *inOut, GAMMAS_SCATT r){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  vector2<Float> vec(inOut);
  Float2<Float> Sin[N_SPINS][N_COLS];
  Float2<Float> Sout[N_SPINS][N_COLS];
  if (sid >= DGC_localVolume) return;
  vec.get(Sin,sid);
  gamma_scattV<LF>(Sout,Sin,r);
  vec.set(Sout,sid);
}

template<typename Float>
static void apply_gamma_scatt_vector(LEFTRIGHT LR,Float *inOut,GAMMAS_SCATT r){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  switch(LR){
  case(LEFT):
    apply_gamma_scatt_vector_kernel<LEFT><<<gridDim,blockDim>>>((Float*) inOut, r);
    break;
  case(RIGHT):
    apply_gamma_scatt_vector_kernel<RIGHT><<<gridDim,blockDim>>>((Float*) inOut, r);
    break;
  }
  checkCudaError();
}

template<typename Float>
static __global__ void apply_gamma5_vector_kernel(Float *inOut){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> *inOut2 = (Float2<Float> *) inOut;
  if (sid >= DGC_localVolume) return;
    
  #pragma unroll
  for(int c1 = 0 ; c1 < N_COLS ; c1++){
    Float2<Float> spinor[4];
    // inline shuffling
    #pragma unroll
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      spinor[(mu+2)%4] = inOut2[(mu*N_COLS+c1)*DGC_localVolume + sid];
    // replacing
    #pragma unroll
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      inOut2[(mu*N_COLS+c1)*DGC_localVolume + sid] = spinor[mu];
  }

}

template<typename Float>
void apply_gamma5_vector(Float *inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_gamma5_vector_kernel<<<gridDim,blockDim>>>(inOut);
}



  
template<typename Float>
static __global__ void conjugate_vector_kernel(Float *inOut){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  #pragma unroll
  for(int i = 0 ; i < N_SPINS*N_COLS ; i++)
    inOut[(i*DGC_localVolume + sid)*2 + 1] *= -1.;
}

template<typename Float>
void conjugate_vector(Float *inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  conjugate_vector_kernel<<<gridDim,blockDim>>>(inOut);
  checkCudaError();
}

template<typename Float>
__inline__ __global__ void scale_vector_kernel(Float a, Float* inOut){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  #pragma unroll
  for(int i = 0 ; i < N_SPINS*N_COLS ; i++) {
    inOut[(i*DGC_localVolume + sid)*2 + 0] *= a;
    inOut[(i*DGC_localVolume + sid)*2 + 1] *= a;
  }
}

template<typename Float>
void scale_vector(Float a, Float* inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  scale_vector_kernel<<<gridDim,blockDim>>>( a, inOut);
  checkCudaError();
}

template<typename Float>
void norm2_device(Float norm, Float* in){

}

template<typename FloatIn, typename FloatOut, bool outEvenB, bool outOddB> 
static __global__ void copy_to_QUDA(FloatIn *in, FloatOut *outEven, FloatOut *outOdd){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume/2) return;

  // take indices on 4d lattice
  int half_stride = DGC_localVolume/2;
  int latt_coord = 2*sid;

  int r1,r2,x_id,y_id,z_id,t_id;
  r1 = latt_coord/(DGC_localL[0]);
  r2 = r1/(DGC_localL[1]);
  x_id = latt_coord - r1*(DGC_localL[0]);
  y_id = r1 - r2*(DGC_localL[1]);
  t_id = r2/(DGC_localL[2]);
  z_id = r2 - t_id*(DGC_localL[2]);
  int evenSiteBit = ((x_id+y_id+z_id+t_id) & 1);
  int oddSiteBit  = evenSiteBit ^ 1;
  Float2<FloatOut> *outEven2 = (Float2<FloatOut> *) outEven;
  Float2<FloatOut> *outOdd2 = (Float2<FloatOut> *) outOdd;
  Float2<FloatIn> *in2 = (Float2<FloatIn> *) in;

  #pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++) {
    #pragma unroll
    for(int ic = 0 ; ic < N_COLS ; ic++){
      if(outEvenB) {
	outEven2[(mu*N_COLS + ic)*half_stride + sid] =
	  in2[(mu*N_COLS + ic)*DGC_localVolume + latt_coord + evenSiteBit];
      } else
	outEven2[(mu*N_COLS + ic)*half_stride + sid] = 0.;

      if(outOddB) {
	outOdd2[(mu*N_COLS + ic)*half_stride + sid] =
	  in2[(mu*N_COLS + ic)*DGC_localVolume + latt_coord + oddSiteBit];
      } else
	outOdd2[(mu*N_COLS + ic)*half_stride + sid] = 0.;
    }
  }
}

template<typename FloatIn, typename FloatOut> 
static void copy_to_QUDA(FloatIn* in,ColorSpinorField &qudaVec, bool isEven){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if( qudaVec.SiteSubset() == QUDA_PARITY_SITE_SUBSET ){
    if( isEven )
      copy_to_QUDA<FloatIn,FloatOut,true,false><<<gridDim,blockDim>>>(in,(FloatOut*) qudaVec.V(), NULL);
    else
      copy_to_QUDA<FloatIn,FloatOut,false,true><<<gridDim,blockDim>>>(in, NULL,(FloatOut*) qudaVec.V());
  } else
    copy_to_QUDA<FloatIn,FloatOut,true,true><<<gridDim,blockDim>>>(in, (FloatOut*) qudaVec.Even().V(),(FloatOut*) qudaVec.Odd().V());
}

template<typename FloatIn> 
static void copy_to_QUDA(FloatIn* in, ColorSpinorField &qudaVec, bool isEven){
  if( qudaVec.Precision() == QUDA_SINGLE_PRECISION )
    copy_to_QUDA<FloatIn,float>(in, qudaVec, isEven);
  else if ( qudaVec.Precision() == QUDA_DOUBLE_PRECISION )
    copy_to_QUDA<FloatIn,double>(in, qudaVec, isEven);
  else
    PLEGMA_error("Precision %d not supported", qudaVec.Precision());

  checkCudaError();
}

template<typename FloatOut, typename FloatIn, bool inEvenB, bool inOddB> 
static __global__ void copy_from_QUDA_kernel(FloatOut *out, FloatIn *inEven, FloatIn *inOdd){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume/2) return;

  int half_stride = DGC_localVolume/2;
  int latt_coord = 2*sid;

  int r1,r2,x_id,y_id,z_id,t_id;
  r1 = latt_coord/(DGC_localL[0]);
  r2 = r1/(DGC_localL[1]);
  x_id = latt_coord - r1*(DGC_localL[0]);
  y_id = r1 - r2*(DGC_localL[1]);
  t_id = r2/(DGC_localL[2]);
  z_id = r2 - t_id*(DGC_localL[2]);
  int evenSiteBit = ((x_id+y_id+z_id+t_id) & 1);
  int oddSiteBit  = evenSiteBit ^ 1;
  Float2<FloatIn> *inEven2 = (Float2<FloatIn> *) inEven;
  Float2<FloatIn> *inOdd2 = (Float2<FloatIn> *) inOdd;
  Float2<FloatOut> *out2 = (Float2<FloatOut> *) out;

  #pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++) {
    #pragma unroll
    for(int ic = 0 ; ic < N_COLS ; ic++) {
      if(inEvenB) {
	out2[(mu*N_COLS + ic)*DGC_localVolume + latt_coord + evenSiteBit] =
	  inEven2[(mu*N_COLS + ic)*half_stride + sid];
      } else
	out2[(mu*N_COLS + ic)*DGC_localVolume + latt_coord + evenSiteBit] = 0.;

      if(inOddB) {
	out2[(mu*N_COLS + ic)*DGC_localVolume + latt_coord + oddSiteBit] =
	  inOdd2[(mu*N_COLS + ic)*half_stride + sid];
      } else
	out2[(mu*N_COLS + ic)*DGC_localVolume + latt_coord + oddSiteBit] = 0.;
    }
  }
}

template<typename FloatOut, typename FloatIn> 
static void copy_from_QUDA(FloatOut* out, ColorSpinorField &qudaVec, bool isEven){
  ProfileStruct ps(HGC_localVolume);
  if( qudaVec.SiteSubset() == QUDA_PARITY_SITE_SUBSET ){
    if( isEven )
      tuneAndRun(ps, "copy_from_QUDA_kernel", copy_from_QUDA_kernel<FloatOut,FloatIn,true,false>,out,(FloatIn*) qudaVec.V(), (FloatIn*) NULL);
    else
      tuneAndRun(ps, "copy_from_QUDA_kernel", copy_from_QUDA_kernel<FloatOut,FloatIn,false,true>, out, (FloatIn*) NULL,(FloatIn*) qudaVec.V());
  } else
    tuneAndRun(ps, "copy_from_QUDA_kernel", copy_from_QUDA_kernel<FloatOut,FloatIn,true,true>, out, (FloatIn*) qudaVec.Even().V(),(FloatIn*) qudaVec.Odd().V());
}

template<typename FloatOut> 
static void copy_from_QUDA(FloatOut* out, ColorSpinorField &qudaVec, bool isEven){
  if( qudaVec.Precision() == QUDA_SINGLE_PRECISION )
    copy_from_QUDA<FloatOut,float>(out, qudaVec, isEven);
  else if ( qudaVec.Precision() == QUDA_DOUBLE_PRECISION )
    copy_from_QUDA<FloatOut,double>(out, qudaVec, isEven);
  else
    PLEGMA_error("Precision %d not supported", qudaVec.Precision());
  
  checkCudaError();
}

template<typename Float>
struct computeRMS{
  int s[3];
  int Nr2;
  int *list_comp_R2;
  Float *output;
  computeRMS(int source_x, int source_y, int source_z, int Nr2_size, int *listR2, Float *psiSq){
    s[0] = source_x; s[1] = source_y;  s[2] = source_z;
    Nr2=Nr2_size;
    list_comp_R2 = listR2;
    output = psiSq;
  }  
  inline __device__ int getIndexElem(int val, int *list, int size){
    int res = -1;
    for(int i = 0 ; i < size; i++)
      if(val == list[i]){
	res = i;
	break;
      }
    return res;
  }
  template<typename Tuple>
  __device__ void operator()(Tuple t){
    int id = thrust::get<0>(t);
    int x[3] = GET_ID_ZYX(id);
#pragma unroll
    for(int i = 0 ; i < 3 ; i++){
      x[i] += DGC_procPosition[i] * DGC_localL[i];
      x[i] = x[i] >= s[i] ? x[i]-s[i] : s[i] - x[i];
      if(x[i] > DGC_totalL[i]/2) x[i] = DGC_totalL[i] - x[i];
    }
    int r2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
    int index_r2 = getIndexElem(r2,list_comp_R2,Nr2);
    if(index_r2 < 0) return;
    Float2<Float> e[N_SPINS*N_COLS];
    Float2<Float> *w = &(thrust::get<1>(t));
#pragma unroll
    for(int i = 0 ; i < N_SPINS*N_COLS; i++) e[i] = *(w+i*DGC_localVolume);
    Float val =0;
#pragma unroll
    for(int i = 0 ; i < N_SPINS*N_COLS ; i++) val += e[i].x*e[i].x  + e[i].y*e[i].y; 
    atomicAdd(&output[index_r2],val);
  }
};

template<typename Float>
static void compute_rms(PLEGMA_Vector<Float> &vec, std::vector<int> &listR2, std::vector<Float> &absPsi, int my_it, int *sourceposition){
  int *d_listR2 = nullptr;
  Float *d_absPsi = nullptr;
  if(listR2.size() != absPsi.size()) PLEGMA_error("List sizes should match");
  cudaMalloc((void**)&d_listR2, listR2.size() * sizeof(int)); checkCudaError();
  cudaMalloc((void**)&d_absPsi, absPsi.size() * sizeof(Float)); checkCudaError();
  cudaMemcpy(d_listR2,listR2.data(), listR2.size() * sizeof(int), cudaMemcpyHostToDevice); checkCudaError();
  cudaMemset(d_absPsi,0,absPsi.size() * sizeof(Float)); checkCudaError();
  int V = HGC_localVolume;
  int V3 = V/HGC_localL[3];
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + V3;
  typedef thrust::device_ptr<Float2<Float> > DpF2;
  DpF2 y( (Float2<Float>*) (vec.D_elem() + my_it*V3*2) );
  typedef thrust::tuple<thrust::counting_iterator<int>,DpF2> tplIntDev2;
  typedef thrust::zip_iterator<tplIntDev2> zipTplIntDev2;
  zipTplIntDev2 z1 = thrust::make_zip_iterator(thrust::make_tuple(first,y));
  zipTplIntDev2 z2 = thrust::make_zip_iterator(thrust::make_tuple(last,y+V3));
  thrust::for_each(z1,z2,computeRMS<Float>(sourceposition[0],sourceposition[1],sourceposition[2],listR2.size(),d_listR2,d_absPsi));
  cudaMemcpy(absPsi.data(), d_absPsi, absPsi.size() * sizeof(Float), cudaMemcpyDeviceToHost); checkCudaError();
  cudaFree(d_listR2);
  cudaFree(d_absPsi);
}

template<typename FloatVo, typename FloatS, typename FloatVi>
static __global__ void mulGV_kernel(FloatVo *Vo, FloatS *u, FloatVi *Vi){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;
  Float2<FloatVo> lVo[N_SPINS][N_COLS];
  Float2<FloatVi> lVi[N_SPINS][N_COLS];
  Float2<FloatS> lu[N_COLS][N_COLS];

  vector2<FloatVo> RVo(Vo);
  vector2<FloatVi> RVi(Vi);
  su3_2<FloatS> Ru(u);

  RVi.get(lVi,sid);
  Ru.get(lu,sid);
  mul_G_V(lVo,lu,lVi);
  RVo.set(lVo,sid);
}

template<typename FloatVo, typename FloatS, typename FloatVi>
static void mulGV_k(PLEGMA_Vector<FloatVo> &Vo, PLEGMA_Su3field<FloatS> &u, PLEGMA_Vector<FloatVi> &Vi){
  ProfileStruct ps(HGC_localVolume);
  tuneAndRun(ps,"mulGV_kernel", mulGV_kernel<FloatVo,FloatS,FloatVi>, Vo.D_elem(), u.D_elem(), Vi.D_elem());
  checkCudaError();
}
