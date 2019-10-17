#include <cublas_v2.h>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_Thrust.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_Random.h>
#include <PLEGMA_Fmunu.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_Gauge.h>
using namespace plegma;

template<typename FloatOut,typename FloatIn>
static __global__ void cast_kernel(FloatOut *out, FloatIn *in, size_t size){
  
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;

  for (; sid < size; sid += gridDim.x * blockDim.x)
    out[sid] = (FloatIn) in[sid];
}

template<typename FloatOut,typename FloatIn>
static void cudaCast(FloatOut *out, FloatIn *in, size_t size){
  ProfileStruct ps(HGC_localVolume); // here we can actually use any size
  tuneAndRun(ps, "cast_kernel", cast_kernel<FloatOut,FloatIn>, (FloatOut*) out, (FloatIn*) in, size);
  checkCudaError();
}


template<typename FloatInOut>
static __global__ void copy_side_to_ghost_kernel(FloatInOut *f, int dir, int sign, int length_field){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_surface3D[dir]) return;
  generic2<FloatInOut> F(f);
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir) {
      id[i] = sign==1 ? (DGC_localL[dir]-1):0;
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  sidStride ss;
  ss.sid = DGC_sideGhost[dir + N_DIMS*sign]*length_field + sid;
  ss.stride = DGC_surface3D[dir];
  for(int i = 0 ; i < length_field ; i++)
    F.set(i, ss, F.get(i,vid));
}

template<typename Float>
static void copy_side_to_ghost(PLEGMA_Field<Float> &f, int dir){
  if( HGC_dimBreak[dir%N_DIMS] ){
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (HGC_surface3D[dir%N_DIMS] + blockDim.x -1)/blockDim.x , 1 , 1);
    copy_side_to_ghost_kernel<<<gridDim,blockDim>>>(f.D_elem(), dir%N_DIMS, dir/N_DIMS, f.Field_length());
    checkCudaError();
  }
}

template<typename FloatInOut>
static __global__ void copy_corner_to_ghost_kernel(FloatInOut *f, int dir1, int dir2, int sign1, int sign2, int length_field){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_surface2D[dir1][dir2]) return;
  generic2<FloatInOut> F(f);
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir1) {
      id[i] = sign1==1 ? (DGC_localL[dir1]-1):0;
    } else if(i==dir2) {
      id[i] = sign2==1 ? (DGC_localL[dir2]-1):0;      
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  sidStride ss;
  ss.sid = DGC_cornerGhost[dir1 + N_DIMS*sign1][dir2 + N_DIMS*sign2]*length_field + sid;
  ss.stride = DGC_surface2D[dir1][dir2];
  for(int i = 0 ; i < length_field ; i++)
    F.set(i, ss, F.get(i,vid));
}

template<typename Float>
static void copy_corner_to_ghost(PLEGMA_Field<Float> &f, int dir1, int dir2){
  if( (dir1%N_DIMS != dir2%N_DIMS ) && HGC_dimBreak[dir1%N_DIMS] && HGC_dimBreak[dir2%N_DIMS] ){
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (HGC_surface2D[dir1%N_DIMS][dir2%N_DIMS] + blockDim.x -1)/blockDim.x , 1 , 1);
    copy_corner_to_ghost_kernel<<<gridDim,blockDim>>>(f.D_elem(), dir1%N_DIMS, dir2%N_DIMS, dir1/N_DIMS, dir2/N_DIMS, f.Field_length());
    checkCudaError();
  }
}

/**
  @brief CUDA kernel to generate random number from the CURAND RNG states
  @param state CURAND RNG state array, which is used to calculate Z_n noise

 */
template<int n>
__inline__ __device__ Float2<float> rootsunity(int order){
  Float2<float> root;
  root.x = cos(2.0*M_PI*((float)order)/(float)n);
  root.y = sin(2.0*M_PI*((float)order)/(float)n);
  return root;
}
template<>
__inline__ __device__ Float2<float> rootsunity<2>(int order){
  Float2<float> root;
  if(order == 0){
    root.x = 1.0;
    root.y = 0.0;
  }
  else{
    root.x = -1.0;
    root.y = 0.0;
  }
  return root;
}

template<>
__inline__ __device__ Float2<float> rootsunity<4>(int order){
  Float2<float> root;
  if(order == 0){
    root.x = 1.0/sqrt(2.0);
    root.y = 1.0/sqrt(2.0);
  }
  else if(order == 1){
    root.x = -1.0/sqrt(2.0);
    root.y = 1.0/sqrt(2.0);
  }
  else if(order == 2){
    root.x = -1.0/sqrt(2.0);
    root.y = -1.0/sqrt(2.0);
  }
  else{
    root.x = 1.0/sqrt(2.0);
    root.y = -1.0/sqrt(2.0);
  }
  return root;
}

template<typename Float, int n>
__global__ void genStochasticUniform_kernel(cuRNGState *state, int length_field, Float *inout){

  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if( n < 2) return;

  for( int i = 0; i < length_field; ++i){

    Float tmp = PLEGMA_Random<Float, Uniform>(state[sid]);

    for( int order = 0; order <= n-1; ++order){

      if( tmp  < ((Float)order+1.0)/(Float)n ){

        inout2[sid + i*(DGC_localVolume)] = rootsunity<n>(order);
        break;
      }
    }
  }
} 

template<typename Float, int n>
void set_stochastic( PLEGMA_RNG &rng_state, PLEGMA_Field<Float> &inOut, int field_deg_free, int rng_size){

  dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim( (rng_size  + blockDim.x -1)/blockDim.x , 1 , 1);
  genStochasticUniform_kernel<Float, n><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem());
}
template<typename Float>
__global__ void genRandomUniform_kernel(cuRNGState *state, int length_field, Float *inout){
  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  for( int i = 0; i < length_field; ++i){
    inout2[sid] = PLEGMA_Random<Float, Uniform>(state[sid]);
  }
} 

template<typename Float>
__global__ void genRandomNormal_kernel(cuRNGState *state, int length_field, Float *inout){
  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  for( int i = 0; i < length_field; ++i){
    inout2[sid] = PLEGMA_Random<Float, Normal>(state[sid]);
  }
} 

template<typename Float>
void set_random( PLEGMA_RNG &rng_state, PLEGMA_Field<Float> &inOut, int field_deg_free, int rng_size, DIST sampling){

  dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim( (rng_size + blockDim.x -1)/blockDim.x , 1 , 1);
  if(sampling == Uniform)
    genRandomUniform_kernel<Float><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem());
  else if( sampling == Normal)
    genRandomNormal_kernel<Float><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem());
  else
    PLEGMA_error("The given distribution is not defined.\n");
}

template<typename Float>
struct HadCol{
  int ih;
  __device__ HadCol(int ih):ih(ih){}
  __device__ int  HadamardElements(int i, int j){
    int sum=0;
    for(int k = 0 ; k < 32 ; k++){
      sum += (i%2)*(j%2);
      i=i>>1;
      j=j>>1;
    }
    if( (sum%2) ==0 )
      return 1;
    else
      return -1;
  }
  template<typename Tuple>
  __device__ void operator()(Tuple t){
    int color = thrust::get<0>(t);
    int signHad = HadamardElements(color,ih);
    Float2<Float> &el = (thrust::get<1>(t));
    el.x = signHad*el.x;
    el.y = signHad*el.y;
  }
};

template<typename Float>
static void apply_hprob_coloring_4D(Float* d_elems, int *d_colors, int ih){
  // make sure before that is not a 3D field
  int V = HGC_localVolume;
  thrust::device_ptr<int> th_c(d_colors);
  thrust::device_ptr<Float2<Float> > th_e((Float2<Float>*)d_elems);
  typedef thrust::tuple<thrust::device_ptr<int>, thrust::device_ptr<Float2<Float> > > tplDIntDFl2;
  typedef thrust::zip_iterator<tplDIntDFl2> zipTplDIntDFl2;
  zipTplDIntDFl2 z1 = thrust::make_zip_iterator(thrust::make_tuple(th_c,th_e));
  zipTplDIntDFl2 z2 = thrust::make_zip_iterator(thrust::make_tuple(th_c+V,th_e+V));
  thrust::for_each(z1,z2,HadCol<Float>(ih));
}

template<typename Float, typename FloatA, typename FloatB, typename FloatC, typename FloatD>
static __global__ void traceMulFmunuSu3FmunuSu3_kernel(Float *F, FloatA *A, FloatB *B, FloatC *C, FloatD *D){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> *F2 = (Float2<Float> *) F;
  if (sid >= DGC_localVolume) return;
  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatB> lB[N_COLS][N_COLS];
  Float2<FloatC> lC[N_COLS][N_COLS];
  Float2<FloatD> lD[N_COLS][N_COLS];
  su3_2<FloatA> RA(A);
  su3_2<FloatB> RB(B);
  su3_2<FloatC> RC(C);
  su3_2<FloatD> RD(D);
  RA.get(lA,sid);
  RB.get(lB,sid);
  RC.get(lC,sid);
  RD.get(lD,sid);
  Float2<Float> res = trace_mul_G_G_G_G<FloatA,FloatB,FloatC,FloatD>(lA,lB,lC,lD);
  F2[sid] = res;
}

template<typename Float, typename FloatA, typename FloatB, typename FloatC,typename FloatD>
static void traceMulFmunuSu3FmunuSu3_k(PLEGMA_Field<Float> &F, PLEGMA_Fmunu<FloatA> &A, std::pair<int,int> munu_l,
				       PLEGMA_Su3field<FloatB> &B, PLEGMA_Fmunu<FloatC> &C,  std::pair<int,int> munu_r,
				       PLEGMA_Su3field<FloatD> &D){
  ProfileStruct ps(HGC_localVolume);
  long int lshift = ((long int) A.munuToIndx(munu_l)) * N_COLS * N_COLS * HGC_localVolume * 2;
  long int rshift = ((long int) C.munuToIndx(munu_r)) * N_COLS * N_COLS * HGC_localVolume * 2;
  tuneAndRun(ps, "traceMulFmunuSu3FmunuSu3_kernel", traceMulFmunuSu3FmunuSu3_kernel<Float,FloatA,FloatB,FloatC,FloatD>,
	     F.D_elem(), A.D_elem()+lshift, B.D_elem(),C.D_elem()+rshift, D.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB>
static void __global__ trPmunu_kernel(FloatA *out, FloatB *gauge, int mu, int nu){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;
  Float2<FloatA> *out2 = (Float2<FloatA> *) out;
  gauge2<FloatA> u(gauge);
  Float2<FloatB> U1[N_COLS][N_COLS], U2[N_COLS][N_COLS], U3[N_COLS][N_COLS];
    /**
      --<-- 
     |     |
     v     ^
    x|-->--|
   **/
  // U_\mu(x) * U_\nu(x+\mu) * U^dag_\mu(x+nu) * U^\dag_\nu(x)
  u.get(U1,mu,sid); u.get<Plus>(U2,nu,sid,mu); mul_G_G(U3,U1,U2);
  u.get<Plus>(U2,mu,sid,nu); mul_G_Gdag(U1,U3,U2);
  u.get(U2,nu,sid); mul_G_Gdag(U3,U1,U2);
  out2[sid]= U3[0][0] + U3[1][1] + U3[2][2];
}

template<typename FloatA, typename FloatB>
static void trPmunu_k(PLEGMA_Field<FloatA> &f,PLEGMA_Gauge<FloatB> &gauge, std::pair<int,int> munu){
  ProfileStruct ps(HGC_localVolume);
  if(std::get<0>(munu) == std::get<1>(munu)) PLEGMA_error("For Pmunu cannot have mu == nu");
  tuneAndRun(ps,"trPmunu_kernel",trPmunu_kernel<FloatA,FloatB>,f.D_elem(),gauge.D_elem(),std::get<0>(munu),std::get<1>(munu));
  checkCudaError();
}

template<typename FloatOut, typename FloatIn>
static __global__ void sumModVector_kernel(FloatOut *out, FloatIn *in){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  vector2<FloatIn> vec(in);
  Float2<FloatOut> *Fout = (Float2<FloatOut> *) out;

  Float2<FloatIn> Sin[N_SPINS][N_COLS];
  FloatOut res=0.0;
    
  if (sid >= DGC_localVolume) return;

  vec.get(Sin,sid);
  
  #pragma unroll
  for(int i=0; i<N_SPINS; i++){
    #pragma unroll
    for(int j=0; j<N_COLS; j++){
      res += Sin[i][j].x*Sin[i][j].x + Sin[i][j].y*Sin[i][j].y;
    }
  }
  
  Fout[sid] = res;
}

template<typename FloatOut, typename FloatIn>
static void sumModVector_k(PLEGMA_Field<FloatOut> &Fo, PLEGMA_Vector<FloatIn> &Vi){
  ProfileStruct ps(HGC_localVolume);
  tuneAndRun(ps,"sumModVector_kernel", summod_kernel<FloatOut,FloatIn>, Fo.D_elem(), Vi.D_elem());
  checkCudaError();
}
