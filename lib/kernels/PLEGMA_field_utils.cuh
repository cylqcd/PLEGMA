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
static __global__ void cast_kernel(pFloat2<FloatOut> out, pFloat2<FloatIn> in){
  
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if(sid>=out.volume()) return;

  in.setSid(sid);
  out.setSid(sid);
  for (int i = 0; i < out.site_size; i++)
    out.set(i, in.get(i));
}

template<typename FloatOut,typename FloatIn>
static void cudaCast(pFloat2<FloatOut> out, pFloat2<FloatIn> in){
  ProfileStruct ps(out.volume()); // here we can actually use any size
  tuneAndRun(ps, "cast_kernel_size_"+std::to_string(out.site_size), cast_kernel<FloatOut,FloatIn>, out, in);
  checkQudaError();
}



template<typename FloatInOut, int side>
static __global__ void copy_side_to_ghost_kernel(pFloat2<FloatInOut> F, short dir, short sign){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= F.sideGhostL(dir)) return;

  size_t id[4], tmp_sid=sid;
  #pragma unroll
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir) {
      id[i] = sign==DIR_MINUS ? (DGC_localL[dir]-side):(side-1);
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  F.setSid(vid);
  pFloat2<FloatInOut> F_ghost=F;
  if(side == 1) F_ghost.accessSideGhost(LEXIC_3D(dir,id), dir, (ORIENTATION) sign);
  if(side == 2) F_ghost.accessSecondSideGhost(LEXIC_3D(dir,id), dir, (ORIENTATION) sign);
  if(side == 3) F_ghost.accessThirdSideGhost(LEXIC_3D(dir,id), dir, (ORIENTATION) sign);
  for(int i = 0 ; i < F.site_size ; i++)
    F_ghost.set(i, F.get(i));
}

template<typename Float>
static void copy_side_to_ghost(pFloat2<Float> F, short dir, short sign, GHOST_FLAG ghost_flag){
  if( HGC_dimBreak[dir] ){
    if(ghost_flag==FIRST_SIDE){
      ProfileStruct ps(F.sideGhostL(dir));
      tuneAndRun(ps, "copy_side_to_ghost_kernel_size_"+std::to_string(F.site_size), copy_side_to_ghost_kernel<Float,1>, F, dir, sign);
    }
    if(ghost_flag==SECOND_SIDE){
      ProfileStruct ps(F.sideGhostL(dir));
      tuneAndRun(ps, "copy_secondside_to_ghost_kernel_size_"+std::to_string(F.site_size), copy_side_to_ghost_kernel<Float,2>, F, dir, sign);
    }
    if(ghost_flag==THIRD_SIDE){
      ProfileStruct ps(F.sideGhostL(dir));
      tuneAndRun(ps, "copy_thirdside_to_ghost_kernel_size_"+std::to_string(F.site_size), copy_side_to_ghost_kernel<Float,3>, F, dir, sign);
    }
  }
}

template<typename FloatInOut, int corner>
static __global__ void copy_corner_to_ghost_kernel(pFloat2<FloatInOut> F, short dir1, short dir2, short sign1, short sign2){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= F.cornerGhostL(dir1,dir2)) return;
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir1) {
      id[i] = sign1==DIR_MINUS ? (DGC_localL[dir1]-corner):(corner-1);
    } else if(i==dir2) {
      id[i] = sign2==DIR_MINUS ? (DGC_localL[dir2]-1):0;      
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  F.setSid(vid);
  pFloat2<FloatInOut> F_ghost=F;
  if(corner == 1) F_ghost.accessCornerGhost(LEXIC_2D(dir1,dir2,id), dir1, dir2, (ORIENTATION) sign1, (ORIENTATION) sign2);
  if(corner == 2) F_ghost.accessSecondCornerGhost(LEXIC_2D(dir1,dir2,id), dir1, dir2, (ORIENTATION) sign1, (ORIENTATION) sign2);
  for(int i = 0 ; i < F.site_size ; i++)
    F_ghost.set(i, F.get(i));
}

template<typename Float>
static void copy_corner_to_ghost(pFloat2<Float> F, short dir1, short dir2, short sign1, short sign2, GHOST_FLAG ghost_flag){
  if( (dir1 != dir2 ) && HGC_dimBreak[dir1] && HGC_dimBreak[dir2] ){
    if(ghost_flag == FIRST_CORNER){
      ProfileStruct ps(F.cornerGhostL(dir1, dir2));
      tuneAndRun(ps, "copy_corner_to_ghost_kernel_size_"+std::to_string(F.site_size), copy_corner_to_ghost_kernel<Float,1>, F, dir1, dir2, sign1, sign2);
    }
    if(ghost_flag == SECOND_CORNER){
      ProfileStruct ps(F.cornerGhostL(dir1, dir2));
      tuneAndRun(ps, "copy_secondcorner_to_ghost_kernel_size_"+std::to_string(F.site_size), copy_corner_to_ghost_kernel<Float,2>, F, dir1, dir2, sign1, sign2);
    }
  }
}

template<typename FloatInOut>
static __global__ void copy_vertex_to_ghost_kernel(pFloat2<FloatInOut> F, short dir1, short dir2, short dir3, short sign1, short sign2, short sign3){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= F.vertexGhostL(dir1,dir2,dir3)) return;
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir1) {
      id[i] = sign1==DIR_MINUS ? (DGC_localL[dir1]-1):0;
    } else if(i==dir2) {
      id[i] = sign2==DIR_MINUS ? (DGC_localL[dir2]-1):0;      
    } else if(i==dir3) {
      id[i] = sign3==DIR_MINUS ? (DGC_localL[dir3]-1):0;      
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  F.setSid(vid);
  pFloat2<FloatInOut> F_ghost=F;
  F_ghost.accessVertexGhost(LEXIC_1D(dir1,dir2,dir3,id), dir1, dir2, dir3, (ORIENTATION) sign1, (ORIENTATION) sign2, (ORIENTATION) sign3);
  for(int i = 0 ; i < F.site_size ; i++)
    F_ghost.set(i, F.get(i));
}

template<typename Float>
static void copy_vertex_to_ghost(pFloat2<Float> F, short dir1, short dir2, short dir3, short sign1, short sign2, short sign3){
  if( (dir1 != dir2 && dir1 != dir3 && dir3 != dir2 ) && HGC_dimBreak[dir1] && HGC_dimBreak[dir2] && HGC_dimBreak[dir3] ){
    ProfileStruct ps(F.vertexGhostL(dir1, dir2, dir3));
    tuneAndRun(ps, "copy_vertex_to_ghost_kernel_size_"+std::to_string(F.site_size), copy_vertex_to_ghost_kernel<Float>, F, dir1, dir2, dir3, sign1, sign2, sign3);
  }
}

template<typename Float, int side>
static __global__ void copy_side_to_ghost_ext_kernel(pFloat2<Float> F, Float* F_ext, short dir, short sign){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= F.sideGhostL(dir)) return;
  
  size_t id[4], tmp_sid=sid;
  #pragma unroll
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir) {
      id[i] = sign==DIR_MINUS ? (DGC_localL[dir]-side) : (side-1);
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  F.setSid(vid);
  pFloat2<Float> F_ghost=F;
  F_ghost.p = (Float2<Float>*) F_ext;
  if(side == 1) F_ghost.accessSideGhost(LEXIC_3D(dir,id), dir, (ORIENTATION) sign, true);
  if(side == 2) F_ghost.accessSecondSideGhost(LEXIC_3D(dir,id), dir, (ORIENTATION) sign, true);
  if(side == 3) F_ghost.accessThirdSideGhost(LEXIC_3D(dir,id), dir, (ORIENTATION) sign, true);
  for(int i = 0 ; i < F.site_size ; i++)
    F_ghost.set(i, F.get(i));
}

template<typename Float>
static void copy_side_to_ghost_ext(pFloat2<Float> F, Float* F_ext, short dir, short sign, GHOST_FLAG ghost_flag){
  if( HGC_dimBreak[dir] ){
    if(ghost_flag == FIRST_SIDE){
      ProfileStruct ps(F.sideGhostL(dir));
      tuneAndRun(ps, "copy_side_to_ghost_ext_kernel_size_"+std::to_string(F.site_size), copy_side_to_ghost_ext_kernel<Float,1>, F, F_ext, dir, sign);
    }
    if(ghost_flag == SECOND_SIDE){
      ProfileStruct ps(F.sideGhostL(dir));
      tuneAndRun(ps, "copy_secondside_to_ghost_ext_kernel_size_"+std::to_string(F.site_size), copy_side_to_ghost_ext_kernel<Float,2>, F, F_ext, dir, sign);
    }
    if(ghost_flag == THIRD_SIDE){
      ProfileStruct ps(F.sideGhostL(dir));
      tuneAndRun(ps, "copy_thirdside_to_ghost_ext_kernel_size_"+std::to_string(F.site_size), copy_side_to_ghost_ext_kernel<Float,3>, F, F_ext, dir, sign);
    }
  }
}

template<typename Float, int corner>
static __global__ void copy_corner_to_ghost_ext_kernel(pFloat2<Float> F, Float* F_ext, short dir1, short dir2, short sign1, short sign2){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= F.cornerGhostL(dir1,dir2)) return;
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir1) {
      id[i] = sign1==DIR_MINUS ? (DGC_localL[dir1]-corner):(corner-1);
    } else if(i==dir2) {
      id[i] = sign2==DIR_MINUS ? (DGC_localL[dir2]-1):0;      
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  F.setSid(vid);
  pFloat2<Float> F_ghost=F;
  F_ghost.p = (Float2<Float>*) F_ext;
  if(corner == 1) F_ghost.accessCornerGhost(LEXIC_2D(dir1,dir2,id), dir1, dir2, (ORIENTATION) sign1, (ORIENTATION) sign2, true);
  if(corner == 2) F_ghost.accessSecondCornerGhost(LEXIC_2D(dir1,dir2,id), dir1, dir2, (ORIENTATION) sign1, (ORIENTATION) sign2, true);
  for(int i = 0 ; i < F.site_size ; i++)
    F_ghost.set(i, F.get(i));
}

template<typename Float>
static void copy_corner_to_ghost_ext(pFloat2<Float> F, Float* F_ext, short dir1, short dir2, short sign1, short sign2, GHOST_FLAG ghost_flag){
  if( (dir1 != dir2 ) && HGC_dimBreak[dir1] && HGC_dimBreak[dir2] ){
    if(ghost_flag == FIRST_CORNER){
      ProfileStruct ps(F.cornerGhostL(dir1, dir2));
      tuneAndRun(ps, "copy_corner_to_ghost_ext_kernel_size_"+std::to_string(F.site_size), copy_corner_to_ghost_ext_kernel<Float,1>, F, F_ext, dir1, dir2, sign1, sign2);
    }
    if(ghost_flag == SECOND_CORNER){
      ProfileStruct ps(F.cornerGhostL(dir1, dir2));
      tuneAndRun(ps, "copy_secondcorner_to_ghost_ext_kernel_size_"+std::to_string(F.site_size), copy_corner_to_ghost_ext_kernel<Float,2>, F, F_ext, dir1, dir2, sign1, sign2);
    }
  }
}

template<typename Float>
static __global__ void copy_vertex_to_ghost_ext_kernel(pFloat2<Float> F, Float* F_ext, short dir1, short dir2, short dir3, short sign1, short sign2, short sign3){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= F.vertexGhostL(dir1,dir2,dir3)) return;
  size_t id[4], tmp_sid=sid;
  for(int i = 0 ; i<N_DIMS; i++) {
    if(i==dir1) {
      id[i] = sign1==DIR_MINUS ? (DGC_localL[dir1]-1):0;
    } else if(i==dir2) {
      id[i] = sign2==DIR_MINUS ? (DGC_localL[dir2]-1):0;      
    } else if(i==dir3) {
      id[i] = sign3==DIR_MINUS ? (DGC_localL[dir3]-1):0;      
    } else {
      id[i] = tmp_sid % DGC_localL[i];
      tmp_sid /= DGC_localL[i];
    }
  }
  size_t vid = LEXIC_ID(id);
  F.setSid(vid);
  pFloat2<Float> F_ghost=F;
  F_ghost.p = (Float2<Float>*) F_ext;
  F_ghost.accessVertexGhost(LEXIC_1D(dir1,dir2,dir3,id), dir1, dir2, dir3, (ORIENTATION) sign1, (ORIENTATION) sign2, (ORIENTATION) sign3, true);
  for(int i = 0 ; i < F.site_size ; i++)
    F_ghost.set(i, F.get(i));
}

template<typename Float>
static void copy_vertex_to_ghost_ext(pFloat2<Float> F, Float* F_ext, short dir1, short dir2, short dir3, short sign1, short sign2, short sign3){
  if( (dir1 != dir2 && dir1 != dir3 && dir3 != dir2 ) && HGC_dimBreak[dir1] && HGC_dimBreak[dir2] && HGC_dimBreak[dir3] ){
    ProfileStruct ps(F.vertexGhostL(dir1, dir2, dir3));
    tuneAndRun(ps, "copy_vertex_to_ghost_ext_kernel_size_"+std::to_string(F.site_size), copy_vertex_to_ghost_ext_kernel<Float>, F, F_ext, dir1, dir2, dir3, sign1, sign2, sign3);
  }
}


template<typename Float>
static __global__ void conjugate_kernel(generic2<Float> field){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= field.volume()) return;

  field.setSid(sid);
  #pragma unroll
  for(int i = 0 ; i < field.site_size; i++)
    field[i].conj();
}

template<typename Float>
void conjugate_k(PLEGMA_Field<Float>& inOut){
  auto field = toField2<generic2>(inOut);
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (field.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  conjugate_kernel<<<gridDim,blockDim>>>(field);
  checkQudaError();
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
__global__ void genStochasticUniform_kernel(cuRNGState *state, int length_field, Float *inout, bool is4D){

  Float2<Float> *inout2 = (Float2<Float> *) inout;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if( n < 2) return;

  int V = is4D?DGC_localVolume:DGC_localVolume3D;
  for( int i = 0; i < length_field; ++i){

    Float tmp = PLEGMA_Random<Float, Uniform>(state[sid]);

    for( int order = 0; order <= n-1; ++order){

      if( tmp  < ((Float)order+1.0)/(Float)n ){

        inout2[sid + i*(V)] = rootsunity<n>(order);
        break;
      }
    }
  }
} 

template<typename Float, int n>
void set_stochastic( PLEGMA_RNG &rng_state, PLEGMA_Field<Float> &inOut, int field_deg_free, int rng_size){

  dim3 blockDim( THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim( (rng_size  + blockDim.x -1)/blockDim.x , 1 , 1);
  genStochasticUniform_kernel<Float, n><<<gridDim,blockDim>>>(rng_state.State(), field_deg_free, inOut.D_elem(), inOut.is4D());
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
static __global__ void traceMulFmunuSu3FmunuSu3_kernel(Float *F, su3_2<FloatA> RA, su3_2<FloatB> RB, su3_2<FloatC> RC, su3_2<FloatD> RD){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> *F2 = (Float2<Float> *) F;
  if (sid >= RA.volume()) return;
  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatB> lB[N_COLS][N_COLS];
  Float2<FloatC> lC[N_COLS][N_COLS];
  Float2<FloatD> lD[N_COLS][N_COLS];
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
  assert(F.checkVolume(A,B,C,D));
  long int lshift = ((long int) A.munuToIndx(munu_l)) * N_COLS * N_COLS * A.Total_length();
  long int rshift = ((long int) C.munuToIndx(munu_r)) * N_COLS * N_COLS * C.Total_length();
  su3_2<Float> RA((Float2<Float>*) A.D_elem()+lshift, B.Field_length(), A.is4D(), false);
  su3_2<Float> RC((Float2<Float>*) C.D_elem()+rshift, B.Field_length(), C.is4D(), false);

  ProfileStruct ps(RA.volume());
  tuneAndRun(ps, "traceMulFmunuSu3FmunuSu3_kernel", traceMulFmunuSu3FmunuSu3_kernel<Float,FloatA,FloatB,FloatC,FloatD>, F.D_elem(), RA, toField2<su3_2>(B), RC, toField2<su3_2>(D));
  checkQudaError();
}

template<typename FloatA, typename FloatB>
static void __global__ trPmunu_kernel(FloatA *out, gauge2<FloatA> u, int mu, int nu){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;
  Float2<FloatA> *out2 = (Float2<FloatA> *) out;
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
static void __global__ SU3Trace_kernel(FloatA *out, su3_2<FloatA> su3){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;
  Float2<FloatA> *out2 = (Float2<FloatA> *) out;
  Float2<FloatB> SU3[N_COLS][N_COLS];
  su3.get(SU3,sid);
  out2[sid]= SU3[0][0] + SU3[1][1] + SU3[2][2];
}

template<typename FloatA, typename FloatB>
static void trPmunu_k(PLEGMA_Field<FloatA> &f,PLEGMA_Gauge<FloatB> &gauge, std::pair<int,int> munu){
  assert(f.checkVolume(gauge));
  ProfileStruct ps(gauge.Total_length());
  if(std::get<0>(munu) == std::get<1>(munu)) PLEGMA_error("For Pmunu cannot have mu == nu");
  tuneAndRun(ps,"trPmunu_kernel",trPmunu_kernel<FloatA,FloatB>,f.D_elem(),toField2<gauge2>(gauge),std::get<0>(munu),std::get<1>(munu));
  checkQudaError();
}

template<typename FloatA, typename FloatB>
static void SU3Trace_k(PLEGMA_Field<FloatA> &f,PLEGMA_Su3field<FloatB> &su3field){
  assert(f.checkVolume(su3field));
  ProfileStruct ps(su3field.Total_length());
  tuneAndRun(ps,"SU3Trace_kernel",SU3Trace_kernel<FloatA,FloatB>,f.D_elem(),toField2<su3_2>(su3field));
  checkQudaError();
}
