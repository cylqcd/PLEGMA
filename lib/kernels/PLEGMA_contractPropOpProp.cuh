#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>
#include <PLEGMA_gammas.cuh>

using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool runFT, bool isLink>
struct ArgsPropOpProp{
  FloatC* block;
  propTex<FloatA> prop1;
  propTex<FloatB> prop2;
  su3Tex<FloatS> su3;
  KernelArr<GAMMAS> listGammas;
  int it, x0, y0, z0;
  int signProps;
};

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool runFT, bool isLink>
__global__ void contractPropOpProp_kernel(ArgsPropOpProp<FloatC,FloatA,FloatB,FloatS,runFT,isLink> args){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + args.it*c_stride_spatial;
  Float2<FloatC> *block2 = (Float2<FloatC> *)args.block;

  Float2<FloatC> accum = 0.;
  int source_pos[3] = {args.x0, args.y0, args.z0};
  Float2<FloatC> R[N_SPINS][N_SPINS];
  if (sid < c_threads/c_localL[3]){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    args.prop1.get(prop1,vid);
    args.prop2.get(prop2,vid);
    if(isLink){
      Float2<FloatS> su3[N_COLS][N_COLS];
      args.su3.get(su3,vid);
      partial_trace_mul_Prop_G_Prop<true>(R,prop1,prop2,su3);
    }
    else
      partial_trace_mul_Prop_Prop<true>(R,prop1,prop2); 
  }


  const Float2<float> (*gtm)[4];
  const short int (*gtmIn)[4][2];
  if(args.signProps > 0){
    gtm = (Float2<float> (*)[4]) gammaTmP;
    gtmIn = gammaIndTmP; 
  }else{
    gtm = (Float2<float> (*)[4]) gammaTmM;
    gtmIn = gammaIndTmM;     
  }
  
  for(int iop = 0; iop < args.listGammas.size; iop++){
    int opId=args.listGammas.array[iop];
    accum=0.;
    if (sid < c_threads/c_localL[3]){
#pragma unroll
      for(int nz = 0 ; nz < N_SPINS ; nz++){
	int mu = gtmIn[opId][nz][0];
	int nu = gtmIn[opId][nz][1];
	Float2<FloatC> val = gtm[opId][nz];
	accum = accum + val*R[mu][nu];
      }
    }
    if(runFT){
      //    extern __shared__ int ext_shared_cache[];
      __shared__ Float2<FloatC> ext_shared_cache[THREADS_PER_BLOCK];
      Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
      fourier_transform_3D(block2+opId*gridDim.x, &accum, shared_cache, 1, sid, source_pos,args.listGammas.size-1,+1);
      // shuffling
    }
    else{
      if (sid < c_threads/c_localL[3])
	for(int iop = 0; iop < args.listGammas.size; iop++)
	  block2[sid*args.listGammas.size + iop] = accum;
    }
  }
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool runFT, bool isLink>
static void contractPropOpProp_k(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatA> prop2,
				 int signProps, su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas){
  if(gammas.size() <= 0)
    errorQuda("Error the container of gamma matrices cannot be zero");
  if(gammas.size() > 16)
    errorQuda("Error maximum number of gamma matrices is 16");
  int SpVol = GK_localVolume/GK_localL[3];
  FloatC *d_partial_block = NULL;
  int isource = corr.getIdSource();
  int site_size=2*gammas.size();
  size_t volume;
  size_t size;
  if(runFT==true){
    volume = GK_Nmoms;
    size = site_size*volume;
  } else {
    volume = SpVol;
    size = site_size*volume;
  }

  ArgsPropOpProp<FloatC,FloatA,FloatB,FloatS,runFT,isLink> args;
  args.prop1 = prop1;
  args.prop2 = prop2;
  args.su3 = su3;
  args.signProps = signProps;
  KernelArr<GAMMAS> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
  checkCudaError();
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);
  checkCudaError();
  args.listGammas = listGammas;

  args.x0 = GK_sourcePosition[isource][0];
  args.y0 = GK_sourcePosition[isource][1];
  args.z0 = GK_sourcePosition[isource][2];
  args.it = it;

  // !!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!! //
  // when do tuning do not set sharedBytesPerThread = site_size*sizeof(Float2<FloatC>); but sharedBytesPerThread = sizeof(Float2<FloatC>)
  
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume
  size_t alloc_size;
  if(runFT==true){
    alloc_size = size * gridDim.x;
  } else {
    alloc_size = size;
  }
  cudaMalloc((void**)&d_partial_block, alloc_size*2*sizeof(FloatC));
  args.block = d_partial_block;
  checkCudaError();

  contractPropOpProp_kernel<<<gridDim,blockDim>>>(args);
  checkCudaError();
  
  FloatC *h_partial_block = NULL;
  h_partial_block = (FloatC*)malloc(alloc_size*sizeof(FloatC));
  if(h_partial_block == NULL) errorQuda("contractPropOpProp: Cannot allocate host block.\n");
  cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_block);
  cudaFree(listGammas.array);
  checkCudaError();
  
  if(runFT==true){
    FloatC *reduction =(FloatC*) calloc(size,sizeof(FloatC));
    for(size_t i = 0 ; i < size/2; i++)
      for(int j = 0 ; j < gridDim.x; j++) {
	reduction[i*2+0] += h_partial_block[(i*gridDim.x + j)*2+0];
	reduction[i*2+1] += h_partial_block[(i*gridDim.x + j)*2+1];
      }
    MPI_Allreduce(reduction, h_partial_block, size, MPI_Type(reduction), MPI_SUM, GK_spaceComm);
    free(reduction);
  }
  
  FloatC *corr_pt = corr.getCorr();
  int sz = corr.getSiteSize();
  if(sz < gammas.size())errorQuda("The size of list with gammas exceeds the site_size of correlators\n");
  for(size_t v = 0 ; v < volume; v++)
    for(int i = 0 ; i < gammas.size(); i++) {
      corr_pt[it*volume*sz*2+v*sz*2+i*2+0] = h_partial_block[(v*gammas.size()+i)*2+0];
      corr_pt[it*volume*sz*2+v*sz*2+i*2+1] = h_partial_block[(v*gammas.size()+i)*2+1];
    }

  free(h_partial_block);

}


template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool isLink>
static void contractPropOpProp(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps,
			       su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas){
  if(corr.getCorrSpace() == POSITION_SPACE)
    contractPropOpProp_k<FloatC,FloatA,FloatB,FloatS,false,isLink>(corr,prop1,prop2,signProps,su3,it,gammas);
  else if(corr.getCorrSpace() == MOMENTUM_SPACE)
    contractPropOpProp_k<FloatC,FloatA,FloatB,FloatS,true,isLink>(corr,prop1,prop2,signProps,su3,it,gammas);
  else
    errorQuda("Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
}

template<typename FloatC,typename FloatA, typename FloatB>
static void contractPropOpProp(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps, int it,
                                 std::vector<GAMMAS> gammas){
  su3Tex<FloatC> su3;
  su3.tex=0;
  contractPropOpProp<FloatC,FloatA,FloatB,FloatC,false>(corr,prop1,prop2,signProps,su3,it, gammas);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
static void contractPropOpProp(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps,
			       su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas){
  contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true>(corr,prop1,prop2,signProps,su3,it, gammas);
}
