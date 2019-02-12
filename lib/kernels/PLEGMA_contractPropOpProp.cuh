#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>
#include <PLEGMA_gammas.cuh>

using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool runFT, bool isLink, int dir, bool isCons>
__global__ void contractPropOpProp_kernel(FloatC* block, propTex<FloatA> prop1Tex, propTex<FloatB> prop2Tex, su3Tex<FloatS> su3Tx, KernelArr<GAMMAS> listGammas, int it, int x0, int y0, int z0, int signProps){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*c_stride_spatial;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;

  Float2<FloatC> R[N_SPINS][N_SPINS];
  Float2<FloatC> noeV;  
  if (sid < DGC_localVolume/DGC_localL[3]){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    if(dir < 0){ // either local or Wilson line
      prop1Tex.get(prop1,vid);
      prop2Tex.get(prop2,vid);
      if(isLink){ // for Wilson line
	Float2<FloatS> su3[N_COLS][N_COLS];
	su3Tx.get(su3,vid);
	partial_trace_mul_Prop_G_Prop<true,ACC_ZERO,false>(R,prop1,prop2,su3);
      }
      else{ // local operators
	partial_trace_mul_Prop_Prop<true,ACC_ZERO>(R,prop1,prop2);}
    }
    else{ // either oneD or conserved current
      // term x, x, x+dir
      Float2<FloatS> su3[N_COLS][N_COLS];
      prop1Tex.get(prop1,vid);  su3Tx.get(su3,vid); prop2Tex.get<Plus>(prop2,vid,dir);
      partial_trace_mul_Prop_G_Prop<true,ACC_ZERO,false>(R,prop1,prop2,su3);
      if(isCons) noeV = trace_gamma_S<true>(listGammas.array[0],NOROT,R) - trace_gamma_S<true>(ONE,NOROT,R);

      //term x, x-dir, x-dir
      su3Tx.get<Minus>(su3,vid,dir); prop2Tex.get<Minus>(prop2,vid,dir);
      if(isCons){ partial_trace_mul_Prop_G_Prop<true,ACC_ZERO,true>(R,prop1,prop2,su3);
	noeV += trace_gamma_S<true>(listGammas.array[0],NOROT,R) + trace_gamma_S<true>(ONE,NOROT,R);}
      else partial_trace_mul_Prop_G_Prop<true,ACC_MINUS,true>(R,prop1,prop2,su3);

      //term x+dir, x, x
      prop1Tex.get<Plus>(prop1,vid,dir); su3Tx.get(su3,vid); prop2Tex.get(prop2,vid);
      if(isCons){ partial_trace_mul_Prop_G_Prop<true,ACC_ZERO,true>(R,prop1,prop2,su3);
	noeV += trace_gamma_S<true>(listGammas.array[0],NOROT,R) + trace_gamma_S<true>(ONE,NOROT,R);}
      else partial_trace_mul_Prop_G_Prop<true,ACC_MINUS,true>(R,prop1,prop2,su3);

      //term x-dir, x-dir, x
      prop1Tex.get<Minus>(prop1,vid,dir); su3Tx.get<Minus>(su3,vid,dir);
      if(isCons){ partial_trace_mul_Prop_G_Prop<true,ACC_ZERO,false>(R,prop1,prop2,su3);
	noeV += trace_gamma_S<true>(listGammas.array[0],NOROT,R) - trace_gamma_S<true>(ONE,NOROT,R);}
      else partial_trace_mul_Prop_G_Prop<true,ACC_PLUS,false>(R,prop1,prop2,su3);      
    }    
  }

  Float2<FloatC> accum = 0.;
  int source_pos[3] = {x0, y0, z0};

  for(int iop = 0; iop < listGammas.size; iop++){
    int opId=listGammas.array[iop];
    accum.x=0.;accum.y=0.;
    if (sid < DGC_localVolume/DGC_localL[3]){
      if(isCons) accum = 0.25*noeV;
      else accum = (dir<0 ? 1. : 0.25) * ( (signProps > 0) ? trace_gamma_S<true>(opId,TMP,R) : trace_gamma_S<true>(opId,TMM,R));
    }
    if(runFT){
      //    extern __shared__ int ext_shared_cache[];
      __shared__ Float2<FloatC> ext_shared_cache[THREADS_PER_BLOCK];
      Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
      fourier_transform_3D(block2+iop*gridDim.x, &accum, shared_cache, 1, sid, source_pos,listGammas.size-1,+1);
    }
    else{
      if (sid < DGC_localVolume/DGC_localL[3])
	for(int iop = 0; iop < listGammas.size; iop++)
	  block2[sid*listGammas.size +iop] = accum;
    }
  }
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool runFT, bool isLink, int dir, bool isCons>
static void contractPropOpProp_k(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatA> prop2,
				 int signProps, su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas){
  if(gammas.size() <= 0)
    errorQuda("Error the container of gamma matrices cannot be zero");
  if(gammas.size() > 16)
    errorQuda("Error maximum number of gamma matrices is 16");
  int SpVol = HGC_localVolume/HGC_localL[3];
  FloatC *d_partial_block = NULL;
  int site_size=gammas.size();
  size_t volume = corr.getVolSize()/HGC_localL[3];
  size_t size = corr.getTotalSize()/HGC_localL[3];
  int3 source = corr.getSource3();

  if(corr.getSiteSize() != site_size)
    errorQuda("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  KernelArr<GAMMAS> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
  checkCudaError();
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);
  checkCudaError();

  // !!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!! //
  // when do tuning do not set sharedBytesPerThread = site_size*sizeof(Float2<FloatC>); but sharedBytesPerThread = sizeof(Float2<FloatC>)
  
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume
  size_t alloc_size;
  if(runFT==true){
    alloc_size = size * gridDim.x * 2;
  } else {
    alloc_size = size * 2;
  }
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(FloatC));
  checkCudaError();
  contractPropOpProp_kernel<FloatC,FloatA, FloatB, FloatS, runFT,
			    isLink, dir,isCons>
    <<<gridDim,blockDim>>>(d_partial_block, prop1, prop2, su3, listGammas, it,
			   source, signProps);
  checkCudaError();
  
  FloatC *h_partial_block = NULL;
  hostMalloc(h_partial_block, alloc_size*sizeof(FloatC));
  cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_block);
  cudaFree(listGammas.array);
  checkCudaError();
  
  if(runFT==true){
    FloatC *reduction;
    hostMalloc(reduction, size*2*sizeof(FloatC));
    for(size_t i = 0 ; i < size; i++) {
      reduction[i*2+0] = 0;
      reduction[i*2+1] = 0;
      for(int j = 0 ; j < gridDim.x; j++) {
	reduction[i*2+0] += h_partial_block[(i*gridDim.x + j)*2+0];
	reduction[i*2+1] += h_partial_block[(i*gridDim.x + j)*2+1];
      }
    }
    MPI_Allreduce(reduction, h_partial_block, size*2, MPI_Type(reduction), MPI_SUM, HGC_spaceComm);
    hostFree(reduction, size*2*sizeof(FloatC));
  }
  
  FloatC *corr_pt = corr.getCorr();
  int shift = (dir<0) ? 0 : dir*gammas.size()*2;
  for(size_t v = 0 ; v < volume; v++)
    for(int i = 0 ; i < gammas.size(); i++) {
      corr_pt[it*volume*site_size*2+v*site_size*2+shift+i*2+0] =
	h_partial_block[(v*gammas.size()+i)*2+0];
      corr_pt[it*volume*site_size*2+v*site_size*2+shift+i*2+1] =
	h_partial_block[(v*gammas.size()+i)*2+1];
    }
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
}


template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool isLink, int dir, bool isCons>
static void contractPropOpProp(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps,
			       su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas){
  if(!isLink && dir>=0) errorQuda("Does not make sence to do not have links and have directions");
  if(isCons && !isLink) errorQuda("Does not make sence to do noether current without links");
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  if(corr.getCorrSpace() == POSITION_SPACE)
    contractPropOpProp_k<FloatC,FloatA,FloatB,FloatS,false,isLink,dir,isCons>(corr,prop1,prop2,signProps,su3,it,gammas);
  else if(corr.getCorrSpace() == MOMENTUM_SPACE)
    contractPropOpProp_k<FloatC,FloatA,FloatB,FloatS,true,isLink,dir,isCons>(corr,prop1,prop2,signProps,su3,it,gammas);
  else
    errorQuda("Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
#else
  errorQuda("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK\n");
#endif
}
