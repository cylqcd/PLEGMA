#include <PLEGMA_Correlator.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>
#include <PLEGMA_gammas.cuh>

using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

template<typename FloatC, typename FloatA, typename FloatB, typename FloatS, bool isLink, int dir, bool isCons>
__global__ void contractPropOpProp_device(Float2<FloatC>* block2, propTex<FloatA> prop1Tex, propTex<FloatB> prop2Tex,
					  su3Tex<FloatS> su3Tx, KernelArr<GAMMAS> listGammas, int it, int time_step,
					  int3 source, int signProps, bool runFT, tex_mom_list moms){
  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  int vid = sid3D + (it+tid)*DGC_localVolume3D;
  
  Float2<FloatC> R[N_SPINS][N_SPINS];
  Float2<FloatC> noeV=0;
  #pragma unroll
  for(int i = 0; i < N_SPINS; i++)
    #pragma unroll
    for(int j = 0; j < N_SPINS; j++)
      R[i][j]=0;

  if (sid3D < DGC_localVolume3D){
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

  Float2<FloatC> accum[16]; // max value
  int source_pos[3] = {source.x, source.y, source.z}; 

  for(int iop = 0; iop < listGammas.size; iop++){
    int opId=listGammas.array[iop];
    if(isCons) accum[iop] = 0.25*noeV;
    else accum[iop] = (dir<0 ? 1. : 0.25) * ( (signProps > 0) ? trace_gamma_S<true>(opId,TMP,R) : trace_gamma_S<true>(opId,TMM,R));
  }
  if(runFT){
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    fourier_transform_3D(block2, accum, shared_cache, listGammas.size, sid3D, source_pos, moms, 0, +1, time_step, tid);
  } else{
    if (sid3D < DGC_localVolume3D)
      for(int iop = 0; iop < listGammas.size; iop++)
	block2[(tid*DGC_localVolume3D + sid3D)*listGammas.size +iop] = accum[iop];
  }
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool isLink, int dir, bool isCons>
static void contractPropOpProp_host(ProfileStruct &ps, Float2<FloatC> *result, PLEGMA_Correlator<FloatC> &corr,
				    propTex<FloatA> prop1, propTex<FloatA> prop2,
				    int signProps, su3Tex<FloatS> su3, std::vector<GAMMAS> gammas){
  
  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;
  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  size_t volume = corr.getVolSize()/HGC_localL[3];
  size_t size = corr.getTotalSize()/HGC_localL[3];
  int site_size = gammas.size();
  int3 source = corr.getSource3();
  tex_mom_list moms = corr.getTexMomList();

  int shift = (dir<0) ? 0 : dir*gammas.size();
  int Mshift = (dir<0) ? 1 : N_DIMS;

  KernelArr<GAMMAS> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);

  if(HGC_verbosity > 2)
    PLEGMA_printf("time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true)? (size * (ps.tp.grid.x/time_step) ) : size;

  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(Float2<FloatC>) );
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatC>));
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess || h_partial_block==NULL) goto exit;

  for(int it=0; it < HGC_localL[3]; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*MIN(HGC_localL[3]-it, time_step);
    contractPropOpProp_device<FloatC,FloatA, FloatB, FloatS, isLink, dir,isCons>
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (d_partial_block, prop1, prop2, su3, listGammas, it, MIN(HGC_localL[3]-it, time_step),
       source, signProps, runFT, moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;

    cudaMemcpy(h_partial_block , d_partial_block , (alloc_size/time_step)*MIN(HGC_localL[3]-it, time_step)*sizeof(Float2<FloatC>) , cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;

    if(runFT==true){
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*MIN(HGC_localL[3]-it, time_step); v++)
	for(int i = 0 ; i < site_size; i++) {
	    result[(it*volume+v)*Mshift*site_size+shift+i] = 0;
	    for(int j = 0 ; j < accumX; j++)
	      result[(it*volume+v)*Mshift*site_size+shift+i] +=
		h_partial_block[(v*site_size+i)*accumX+j];
	}
    } else {
      for(size_t v = 0 ; v < volume*MIN(HGC_localL[3]-it, time_step); v++)
	for(int i = 0 ; i < site_size; i++)
	  result[(it*volume+v)*Mshift*site_size+shift+i] +=
	    h_partial_block[v*site_size+i];
    }
  }
  
 exit:
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  cudaFree(d_partial_block);
  cudaFree(listGammas.array);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS, bool isLink, int dir, bool isCons>
static void contractPropOpProp(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			       int signProps, su3Tex<FloatS> su3, std::vector<GAMMAS> gammas){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  if(!isLink && dir>=0) PLEGMA_error("Does not make sense to do not have links and have directions");
  if(isCons && !isLink) PLEGMA_error("Does not make sense to do noether current without links");
  if(gammas.size() <= 0)
    PLEGMA_error("Error the container of gamma matrices cannot be zero");
  if(gammas.size() > 16)
    PLEGMA_error("Error maximum number of gamma matrices is 16");
  if(isCons && gammas.size()!=1) PLEGMA_error("Does not make sense to do noether current with more than one gamma");

  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  int site_size = gammas.size();
  if(dir <  0){
    if(corr.getSiteSize() != site_size)
      PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);
  }
  else{
    if(corr.getSiteSize() != N_DIMS * site_size)
      PLEGMA_error("Correlator siteSize do not match: %d != %d * %d\n", corr.getSiteSize(), N_DIMS, site_size);
  }

  ProfileStruct ps(HGC_localVolume3D, (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0);
  ps.max_volume = HGC_localVolume;

  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.getCorr();
  
  std::string name = (std::string) "contract_threep_"+(isLink?"isLink_":"")+(isCons?"isCons_":"")+
    "dir"+std::to_string(dir)+"_nGamma"+std::to_string(site_size);
  tuneAndRun( ps, name, contractPropOpProp_host<FloatC,FloatA,FloatB,FloatS,isLink,dir,isCons>,
	      ps, result, corr, prop1, prop2, signProps, su3, gammas);

  if(runFT) {
    MPI_Allreduce(result, corr.getCorr(), corr.getTotalSize()*2, MPI_Type(corr.getCorr()),
		  MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK\n");
#endif
}
