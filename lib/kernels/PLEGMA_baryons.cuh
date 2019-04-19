#pragma once

#include <PLEGMA_kernel_utils.cuh>

enum BARYONS_TYPE{NtoN,		
#ifdef PLEGMA_ALL_BARYONS		
		  NtoR, RtoN, RtoR, DELTA_1O2_1, DELTA_1O2_2, DELTA_1O2_3,		
		  DELTA_3O2_1, DELTA_3O2_2, DELTA_3O2_3,		
#endif		
		  // add here		
		  N_BARYONS}; // N_BARYONS must be last 

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_NtoN_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_NtoR_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_RtoN_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_RtoR_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC, int gamma>
__device__ void contract_deltas_iso1o2_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC, int gamma>
__device__ void contract_deltas_iso3o2_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__global__ void contract_baryons_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, FloatC* block,
					int it, int3 source, BARYONS_TYPE ip){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*DGC_localVolume3D;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;

  Float2<FloatC> accum[2*N_SPINS*N_SPINS];
  
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (sid < DGC_localVolume3D){ // I work only on the spatial volume
    switch(ip){
    case NtoN:
      contract_NtoN_kernel<FloatA,FloatB,FloatC>(texProp1, texProp2, accum, vid);
      break;
#ifdef PLEGMA_ALL_BARYONS
    case NtoR:
      contract_NtoR_kernel<FloatA,FloatB,FloatC>(texProp1, texProp2, accum, vid);
      break;
    case RtoN:
      contract_RtoN_kernel<FloatA,FloatB,FloatC>(texProp1, texProp2, accum, vid);
      break;
    case RtoR:
      contract_RtoR_kernel<FloatA,FloatB,FloatC>(texProp1, texProp2, accum, vid);
      break;
    case DELTA_1O2_1:
      contract_deltas_iso1o2_kernel<FloatA,FloatB,FloatC,0>(texProp1, texProp2, accum, vid);
      break;
    case DELTA_1O2_2:
      contract_deltas_iso1o2_kernel<FloatA,FloatB,FloatC,1>(texProp1, texProp2, accum, vid);
      break;
    case DELTA_1O2_3:
      contract_deltas_iso1o2_kernel<FloatA,FloatB,FloatC,2>(texProp1, texProp2, accum, vid);
      break;
    case DELTA_3O2_1:
      contract_deltas_iso3o2_kernel<FloatA,FloatB,FloatC,0>(texProp1, texProp2, accum, vid);
      break;
    case DELTA_3O2_2:
      contract_deltas_iso3o2_kernel<FloatA,FloatB,FloatC,1>(texProp1, texProp2, accum, vid);
      break;
    case DELTA_3O2_3:
      contract_deltas_iso3o2_kernel<FloatA,FloatB,FloatC,2>(texProp1, texProp2, accum, vid);
      break;
#endif
    }
  }
  
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z}; 
    fourier_transform_3D(block2, accum, shared_cache, 2*N_SPINS*N_SPINS, sid, source_pos);
  } else {
    if(block2!=NULL)
      for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
	block2[sid*2*N_SPINS*N_SPINS + i] = accum[i];
    }
  }
}

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
static void contract_baryons(propTex<FloatA> texProp1, propTex<FloatB> texProp2, PLEGMA_Correlator<FloatC> &corr, int it){

  int SpVol = HGC_localVolume3D;

  FloatC *h_partial_block = NULL;
  FloatC *d_partial_block = NULL;

  int site_size=2*N_SPINS*N_SPINS;
  size_t volume = corr.getVolSize()/HGC_localL[3];
  size_t size = corr.getTotalSize()/HGC_localL[3]/N_BARYONS;
  int3 source = corr.getSource3();
  
  if(corr.getSiteSize()/N_BARYONS != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*2*sizeof(FloatC) : 0;
  ProfileStruct ps(SpVol, shared_size);
  tune( ps, "contract_baryons_kernel", contract_baryons_kernel<FloatA,FloatB,FloatC,runFT>,
	texProp1, texProp2, d_partial_block, it, source,  (BARYONS_TYPE) 0); // tuning done for first baryon
  
  size_t alloc_size;
  if(runFT==true){
      alloc_size = size * ps.tp.grid.x * 2;
  } else {
      alloc_size = size * 2;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(FloatC));
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(FloatC) );
  checkCudaError();

  if(runFT) cudaFuncSetCacheConfig(contract_baryons_kernel<FloatA,FloatB,FloatC,runFT>, cudaFuncCachePreferShared);

  for(int ip=0; ip<N_BARYONS; ip++) {
    run( ps, "contract_baryons_kernel", contract_baryons_kernel<FloatA,FloatB,FloatC,runFT>,
	 texProp1, texProp2, d_partial_block, it, source,  (BARYONS_TYPE) ip);
    checkCudaError();    
    cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
    checkCudaError();
    if(runFT==true){
      int gridDimX = ps.tp.grid.x;
      FloatC *reduction;
      hostMalloc(reduction, size*2*sizeof(FloatC));
      for(size_t i = 0 ; i < size; i++) {
	reduction[i*2+0] = 0;
	reduction[i*2+1] = 0;
	for(int j = 0 ; j < gridDimX; j++) {
	  reduction[i*2+0] += h_partial_block[(i*gridDimX + j)*2+0];
	  reduction[i*2+1] += h_partial_block[(i*gridDimX + j)*2+1];
	}
      }
      MPI_Allreduce(reduction, h_partial_block, size*2, MPI_Type(reduction), MPI_SUM, HGC_spaceComm);
      hostFree(reduction, size*2*sizeof(FloatC));
    }

    // Reordering accordingly to the wanted data layout
    FloatC *corr_ip = corr.getCorr() + ip*HGC_localL[3]*size*2;
    for(size_t v = 0 ; v < volume; v++)
      for(int f = 0 ; f < 2; f++)
	for(int i = 0 ; i < site_size; i++)
	  corr_ip[((f*HGC_localL[3] + it)*volume +v)*site_size+i] = h_partial_block[(v*2+f)*site_size+i];
    
  }
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  cudaFree(d_partial_block);
  checkCudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_baryons(propTex<FloatA> texProp1, propTex<FloatB> texProp2,
			     PLEGMA_Correlator<FloatC> &corr, int it){
  if (corr.getCorrSpace()==POSITION_SPACE){
    contract_baryons<FloatA,FloatB,FloatC,false>(texProp1,texProp2,corr,it);
  }
  else if(corr.getCorrSpace()==MOMENTUM_SPACE) {
    contract_baryons<FloatA,FloatB,FloatC,true>(texProp1,texProp2,corr,it);
  }
  else
    PLEGMA_error("run_contract_baryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();
}
