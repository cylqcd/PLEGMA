#pragma once

#include <PLEGMA_kernel_utils.cuh>

enum BARYONS_TYPE{NtoN,		
#ifdef PLEGMA_LIGHT_BARYONS		
		  NtoR, RtoN, RtoR, DELTA_1O2_1, DELTA_1O2_2, DELTA_1O2_3,		
		  DELTA_3O2_1, DELTA_3O2_2, DELTA_3O2_3,		
#endif		
		  // add here		
		  N_BARYONS}; // N_BARYONS must be last 

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_NtoN_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_NtoR_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_RtoN_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__device__ void contract_RtoR_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC, int gamma>
__device__ void contract_deltas_iso1o2_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC, int gamma>
__device__ void contract_deltas_iso3o2_kernel(propTex<FloatA>& texProp1, propTex<FloatB>& texProp2, Float2<FloatC> accum[2*N_SPINS*N_SPINS], int vid);

template<typename FloatA, typename FloatB, typename FloatC>
__global__ void contract_baryons_device(propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatC>* block2,
					int it, int time_step, int maxT, int4 source, BARYONS_TYPE ip, bool runFT, tex_mom_list mom_list){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  Float2<FloatC> accum[2*N_SPINS*N_SPINS];
  
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (sid3D < DGC_localVolume3D){ // I work only on the spatial volume
    switch(ip){
    case NtoN:
      contract_NtoN_kernel<FloatA,FloatB,FloatC>(texProp1, texProp2, accum, vid);
      break;
#ifdef PLEGMA_LIGHT_BARYONS
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
    fourier_transform_3D(block2, accum, shared_cache, 2*N_SPINS*N_SPINS, sid3D, source_pos, mom_list, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
	block2[(tid*DGC_localVolume3D + sid3D)*2*N_SPINS*N_SPINS + i] = accum[i];
      }
  }
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_baryons_host( ProfileStruct &ps,
				   PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2,
				   PLEGMA_Correlator<FloatC> &corr, Float2<FloatC> *result, int ip){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);

  bool runFT = corr.getCorrSpace()==MOMENTUM_SPACE;
  size_t volume = corr.getVolSize()/t_size;
  size_t size = corr.getTotalSize()/t_size/N_BARYONS*time_step;
  int site_size=2*N_SPINS*N_SPINS;
  int4 source = corr.getSource();
  auto mom_list = corr.getTexMomList();

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true)? (size * (ps.tp.grid.x/time_step) ) : size;
  
  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(Float2<FloatC>) );
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatC>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto propTex2 = toTexture<propTex>(prop2);

  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    contract_baryons_device
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (*propTex1, *propTex2, d_partial_block, it, std::min(t_size-it, time_step), maxT, source,
       (BARYONS_TYPE) ip, runFT, *mom_list);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block , d_partial_block , (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatC>) , cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    if(runFT==true){
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	for(int f = 0 ; f < 2; f++)
	  for(int i = 0 ; i < site_size/2; i++) {
	    result[((f*t_size + it)*volume +v)*site_size/2+i] = 0;
	    for(int j = 0 ; j < accumX; j++)
	      result[((f*t_size + it)*volume +v)*site_size/2+i] +=
		h_partial_block[((v*2+f)*site_size/2+i)*accumX+j];
	  }
    } else {
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	for(int f = 0 ; f < 2; f++)
	  for(int i = 0 ; i < site_size/2; i++)
	    result[((f*t_size + it)*volume +v)*site_size/2+i] = 
	      h_partial_block[(v*2+f)*site_size/2+i];

    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(Float2<FloatC>));
  cudaFree(d_partial_block);
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_baryons(PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, PLEGMA_Correlator<FloatC> &corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size=2*N_SPINS*N_SPINS;
  
  if(corr.getSiteSize()/N_BARYONS != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize()/N_BARYONS, site_size);

  int shared_size = (runFT==true) ? (site_size*sizeof(Float2<FloatC>)) : 0;

  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, (corr.getTotalSize()/N_BARYONS)*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  for(int ip=0; ip<N_BARYONS; ip++) {
    std::string name = "contract_baryons_"+std::to_string(ip);
    tuneAndRun( ps, name, contract_baryons_host<FloatA,FloatB,FloatC>,
		ps, prop1, prop2, corr, result, ip);
    if(runFT) {
      FloatC *corr_ip = corr.H_elem() + ip*corr.getTotalSize()/N_BARYONS*2;
      MPI_Allreduce(result, corr_ip, corr.getTotalSize()/N_BARYONS*2, MPI_Type(corr_ip),
		    MPI_SUM, HGC_spaceComm);
    } else {
      result += corr.getTotalSize()/N_BARYONS;
    }
  } 
  if(runFT)
    hostFree(result, (corr.getTotalSize()/N_BARYONS)*sizeof(Float2<FloatC>));
}
