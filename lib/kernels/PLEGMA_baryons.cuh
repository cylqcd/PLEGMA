#include <PLEGMA_kernel_utils.cuh>

static const __device__ short int Delta_indices[3][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
static const __device__ float Delta_values[3][16] = {1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

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
					int it, int x0, int y0, int z0, BARYONS_TYPE ip){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*DGC_stride_spatial;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;

  Float2<FloatC> accum[2*N_SPINS*N_SPINS];
  
  for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
    accum[i]=0;
  }
  if (sid < DGC_threads/DGC_localL[3]){ // I work only on the spatial volume
    switch(ip){
    case NtoN:
      contract_NtoN_kernel<FloatA,FloatB,FloatC>(texProp1, texProp2, accum, vid);
      break;
#ifdef ALL_BARYONS
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
    int source_pos[3] = {x0, y0, z0}; 
    __shared__ Float2<FloatC> shared_cache[2*N_SPINS*N_SPINS*THREADS_PER_BLOCK];
    fourier_transform_3D(block2, accum, shared_cache, 2*N_SPINS*N_SPINS, sid, source_pos);
  } else {
    for(int i = 0 ; i < 2*N_SPINS*N_SPINS ; i++){
      block2[sid*2*N_SPINS*N_SPINS + i] = accum[i];
    }
  }
}

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
static void contract_baryons(propTex<FloatA> texProp1, propTex<FloatB> texProp2, PLEGMA_Correlator<FloatC> &corr, int it){

  int SpVol = HGC_localVolume/HGC_localL[3];

  FloatC *h_partial_block = NULL;
  FloatC *d_partial_block = NULL;

  int n_flavors=2;
  int site_size=N_SPINS*N_SPINS*2;
  size_t volume;
  size_t size;
  if(runFT==true){
    volume = HGC_Nmoms;
    size = n_flavors*site_size*volume;
  } else {
    volume = SpVol;
    size = n_flavors*site_size*volume;
  }

  int isource = corr.getIdSource();
  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0;
  ProfileStruct ps(SpVol, shared_size);
  tune( ps, "contract_baryons_kernel", contract_baryons_kernel<FloatA,FloatB,FloatC,runFT>,
	texProp1, texProp2, d_partial_block, it,
	GK_sourcePosition[isource][0],
	GK_sourcePosition[isource][1],
	GK_sourcePosition[isource][2],  (BARYONS_TYPE) 0); // tuning done for first baryon
  
  int gridDimX = ps.tp.grid.x;
  size_t alloc_size;
  if(runFT==true){
      alloc_size = size * gridDimX;
  } else {
      alloc_size = size;
  }
  
  h_partial_block = (FloatC*)malloc(alloc_size*sizeof(FloatC));
  if(h_partial_block == NULL) errorQuda("contract_baryons_kernel: Cannot allocate host block.\n");
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(FloatC) );
  checkCudaError();

  if(runFT) cudaFuncSetCacheConfig(contract_baryons_kernel<FloatA,FloatB,FloatC,runFT>, cudaFuncCachePreferShared);

  for(int ip=0; ip<N_BARYONS; ip++) {
    contract_baryons_kernel<FloatA,FloatB,FloatC,runFT><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>( texProp1, texProp2,
													d_partial_block, it,
													HGC_sourcePosition[isource][0],
													HGC_sourcePosition[isource][1],
													HGC_sourcePosition[isource][2],
													(BARYONS_TYPE) ip);
    checkCudaError();    
    cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
    checkCudaError();
    if(runFT==true){
      FloatC *reduction =(FloatC*) calloc(size,sizeof(FloatC));
      for(size_t i = 0 ; i < size/2; i++)
	for(int j = 0 ; j < gridDimX; j++) {
	  reduction[i*2+0] += h_partial_block[(i*gridDimX + j)*2+0];
	  reduction[i*2+1] += h_partial_block[(i*gridDimX + j)*2+1];
	}
      MPI_Allreduce(reduction, h_partial_block, size, MPI_Type(reduction), MPI_SUM, HGC_spaceComm);
      free(reduction);
    }

    FloatC *corr_ip = corr.getCorr() + ip*HGC_localL[3]*size;
    for(size_t v = 0 ; v < volume; v++)
      for(int f = 0 ; f < n_flavors; f++)
	for(int i = 0 ; i < site_size; i++)
	  corr_ip[((f*HGC_localL[3] + it)*volume +v)*site_size+i] = h_partial_block[(v*n_flavors+f)*site_size+i];
    
  }
  free(h_partial_block);
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
    errorQuda("run_contract_baryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();
}
