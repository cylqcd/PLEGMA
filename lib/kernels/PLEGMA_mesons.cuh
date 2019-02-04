#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
// TODO: This is hard to extend. These variables should replaced by compile-time functions.
const __device__ short int mesons_indices[10][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1};

const __device__ float mesons_values[10][16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__global__ void contract_mesons_kernel( propTex<FloatA> texProp1,
					propTex<FloatB> texProp2,
					FloatC* block,
					int it, int x0, int y0, int z0){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*DGC_stride_spatial;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;
    
  register Float2<FloatC> accum[2*N_MESONS];
  for(int i = 0 ; i < 2*N_MESONS ; i++){
    accum[i] = 0.;
  }

  if (sid < DGC_threads/DGC_localL[3]){ // run only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
#pragma unroll
    for(int ip = 0 ; ip < N_MESONS ; ip++){
#pragma unroll
      for(int is = 0 ; is < N_SPINS*N_SPINS ; is++){
	short int beta = mesons_indices[ip][is][0];
	short int gamma = mesons_indices[ip][is][1];
	short int delta = mesons_indices[ip][is][2];
	short int alpha = mesons_indices[ip][is][3];
	FloatC value = mesons_values[ip][is];
#pragma unroll
	for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
	  for(int b = 0 ; b < N_COLS ; b++){
	    accum[ip*2+0] = accum[ip*2+0] + value * prop1[alpha][beta][a][b] * conj(prop1[delta][gamma][a][b]);
	    accum[ip*2+1] = accum[ip*2+1] + value * prop2[alpha][beta][a][b] * conj(prop2[delta][gamma][a][b]);
	  }
	}
      }
    }
    if(runFT) {
      extern __shared__ int ext_shared_cache[];
      Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
      int source_pos[3] = {x0, y0, z0}; 
      fourier_transform_3D(block2, accum, shared_cache, 2*N_MESONS, sid, source_pos);
    } else {
      if(block2 != NULL)
	for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	  block2[sid*2*N_MESONS + ip] = accum[ip];
	}
    }
  }
}

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
static void contract_mesons(propTex<FloatA> texProp1, propTex<FloatB> texProp2, PLEGMA_Correlator<FloatC> &corr, int it){

  int SpVol = HGC_localVolume/HGC_localL[3];
  FloatC *d_partial_block = NULL;
  int isource = corr.getIdSource();
  int site_size=2*N_MESONS*2;
  size_t volume;
  size_t size;
  if(runFT==true){
    volume = HGC_Nmoms;
    size = site_size*volume;
  } else {
    volume = SpVol;
    size = site_size*volume;
  }

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0;
  
  ProfileStruct ps(SpVol, shared_size);
 
  tune( ps, "contract_mesons_kernel", contract_mesons_kernel<FloatA,FloatB,FloatC,runFT>,
	texProp1, texProp2, d_partial_block, it,
	GK_sourcePosition[isource][0], GK_sourcePosition[isource][1],
	GK_sourcePosition[isource][2]);
    
#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif

  int gridDimX = ps.tp.grid.x;
  size_t alloc_size;
  if(runFT==true){
    alloc_size = size * gridDimX;
  } else {
    alloc_size = size;
  }
  cudaMalloc((void**)&d_partial_block, alloc_size*2*sizeof(FloatC));
  contract_mesons_kernel<FloatA,FloatB,FloatC,runFT><<<ps.tp.grid,
    ps.tp.block,ps.tp.shared_bytes>>>(texProp1,texProp2, d_partial_block, it,
				      GK_sourcePosition[isource][0],
				      GK_sourcePosition[isource][1],
				      GK_sourcePosition[isource][2]);
  checkCudaError();
  
#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);
#endif

  FloatC *h_partial_block = NULL;
  h_partial_block = (FloatC*)malloc(alloc_size*sizeof(FloatC));
  if(h_partial_block == NULL) errorQuda("contract_mesons_kernel: Cannot allocate host block.\n");
  cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_block);
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
  
  FloatC *corr_pt = corr.getCorr();
  for(size_t v = 0 ; v < volume; v++)
    for(int f = 0 ; f < site_size/2; f++) {
      corr_pt[((f*HGC_localL[3] + it)*volume+v)*2+0] = h_partial_block[(v*site_size/2+f)*2+0];
      corr_pt[((f*HGC_localL[3] + it)*volume+v)*2+1] = h_partial_block[(v*site_size/2+f)*2+1];
    }

  free(h_partial_block);
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_mesons(propTex<FloatA> texProp1, propTex<FloatB> texProp2, PLEGMA_Correlator<FloatC> &corr, int it){
  if (corr.getCorrSpace()==POSITION_SPACE){
    contract_mesons<FloatA,FloatB,FloatC,false>(texProp1,texProp2,corr,it);
  }
  else if(corr.getCorrSpace()==MOMENTUM_SPACE) {
    contract_mesons<FloatA,FloatB,FloatC,true>(texProp1,texProp2,corr,it);
  }
  else
    errorQuda("run_contract_mesons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();
}
