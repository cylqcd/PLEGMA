#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
// TODO: This is hard to extend. These variables should replaced by compile-time functions.
const __device__ short int mesons_indices[10][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1};

const __device__ float mesons_values[10][16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
__global__ void contract_mesons_kernel(propTex<FloatA> texProp1, propTex<FloatB> texProp2, FloatC* block,
				      int it, int x0, int y0, int z0){


  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*c_stride_spatial;
  int locV = blockDim.x * gridDim.x;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;

  register Float2<FloatC> accum[2*N_MESONS];
  for(int i = 0 ; i < 2*N_MESONS ; i++){
    accum[i] = 0.;
  }

  if (sid < c_threads/c_localL[3]){ // run only on the spatial volume
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
    __syncthreads();

    if(runFT) {
      int cacheIndex = threadIdx.x;
      __shared__ Float2<FloatC> shared_cache[2*N_MESONS*THREADS_PER_BLOCK];
      
      int x_id, y_id , z_id;
      int r1,r2;
      
      r1 = sid / c_localL[0];
      x_id = sid - r1 * c_localL[0];
      r2 = r1 / c_localL[1];
      y_id = r1 - r2*c_localL[1];
      z_id = r2;
      
      int x,y,z;
      
      x = x_id + c_procPosition[0] * c_localL[0] - x0;
      y = y_id + c_procPosition[1] * c_localL[1] - y0;
      z = z_id + c_procPosition[2] * c_localL[2] - z0;
      
      FloatC phase;
      Float2<FloatC> expon;
      for(int imom = 0 ; imom < c_Nmoms ; imom++){
	phase = ( ((double) c_moms[imom][0]*x)/c_totalL[0] + ((double) c_moms[imom][1]*y)/c_totalL[1] + ((double) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
	expon.x = cos(phase);
	expon.y = -sin(phase);
	for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	  shared_cache[ip*THREADS_PER_BLOCK + cacheIndex] = accum[ip] * expon; 
	}
	__syncthreads();
	int i = blockDim.x/2;
	while (i != 0){
	  if(cacheIndex < i){
	    for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	      shared_cache[ip*THREADS_PER_BLOCK + cacheIndex] = shared_cache[ip*THREADS_PER_BLOCK + cacheIndex] +
		shared_cache[ip*THREADS_PER_BLOCK + cacheIndex + i];
	    }
	  }
	  __syncthreads();
	  i /= 2;
	}
	
	if(cacheIndex == 0){
	  for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	    block2[(imom*2*N_MESONS + ip)*gridDim.x + blockIdx.x] = shared_cache[ip*THREADS_PER_BLOCK];
	  }
	}
      } // close momentum
    } else {
      for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	block2[ip*locV + sid] = accum[ip];
      }
    }
    __syncthreads();
  }
}

template<typename FloatA, typename FloatB, typename FloatC, bool runFT>
static void contract_mesons(propTex<FloatA> texProp1, propTex<FloatB> texProp2, PLEGMA_Correlator<FloatC> &corr, int it){

  int SpVol = GK_localVolume/GK_localL[3];

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume

  FloatC *h_partial_block = NULL;
  FloatC *d_partial_block = NULL;

  size_t alloc_size;
  FloatC *corr_it;
  if(runFT==true){
    alloc_size = corr.getSiteSize() * GK_Nmoms * gridDim.x;
    corr_it = corr.getCorr() + it*corr.getSiteSize()*GK_Nmoms*2;
  } else {
    alloc_size = corr.getSiteSize() * SpVol; 
    corr_it = corr.getCorr() + it*alloc_size*2;
  }
  h_partial_block = (FloatC*)malloc(alloc_size*2*sizeof(FloatC));
  if(h_partial_block == NULL) errorQuda("contract_mesons_kernel: Cannot allocate host block.\n");
  cudaMalloc((void**)&d_partial_block, alloc_size*2 * sizeof(FloatC) );
  checkCudaError();

  int isource = corr.getIdSource();
  cudaFuncSetCacheConfig(contract_mesons_kernel<FloatA,FloatB,FloatC,runFT>, cudaFuncCachePreferShared);
  contract_mesons_kernel<FloatA,FloatB,FloatC,runFT><<<gridDim,blockDim>>>( texProp1, texProp2, d_partial_block, it,
									    GK_sourcePosition[isource][0],
									    GK_sourcePosition[isource][1],
									    GK_sourcePosition[isource][2]);
  checkCudaError();

  cudaMemcpy(h_partial_block , d_partial_block , alloc_size*2*sizeof(FloatC) , cudaMemcpyDeviceToHost);
  checkCudaError();

  if(runFT==true){
    alloc_size = corr.getSiteSize() * GK_Nmoms;
    FloatC *reduction =(FloatC*) calloc(alloc_size*2,sizeof(FloatC));
    for(size_t i = 0 ; i < alloc_size; i++)
      for(int j = 0 ; j < gridDim.x; j++) {
	reduction[i*2+0] += h_partial_block[(i*gridDim.x + j)*2+0];
	reduction[i*2+1] += h_partial_block[(i*gridDim.x + j)*2+1];
      }
    MPI_Allreduce(reduction, corr_it, alloc_size*2, MPI_Type(reduction), MPI_SUM, GK_spaceComm);
    free(reduction);
  } else {
    for(size_t i = 0 ; i < alloc_size*2; i++)
      corr_it[i] = h_partial_block[i];
  }
  free(h_partial_block);
  cudaFree(d_partial_block);
  checkCudaError();
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
