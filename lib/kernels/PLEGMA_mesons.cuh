#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
const int N_MESONS=10;
// TODO: This is hard to extend. These variables should replaced by compile-time functions.
const __device__ short int mesons_indices[N_MESONS][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1};

const __device__ float mesons_values[N_MESONS][16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

template<typename FloatA, typename FloatB, typename FloatC>
__global__ void contract_mesons_kernel( propTex<FloatA> texProp1,
					propTex<FloatB> texProp2,
					FloatC* block, int it, int3 source,
					bool runFT, tex_mom_list moms){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int vid = sid + it*DGC_localVolume3D;
  Float2<FloatC> *block2 = (Float2<FloatC> *)block;
    
  register Float2<FloatC> accum[2*N_MESONS];
  for(int i = 0 ; i < 2*N_MESONS ; i++){
    accum[i] = 0.;
  }

  if (sid < DGC_localVolume3D){ // run only on the spatial volume
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
      int source_pos[3] = {source.x, source.y, source.z}; 
      fourier_transform_3D(block2, accum, shared_cache, 2*N_MESONS, sid, source_pos, moms);
    } else {
      if(block2 != NULL)
	for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	  block2[sid*2*N_MESONS + ip] = accum[ip];
	}
    }
  }
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_mesons(propTex<FloatA> texProp1, propTex<FloatB> texProp2,
			    PLEGMA_Correlator<FloatC> &corr, int it){
  int SpVol = HGC_localVolume/HGC_localL[3];
  FloatC *d_partial_block = NULL;

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = 2*N_MESONS;
  size_t volume = corr.getVolSize()/HGC_localL[3];
  size_t size = corr.getTotalSize()/HGC_localL[3];
  int3 source = corr.getSource3();
  tex_mom_list moms = corr.getTexMomList();
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*2*sizeof(FloatC) : 0;
  
  ProfileStruct ps(SpVol, shared_size);
  tune( ps, "contract_mesons_kernel", contract_mesons_kernel<FloatA,FloatB,FloatC>,
	texProp1, texProp2, d_partial_block, it, source, runFT, moms);
  
  size_t alloc_size;
  if(runFT==true){
    alloc_size = size * ps.tp.grid.x * 2;
  } else {
    alloc_size = size * 2;
  }
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(FloatC));
  run( ps, "contract_mesons_kernel", contract_mesons_kernel<FloatA,FloatB,FloatC>,
       texProp1, texProp2, d_partial_block, it, source, runFT, moms);
  checkCudaError();
  
  FloatC *h_partial_block = NULL;
  hostMalloc(h_partial_block, alloc_size*sizeof(FloatC));
  cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_block);
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
  
  FloatC *corr_pt = corr.getCorr();
  for(size_t v = 0 ; v < volume; v++)
    for(int f = 0 ; f < site_size; f++) {
      corr_pt[((f*HGC_localL[3] + it)*volume+v)*2+0] = h_partial_block[(v*site_size/2+f)*2+0];
      corr_pt[((f*HGC_localL[3] + it)*volume+v)*2+1] = h_partial_block[(v*site_size/2+f)*2+1];
    }
  
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
}
