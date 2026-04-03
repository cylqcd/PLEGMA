#include <PLEGMA_kernel_utils.cuh>
#include <malloc_quda.h>
#pragma once
using namespace plegma;
const int N_MESONS=10;
// TODO: This is hard to extend. These variables should replaced by compile-time functions.
const __device__ short int mesons_indices[N_MESONS][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1};

const __device__ float mesons_values[N_MESONS][16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};

template<typename FloatA, typename FloatB, typename FloatC>
__global__ void contract_mesons_device( propTex<FloatA> texProp1,
					propTex<FloatB> texProp2,
					Float2<FloatC> *block2,
					int it, int time_step, int maxT, int4 source,
					bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  register Float2<FloatC> accum[2*N_MESONS];
  for(int i = 0 ; i < 2*N_MESONS ; i++){
    accum[i] = 0.;
  }
  if (sid3D < DGC_localVolume3D){
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
	    accum[ip] = accum[ip] + value * prop1[alpha][beta][a][b] * conj(prop1[delta][gamma][a][b]);
	    accum[N_MESONS+ip] = accum[N_MESONS+ip] + value * prop2[alpha][beta][a][b] * conj(prop2[delta][gamma][a][b]);
	  }
	}
      }
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z}; 
    fourier_transform_3D(block2, accum, shared_cache, 2*N_MESONS, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
	block2[(tid*DGC_localVolume3D + sid3D)*2*N_MESONS + ip] = accum[ip];
      }
  }
}

template<typename FloatA, typename FloatB, typename FloatC, typename FloatD, typename FloatE>
__global__ void contract_mesons_fourp_ultralocal_device( propTex<FloatA> texProp1,
                                        		 propTex<FloatB> texProp2,
					                 propTex<FloatC> texProp3,
							 propTex<FloatD> texProp4,
							 Float2<FloatE> *block2,
							 int it, int time_step, int maxT, int4 source, bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;

  register Float2<FloatE> accum[16];
  for(int i = 0 ; i < 16 ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC_localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatC> prop3[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatD> prop4[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    texProp3.get(prop3,vid);
    texProp4.get(prop4,vid);

    const Float2<float> (*g)[4];
    const short int (*gIn)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gIn = plegma::gammaInd;

#pragma unroll
    for(int ip = 0 ; ip < 16 ; ip++){
#pragma unroll
      for(int alpha = 0 ; alpha < N_SPINS ; alpha++){
#pragma unroll
	for(int delta = 0 ; delta < N_SPINS ; delta++){
#pragma unroll
	  for(int g1 = 0 ; g1 < 4 ; g1++){
	    int mu = gIn[ip][g1][0];
	    int nu = gIn[ip][g1][1];
	    Float2<FloatE> val1 = g[ip][g1];
#pragma unroll
	    for(int g2 = 0 ; g2 < 4 ; g2++){
	      int rho = gIn[ip][g2][0];
	      int sigma = gIn[ip][g2][1];
	      Float2<FloatE> val2 = g[ip][g2];
#pragma unroll
	      for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
		for(int b = 0 ; b < N_COLS ; b++){
#pragma unroll
		  for(int c = 0 ; c < N_COLS ; c++){
#pragma unroll
		    for(int d = 0 ; d < N_COLS ; d++){
		      accum[ip] += val1 * val2 * conj(prop1[mu][alpha][b][a]) * prop2[nu][delta][b][c] * conj(prop3[rho][delta][d][c]) * prop4[sigma][alpha][d][a];
		    }
		  }
		}
              }
	    }
	  }
	}
      }
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatE> *shared_cache = (Float2<FloatE> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block2, accum, shared_cache, 16, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      for(int ip = 0 ; ip < 16 ; ip++){
	block2[(tid*DGC_localVolume3D + sid3D)*16 + ip] = accum[ip];
      }
  }
}
																									
template<typename FloatA, typename FloatB, typename FloatE>
__global__ void contract_mesons_fourp_ultralocal_oneendtrick_device( propTex<FloatA> texProp1, propTex<FloatB> texProp2, Float2<FloatE> *block2, int it, int time_step, int maxT, int4 source, bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;

  register Float2<FloatE> accum[16];
  for(int i = 0 ; i < 16 ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC_localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);

    const Float2<float> (*g)[4];
    const short int (*gIn)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gIn = plegma::gammaInd;

#pragma unroll
    for(int ip = 0 ; ip < 16 ; ip++){
#pragma unroll
      for(int g1 = 0 ; g1 < 4 ; g1++){
        int mu = gIn[ip][g1][0];
        int nu = gIn[ip][g1][1];
	Float2<FloatE> val1 = g[ip][g1];
#pragma unroll
	for(int g2 = 0 ; g2 < 4 ; g2++){
	  int rho = gIn[ip][g2][0];
	  int sigma = gIn[ip][g2][1];
	  Float2<FloatE> val2 = g[ip][g2];
#pragma unroll
	  for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
	    for(int b = 0 ; b < N_COLS ; b++){
	      accum[ip] += val1 * val2 * prop1[nu][rho][a][b] * prop2[sigma][mu][b][a];
	    }
	  }
	}
      }
    }
  }

  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatE> *shared_cache = (Float2<FloatE> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block2, accum, shared_cache, 16, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      for(int ip = 0 ; ip < 16 ; ip++){
        block2[(tid*DGC_localVolume3D + sid3D)*16 + ip] = accum[ip];
      }
  }
}
					      
template<typename FloatA, typename FloatB, typename FloatC>
void contract_mesons_host( ProfileStruct &ps,
			   PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2,
			   PLEGMA_Correlator<FloatC>& corr, Float2<FloatC> *result){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = 2*N_MESONS;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatC>));
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
    contract_mesons_device
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (*propTex1, *propTex2, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatC>), cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
    if(runFT==true) {
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	for(int f = 0 ; f < site_size; f++) {
	  result[(((f/N_MESONS)*t_size+it)*volume+v)*N_MESONS+f % N_MESONS] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(((f/N_MESONS)*t_size+it)*volume+v)*N_MESONS+f % N_MESONS] += h_partial_block[(v*site_size+f)*accumX+j];
	}
    } else {
      for(size_t v = 0 ; v < volume; v++)
	for(int f = 0 ; f < site_size; f++) {
	  result[(((f/N_MESONS)*t_size+it)*volume+v)*N_MESONS+f % N_MESONS] = h_partial_block[v*site_size+f];
	}
    }
  }

  printf("PLEGMA_mesons res %e %e %e t_size = %d, maxT = %d, source.w = %d, HGC_localVolume3D %d time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", result[0].norm2(),result[1].norm2(),result[2].norm(),t_size, maxT, source.w, HGC_localVolume3D, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  cudaFree(d_partial_block);
}

template<typename FloatA, typename FloatB, typename FloatC, typename FloatD, typename FloatE>
void contract_mesons_fourp_ultralocal_host( ProfileStruct &ps,
                           		    PLEGMA_Propagator<FloatA> &prop1, PLEGMA_Propagator<FloatB> &prop2, PLEGMA_Propagator<FloatC> &prop3, PLEGMA_Propagator<FloatD> &prop4,
			                    PLEGMA_Correlator<FloatE>& corr, Float2<FloatE> *result){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT();
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = 16;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<FloatE> *h_partial_block = NULL;
  Float2<FloatE> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatE>));
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatE>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto propTex2 = toTexture<propTex>(prop2);
  auto propTex3 = toTexture<propTex>(prop3);
  auto propTex4 = toTexture<propTex>(prop4);

  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    contract_mesons_fourp_ultralocal_device
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (*propTex1, *propTex2, *propTex3, *propTex4, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatE>), cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    if(runFT==true) {
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
        for(int f = 0 ; f < site_size; f++) {
	  result[(((f/16)*t_size+it)*volume+v)*16+f % 16] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(((f/16)*t_size+it)*volume+v)*16+f % 16] += h_partial_block[(v*site_size+f)*accumX+j];
	}
    } else {
      for(size_t v = 0 ; v < volume; v++)
        for(int f = 0 ; f < site_size; f++) {
	  result[(((f/16)*t_size+it)*volume+v)*16+f % 16] = h_partial_block[v*site_size+f];
	}
    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(FloatE));
  cudaFree(d_partial_block);
}													      

template<typename FloatA, typename FloatB, typename FloatE>
void contract_mesons_fourp_ultralocal_oneendtrick_host( ProfileStruct &ps,
                                                        PLEGMA_Propagator<FloatA> &prop1, PLEGMA_Propagator<FloatB> &prop2,
			                                PLEGMA_Correlator<FloatE>& corr, Float2<FloatE> *result){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT();
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = 16;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<FloatE> *h_partial_block = NULL;
  Float2<FloatE> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatE>));
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatE>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto propTex2 = toTexture<propTex>(prop2);

  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    contract_mesons_fourp_ultralocal_oneendtrick_device
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (*propTex1, *propTex2, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatE>), cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    if(runFT==true) {
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
        for(int f = 0 ; f < site_size; f++) {
	  result[(((f/16)*t_size+it)*volume+v)*16+f % 16] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(((f/16)*t_size+it)*volume+v)*16+f % 16] += h_partial_block[(v*site_size+f)*accumX+j];
	}
      } else {
	for(size_t v = 0 ; v < volume; v++)
	  for(int f = 0 ; f < site_size; f++) {
	    result[(((f/16)*t_size+it)*volume+v)*16+f % 16] = h_partial_block[v*site_size+f];
	  }
      }
    }
    
  hostFree(h_partial_block, alloc_size*sizeof(FloatE));
  cudaFree(d_partial_block);
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_mesons(PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2,
			    PLEGMA_Correlator<FloatC>& corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = 2*N_MESONS;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0;

  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;
  
  tuneAndRun( ps, "contract_mesons", contract_mesons_host<FloatA,FloatB,FloatC>,
	      ps, prop1, prop2, corr, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatC>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
}

template<typename FloatA, typename FloatB, typename FloatC, typename FloatD, typename FloatE>
static void contract_mesons_fourp_ultralocal(PLEGMA_Propagator<FloatA> &prop1, PLEGMA_Propagator<FloatB> &prop2, PLEGMA_Propagator<FloatC> &prop3, PLEGMA_Propagator<FloatD> &prop4, PLEGMA_Correlator<FloatE>& corr){

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = 16;

  if(corr.getSiteSize() != site_size)
  PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatE>) : 0;

  Float2<FloatE> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatE>));
  else
    result = (Float2<FloatE> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  tuneAndRun( ps, "contract_mesons_fourp_ultralocal", contract_mesons_fourp_ultralocal_host<FloatA,FloatB,FloatC,FloatD,FloatE>,
                ps, prop1, prop2, prop3, prop4, corr, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatE>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatE>));
  }
}	    

template<typename FloatA, typename FloatB, typename FloatE>
static void contract_mesons_fourp_ultralocal_oneendtrick(PLEGMA_Propagator<FloatA> &prop1, PLEGMA_Propagator<FloatB> &prop2, PLEGMA_Correlator<FloatE>& corr){

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = 16;

  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatE>) : 0;

  Float2<FloatE> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatE>));
  else
    result = (Float2<FloatE> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  tuneAndRun( ps, "contract_mesons_fourp_ultralocal_oneendtrick", contract_mesons_fourp_ultralocal_oneendtrick_host<FloatA,FloatB,FloatE>,
                ps, prop1, prop2, corr, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatE>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatE>));
  }
}
	    
