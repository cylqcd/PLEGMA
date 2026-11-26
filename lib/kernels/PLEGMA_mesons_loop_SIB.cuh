#include <PLEGMA_kernel_utils.cuh>
#include <utils/PLEGMA_auxiliary.h>
#pragma once
using namespace plegma;

template<typename Float>
__global__ void contract_loop_SIB_device( vectorTex<Float> vect,
				      propTex<Float> prop,
				      Float2<Float> *block2,
				      int it, int time_step, int maxT, int4 source,
				      bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  //int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int t=0; // here we use prop3D and vect 3D
  int vid = sid3D + t*DGC_localVolume3D;
  
  register Float2<Float> accum[3*N_SPINS*N_SPINS];
  for(int i = 0 ; i < 3*N_SPINS*N_SPINS; i++){
    accum[i] = 0.;
  }
  
  const Float2<float> (*g)[4];
  g = (Float2<float> (*)[4]) plegma::gamma;
  
  if (sid3D < DGC_localVolume3D){
    Float2<Float> vect1[N_SPINS][N_COLS];
    Float2<Float> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];

    vect.get(vect1,vid);
    prop.get(prop1,vid);
#pragma unroll
    for(int is = 0 ; is < 3*N_SPINS*N_SPINS ; is++){
    #pragma unroll
      for(int nz = 0 ; nz < N_SPINS ; nz++){
	int mu = gammaInd[is%(N_SPINS*N_SPINS)][nz][0];
	int nu = gammaInd[is%(N_SPINS*N_SPINS)][nz][1];
	Float2<Float> val = g[is%(N_SPINS*N_SPINS)][nz];
#pragma unroll
	for(int a = 0 ; a < N_COLS ; a++){
	  accum[is] = accum[is] + val * conj(vect1[nu][a]) * prop1[mu][nu][a][is/(N_SPINS*N_SPINS)];
	}
      }
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<Float> *shared_cache = (Float2<Float> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z}; 
    fourier_transform_3D(block2, accum, shared_cache, 3*N_SPINS*N_SPINS, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      for(int ip = 0 ; ip < 3*N_SPINS*N_SPINS ; ip++){
	block2[(tid*DGC_localVolume3D + sid3D)*3*N_SPINS*N_SPINS + ip] = accum[ip];
      }
  }
}

template<typename Float>
void contract_loop_SIB_host( ProfileStruct &ps,
			 PLEGMA_Vector3D<Float>& vect, PLEGMA_Propagator3D<Float>& prop,
			 PLEGMA_Correlator<Float>& corr, Float2<Float> *result){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = 3*N_SPINS*N_SPINS;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<Float> *h_partial_block = NULL;
  Float2<Float> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<Float>));
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<Float>));

  auto vectTex1 = toTexture<vectorTex>(vect);
  auto propTex1 = toTexture<propTex>(prop);
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    contract_loop_SIB_device
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (*vectTex1, *propTex1, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<Float>), cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
    if(runFT==true) {
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	for(int f = 0 ; f < site_size; f++) {
	  result[(f*t_size + it)*volume+v] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(f*t_size + it)*volume+v] += h_partial_block[(v*site_size+f)*accumX+j];
	}
    } else {
      for(size_t v = 0 ; v < volume; v++)
	for(int f = 0 ; f < site_size; f++) {
	  result[(f*t_size + it)*volume+v] = h_partial_block[v*site_size+f];
	}
    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(Float));
  cudaFree(d_partial_block);
}


template<typename Float>
static void contract_loop_SIB(PLEGMA_Vector3D<Float>& vect, PLEGMA_Propagator3D<Float>& prop,
			  PLEGMA_Correlator<Float>& corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = 3*N_SPINS*N_SPINS;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<Float>) : 0;

  Float2<Float> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<Float>));
  else
    result = (Float2<Float> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  tuneAndRun( ps, "contract_loop_SIB", contract_loop_SIB_host<Float>,
	      ps, vect, prop, corr, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<Float>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<Float>));
  }
}
