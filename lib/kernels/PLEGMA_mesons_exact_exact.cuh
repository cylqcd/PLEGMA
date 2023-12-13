#include <PLEGMA_kernel_utils.cuh>
#include <utils/PLEGMA_auxiliary.h>
#pragma once
using namespace plegma;

template<typename Float>
__global__ void contract_exact_exact_device( vectorTex<Float> texVec1,
					     vectorTex<Float> texVec2,
					     Float2<Float> *block2,
					     int it, int time_step, int maxT, int4 source,
					     bool runFT, tex_mom_list moms, int is){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  register Float2<Float> accum = 0;

  const Float2<float> (*g)[4];
  g = (Float2<float> (*)[4]) plegma::gamma;
  
  if (sid3D < DGC_localVolume3D){
    Float2<Float> vec1[N_SPINS][N_COLS];
    Float2<Float> vec2[N_SPINS][N_COLS];
    texVec1.get(vec1,vid);
    texVec2.get(vec2,vid);
    #pragma unroll
    for(int nz = 0 ; nz < N_SPINS ; nz++){
      int mu = gammaInd[is][nz][0];
      int nu = gammaInd[is][nz][1];
      Float2<Float> val = g[is][nz];
      #pragma unroll
      for(int a = 0 ; a < N_COLS ; a++){
	accum = accum + val * conj(vec1[mu][a]) * vec2[nu][a];
      }
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<Float> *shared_cache = (Float2<Float> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z}; 
    fourier_transform_3D(block2, &accum, shared_cache, 1, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      block2[tid*DGC_localVolume3D + sid3D] = accum;
  }
}

template<typename Float>
void contract_exact_exact_host( ProfileStruct &ps,
			   PLEGMA_Vector<Float>& vec1, PLEGMA_Vector<Float>& vec2,
			   PLEGMA_Correlator<Float>& corr, Float2<Float> *result, int veci){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int full_site_size = corr.getSiteSize();
  const int site_size = N_SPINS*N_SPINS;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size/full_site_size * (ps.tp.grid.x/time_step)) : (size/full_site_size);

  Float2<Float> *h_block[site_size];
  Float2<Float> *d_block[site_size];
  cudaStream_t stream[site_size];
  for(int k=0; k<site_size; k++) {
    cudaMalloc((void**)&d_block[k], alloc_size*sizeof(Float2<Float>));
    hostMallocPinned(h_block[k], alloc_size*sizeof(Float2<Float>));
    cudaStreamCreate(stream+k) ;
    // Checking for allocation error. In case we return and let the tuner handle the error.
    cudaError_t error=cudaPeekAtLastError();
    if(error != cudaSuccess) {
      for(int i=0; i<k; i++){
	cudaFree(d_block[i]);
	hostFreePinned(h_block[i], alloc_size*sizeof(Float));
	cudaStreamDestroy(stream[i]);
      }
      return;
    }
  }
  
  auto vectorTex1 = toTexture<vectorTex>(vec1);
  auto vectorTex2 = toTexture<vectorTex>(vec2);
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    for(int ip=0; ip < site_size; ip++) {
      contract_exact_exact_device<Float>
	<<<grid,ps.tp.block,ps.tp.shared_bytes,stream[ip]>>>
	  (*vectorTex1, *vectorTex2, d_block[ip], it, std::min(t_size-it, time_step), maxT, source, runFT, *moms, ip);
      cudaError_t error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      cudaMemcpyAsync(h_block[ip], d_block[ip], alloc_size*sizeof(Float2<Float>), cudaMemcpyDeviceToHost, stream[ip]);
    }
    cudaError_t error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
    for(int ip=0; ip < site_size; ip++) {
      cudaStreamSynchronize(stream[ip]);
      if(runFT==true) {
	int accumX = ps.tp.grid.x/time_step;
	for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++) {
	  result[(it*volume+v)*full_site_size+veci*site_size+ip] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(it*volume+v)*full_site_size+veci*site_size+ip] += h_block[ip][v*accumX+j];
	}
      } else {
	  for(size_t v = 0 ; v < volume; v++)
	    result[(it*volume+v)*full_site_size+veci*site_size+ip] = h_block[ip][v];
      }
    }
  }
  for(int k=0; k<site_size; k++) {
    hostFreePinned(h_block[k], alloc_size*sizeof(Float));
    cudaFree(d_block[k]);
    cudaStreamDestroy(stream[k]);
  }
}


template<typename Float>
static void contract_exact_exact(Float* ptr, int nvecs, size_t vec_size,
				 PLEGMA_Correlator<Float>& corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = (nvecs*(nvecs+1)*N_SPINS*N_SPINS)/2;
  std::vector<double> runtime;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? sizeof(Float2<Float>) : 0;

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

  const int ils = 4;
  PLEGMA_Vector<Float> vec1;
  std::vector<PLEGMA_Vector<Float>*> vec2;
  size_t vec_bytes = vec_size*sizeof(Float);
  int veci = 0;
  cudaStream_t stream[ils];
  for(int k=0; k<ils; k++) {
    vec2.push_back(new PLEGMA_Vector<Float>());
    cudaStreamCreate(stream+k) ;
  }
  
  for(int i=0; i<nvecs; i++) {
    PLEGMA_printf("### Contractions for exact_exact vector %d, %s\n", i, getDateAndTime().c_str());
    for(int k=0; k<ils and i+k+1<nvecs; k++) {
      cudaMemcpyAsync(vec2[k]->D_elem(), ptr+(i+k+1)*vec_size, vec_bytes, cudaMemcpyHostToDevice, stream[k]);
    }
    
    cudaMemcpy(vec1.D_elem(), ptr+i*vec_size, vec_bytes, cudaMemcpyHostToDevice);
    checkCudaError();
    
    for(int j=i; j<nvecs; j++) {
      if(j==i) {
	tuneAndRun( ps, "contract_exact_exact_ii", contract_exact_exact_host<Float>,
			 ps, vec1, vec1, corr, result, veci);
      } else {
	int k=(j-i-1)%ils;
	cudaStreamSynchronize(stream[k]);
	checkCudaError();
	tuneAndRun( ps, "contract_exact_exact", contract_exact_exact_host<Float>,
			 ps, vec1, *vec2[k], corr, result, veci);
	// Start copying next vector to use
	if(j+ils<nvecs)
	  cudaMemcpyAsync(vec2[k]->D_elem(), ptr+(j+ils)*vec_size, vec_bytes, cudaMemcpyHostToDevice, stream[k]);
      }
      veci++;
    }
  }
  
  for(int k=0; k<ils; k++) {
    delete vec2[k];
    cudaStreamDestroy(stream[k]);
  }
  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<Float>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<Float>));
  }
}
