#include <PLEGMA_kernel_utils.cuh>
#pragma once
using namespace plegma;

template<typename Float>
struct Spin { Float2<Float> vals[N_SPINS]; };

template<typename Float>
__global__ void contract_stoch_exact_device( propTex<Float> texProp,
					     vectorTex<Float> texVec,
					     Spin<Float> spin,
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

  if (sid3D < DGC_localVolume3D){
    Float2<Float> prop[N_SPINS][N_SPINS][N_COLS][1];
    Float2<Float> vec[N_SPINS][N_COLS];
    texProp.get<1>(prop,vid);
    texVec.get(vec,vid);
    short int alpha = (is/N_SPINS/N_SPINS/N_SPINS)%N_SPINS;
    short int beta = (is/N_SPINS/N_SPINS)%N_SPINS;
    short int gamma = (is/N_SPINS)%N_SPINS;
    short int delta = is%N_SPINS;
    #pragma unroll
    for(int b = 0 ; b < N_COLS ; b++)
      accum = accum + vec[alpha][b] * spin.vals[beta] * conj(prop[delta][gamma][b][0]);
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
void contract_stoch_exact_host( ProfileStruct &ps,
				PLEGMA_Propagator<Float> &prop, PLEGMA_Vector<Float>& vec,
				Spin<Float>& spin,
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
  int site_size = N_SPINS*N_SPINS*N_SPINS*N_SPINS;
  
  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
  
  size_t alloc_size = (runFT==true) ? (size/full_site_size * (ps.tp.grid.x/time_step)) : size/full_site_size;
  
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

  auto propT = toTexture<propTex>(prop);
  auto vecT = toTexture<vectorTex>(vec);
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    for(int ip=0; ip < site_size; ip++) {
      contract_stoch_exact_device<Float>
	<<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	(*propT, *vecT, spin, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms, ip);
      error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
      cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<Float>), cudaMemcpyDeviceToHost);
      error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
      if(runFT==true) {
	int accumX = ps.tp.grid.x/time_step;
	for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++) {
	  result[(it*volume+v)*full_site_size+veci*site_size+ip] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(it*volume+v)*full_site_size+veci*site_size+ip] += h_partial_block[v*accumX+j];
	}
      } else {
	for(size_t v = 0 ; v < volume; v++)
	  result[(it*volume+v)*full_site_size+veci*site_size+ip] = h_partial_block[v];
      }
    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(Float));
  cudaFree(d_partial_block);
}


template<typename Float>
static void contract_stoch_exact(PLEGMA_Propagator<Float> &prop, Float *spinVals,
				 Float* ptr, int nvecs, size_t vec_size,
				 PLEGMA_Correlator<Float>& corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = corr.getSiteSize();
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
  std::vector<PLEGMA_Vector<Float>*> vec;
  size_t vec_bytes = vec_size*sizeof(Float);
  cudaStream_t stream[ils];
  for(int k=0; k<ils; k++) {
    vec.push_back(new PLEGMA_Vector<Float>());
    cudaStreamCreate(stream+k) ;
    cudaMemcpyAsync(vec[k]->D_elem(), ptr+k*vec_size, vec_bytes, cudaMemcpyHostToDevice, stream[k]);
  }

  Spin<Float> spin;
  
  for(int i=0; i<nvecs; i++) {
    for(int mu=0; mu<4; mu++) {
      spin.vals[mu] = {spinVals[mu*nvecs*2+i*2+0], spinVals[mu*nvecs*2+i*2+1]};
    }
    
    int k=i%ils;
    cudaStreamSynchronize(stream[k]);
    checkCudaError();
    tuneAndRun( ps, "contract_stoch_exact", contract_stoch_exact_host<Float>,
		ps, prop, *vec[k], spin, corr, result, i);
    
    // Start copying next vector to use
    if(i+ils<nvecs)
      cudaMemcpyAsync(vec[k]->D_elem(), ptr+(i+ils)*vec_size, vec_bytes, cudaMemcpyHostToDevice, stream[k]);
  }
  
  for(int k=0; k<ils; k++) {
    delete vec[k];
    cudaStreamDestroy(stream[k]);
  }
  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<Float>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<Float>));
  }
}
