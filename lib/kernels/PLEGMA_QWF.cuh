#include "PLEGMA_kernel_utils.cuh"
using namespace plegma;

template<typename FloatA, typename FloatC>
__global__ void contract_TMDWF_mesons_trick_zfac_device(propTex<FloatA>texProp1,
                                                        su3Tex<float> TexStaple,
							int mu, int nu, int c1, int c2, Float2<FloatC> *block2, int it, int time_step, int maxT, int4 source, bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC->localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC->localVolume3D;

  Float2<FloatC> accum[16];
#pragma unroll
  for(int i = 0 ; i < 16 ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC->localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    Float2<float> staple[N_COLS][N_COLS];
    TexStaple.get(staple,vid);

    const Float2<float> (*g)[4];
    const short int (*gIn)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gIn = plegma::gammaInd;

#pragma unroll
    for(int ip = 0 ; ip < 16 ; ip++){
#pragma unroll
      for(int g1=0;g1<4;g1++){
        int alpha = gIn[ip][g1][0];
	int beta = gIn[ip][g1][1];
        if (beta == nu){
	  Float2<FloatC> value = g[ip][g1];
#pragma unroll
	  for(int a = 0 ; a < N_COLS ; a++){
	    accum[ip] = accum[ip] + value * prop1[mu][alpha][c1][a] * staple[c2][a];
	  }
	}
      }
    }
  }

  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block2, accum, shared_cache, 16, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC->localVolume3D)
      for(int ip = 0 ; ip < 16; ip++){
	block2[(tid*DGC->localVolume3D + sid3D)*16 + ip] = accum[ip];
      }
  }
}

template<typename FloatA,typename FloatB,typename FloatC>
__global__ void contract_TMDWF_mesons_zfac_device(propTex<FloatA>texProp1,
                                                  propTex<FloatB>texProp2,
                                                  su3Tex<float> TexStaple,
                                                  int mu, int nu, int c1, int c2, Float2<FloatC> *block2, int it, int time_step, int maxT, int4 source, bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC->localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC->localVolume3D;

  Float2<FloatC> accum[16];
#pragma unroll
  for(int i = 0 ; i < 16 ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC->localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    Float2<float> staple[N_COLS][N_COLS];
    TexStaple.get(staple,vid);

    const Float2<float> (*g)[4];
    const short int (*gIn)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gIn = plegma::gammaInd;

#pragma unroll
    for(int ip = 0 ; ip < 16 ; ip++){
#pragma unroll
      for(int g1=0;g1<4;g1++){
        int alpha = gIn[ip][g1][0];
        int beta = gIn[ip][g1][1];
        if (beta == nu){
          Float2<FloatC> value = g[ip][g1];
#pragma unroll
          for(int rho = 0 ; rho < N_SPINS ; rho++){
#pragma unroll
            for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
	      for(int b = 0 ; b < N_COLS ; b++){
                accum[ip] = accum[ip] + value * prop1[mu][rho][c1][a] * conj(prop2[alpha][rho][b][a]) * staple[c2][b];
              }
            }
          }
        }
      }
    }
  }

  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block2, accum, shared_cache, 16, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC->localVolume3D)
      for(int ip = 0 ; ip < 16; ip++){
        block2[(tid*DGC->localVolume3D + sid3D)*16 + ip] = accum[ip];
      }
  }
}

template<typename FloatA, typename FloatC>
void contract_TMDWF_mesons_trick_zfac_host( ProfileStruct &ps,PLEGMA_Propagator<FloatA>& prop1,
                                            PLEGMA_Correlator<FloatC>& corr, PLEGMA_Su3field<float>& staple,Float2<FloatC> *result){

  int extra = N_SPINS*N_SPINS*N_COLS*N_COLS;
  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT();
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/extra/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = corr.getSiteSize()/extra;

  if(HGC.verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  d_partial_block=(Float2<FloatC>*)device_malloc(alloc_size*sizeof(Float2<FloatC>));
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatC>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto stapleTex = toTexture<su3Tex>(staple);

  //cudaError_t error=cudaPeekAtLastError();
  if(h_partial_block==NULL) {
    hostFree(h_partial_block, alloc_size*sizeof(FloatC));
    device_free(d_partial_block);
    return;
  }

  for(int it=0; it < t_size; it+=time_step) {
    for(int et=0; et < extra; et++){
      int mu=et/N_SPINS/N_COLS/N_COLS;
      int nu=(et/N_COLS/N_COLS)%N_SPINS;
      int c1=(et/N_COLS)%N_COLS;
      int c2=et%N_COLS;
      dim3 grid = ps.tp.grid;
      grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
      contract_TMDWF_mesons_trick_zfac_device
        <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	(*propTex1, *stapleTex, mu, nu, c1, c2, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
    //  error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

      qudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatC>), qudaMemcpyDeviceToHost);
    //  error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

      if(runFT==true) {
        int accumX = ps.tp.grid.x/time_step;
	for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	  for(int f = 0 ; f < site_size; f++) {
	    result[((it*volume+v)*extra+et)*site_size+f] = 0;
	    for(int j = 0 ; j < accumX; j++)
	      result[((it*volume+v)*extra+et)*site_size+f]+= h_partial_block[(v*site_size+f)*accumX+j];
	  }
      } else {
	for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	  for(int f = 0 ; f < site_size; f++) {
	    result[((it*volume+v)*extra+et)*site_size+f] += h_partial_block[v*site_size+f];
	  }
      }
																    
    }
  }

  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  device_free(d_partial_block);
}

template<typename FloatA,typename FloatB,typename FloatC>
void contract_TMDWF_mesons_zfac_host( ProfileStruct &ps,PLEGMA_Propagator<FloatA>& prop1,PLEGMA_Propagator<FloatB>& prop2,
                                            PLEGMA_Correlator<FloatC>& corr, PLEGMA_Su3field<float>& staple,Float2<FloatC> *result){

  int extra = N_SPINS*N_SPINS*N_COLS*N_COLS;
  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT();
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/extra/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = corr.getSiteSize()/extra;

  if(HGC.verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  d_partial_block=(Float2<FloatC>*)device_malloc(alloc_size*sizeof(Float2<FloatC>));
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatC>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto propTex2 = toTexture<propTex>(prop2);
  auto stapleTex = toTexture<su3Tex>(staple);

  //cudaError_t error=cudaPeekAtLastError();
  if(h_partial_block==NULL) {
    hostFree(h_partial_block, alloc_size*sizeof(FloatC));
    device_free(d_partial_block);
    return;
  }

  for(int it=0; it < t_size; it+=time_step) {
    for(int et=0; et < extra; et++){
      int mu=et/N_SPINS/N_COLS/N_COLS;
      int nu=(et/N_COLS/N_COLS)%N_SPINS;
      int c1=(et/N_COLS)%N_COLS;
      int c2=et%N_COLS;
      dim3 grid = ps.tp.grid;
      grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
      contract_TMDWF_mesons_zfac_device
        <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
        (*propTex1, *propTex2, *stapleTex, mu, nu, c1, c2, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
      //error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

      qudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatC>), qudaMemcpyDeviceToHost);
//      error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

      if(runFT==true) {
        int accumX = ps.tp.grid.x/time_step;
        for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
          for(int f = 0 ; f < site_size; f++) {
            result[((it*volume+v)*extra+et)*site_size+f] = 0;
            for(int j = 0 ; j < accumX; j++)
              result[((it*volume+v)*extra+et)*site_size+f]+= h_partial_block[(v*site_size+f)*accumX+j];
          }
      } else {
        for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
          for(int f = 0 ; f < site_size; f++) {
            result[((it*volume+v)*extra+et)*site_size+f] += h_partial_block[v*site_size+f];
          }
      }

    }
  }

  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  device_free(d_partial_block);
}

template<typename FloatA, typename FloatC>
static void contract_TMDWF_mesons_trick_zfac(PLEGMA_Propagator<FloatA>& prop1,PLEGMA_Correlator<FloatC>& corr, PLEGMA_Su3field<float>& staple){

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = N_SPINS*N_SPINS*N_COLS*N_COLS*16;

  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  site_size=16;

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0;

  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.H_elem();

  ProfileStruct ps(HGC.localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC.fullComm);
  ps.max_volume = HGC.localVolume3D*maxLocalT;
  ps.tune_globally = true;

  tuneAndRun( ps, "contract_TMDWF_mesons_trick_zfac", contract_TMDWF_mesons_trick_zfac_host<FloatA,FloatC>,
              ps, prop1, corr, staple, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatC>(), MPI_SUM, HGC.spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
}
	    
template<typename FloatA,typename FloatB,typename FloatC>
static void contract_TMDWF_mesons_zfac(PLEGMA_Propagator<FloatA>& prop1,PLEGMA_Propagator<FloatB>& prop2,PLEGMA_Correlator<FloatC>& corr, PLEGMA_Su3field<float>& staple){

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = N_SPINS*N_SPINS*N_COLS*N_COLS*16;

  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  site_size=16;

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0;

  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.H_elem();

  ProfileStruct ps(HGC.localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC.fullComm);
  ps.max_volume = HGC.localVolume3D*maxLocalT;
  ps.tune_globally = true;

  tuneAndRun( ps, "contract_TMDWF_mesons_zfac", contract_TMDWF_mesons_zfac_host<FloatA,FloatB,FloatC>,
              ps, prop1, prop2, corr, staple, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatC>(), MPI_SUM, HGC.spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
}
