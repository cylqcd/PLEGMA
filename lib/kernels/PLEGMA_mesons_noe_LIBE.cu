#include <PLEGMA_Correlator.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>
#include <PLEGMA_gammas.cuh>
#include <PLEGMA_threep.cuh>

template<typename FloatC, typename FloatA, typename FloatB, typename FloatG>
__global__ void mesons_noe_LIBE_device(Float2<FloatC>* block2,
				  propTex<FloatA> prop1Tex, propTex<FloatB> prop2Tex,
				  gaugeTex<FloatG> gaugeTex,
				  int it, int time_step, int maxT, int4 source,
				  bool runFT, tex_mom_list moms){
  return;
  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  const int site_size = 9*N_DIMS;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  Float2<FloatC> noeV[site_size];
  #pragma unroll
  for(int i = 0; i < site_size; i++)
    noeV[i] = 0;

  if (sid3D < DGC_localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatC> R[N_COLS][N_COLS][N_SPINS][N_SPINS];
    Float2<FloatG> su3[N_COLS][N_COLS];
    GAMMAS gamma_dir[4] = {G5G1, G5G2, G5G3, G5G4};

    for(int dir = 0; dir < N_DIMS; dir++) {
      // term x, x, x+dir
      prop1Tex.get(prop1,vid); gaugeTex.get(su3,dir,vid); prop2Tex.get<Plus>(prop2,vid,dir);
      partial_trace_mul_Prop_G_Prop_meson_LIBE<ACC_ZERO,false>(R,prop1,prop2,su3);
      for(int s=0; s<9; s++){
	noeV[s*N_DIMS+dir] += trace_gamma_S<true>(gamma_dir[dir],NOROT,R[s/3][s%3]) - trace_gamma_S<true>(G5,NOROT,R[s/3][s%3]);
      }

      //term x, x-dir, x-dir
      gaugeTex.get<Minus>(su3,dir,vid,dir); prop2Tex.get<Minus>(prop2,vid,dir);
      partial_trace_mul_Prop_G_Prop_meson_LIBE<ACC_ZERO,true>(R,prop1,prop2,su3);
      for(int s=0; s<9; s++){
	noeV[s*N_DIMS+dir] += trace_gamma_S<true>(gamma_dir[dir],NOROT,R[s/3][s%3]) - trace_gamma_S<true>(G5,NOROT,R[s/3][s%3]);
      }

      //term x+dir, x, x
      prop1Tex.get<Plus>(prop1,vid,dir); gaugeTex.get(su3,dir,vid); prop2Tex.get(prop2,vid);
      partial_trace_mul_Prop_G_Prop_meson_LIBE<ACC_ZERO,true>(R,prop1,prop2,su3);
      for(int s=0; s<9; s++){
	noeV[s*N_DIMS+dir] += trace_gamma_S<true>(gamma_dir[dir],NOROT,R[s/3][s%3]) - trace_gamma_S<true>(G5,NOROT,R[s/3][s%3]);
      }

      //term x-dir, x-dir, x
      prop1Tex.get<Minus>(prop1,vid,dir); gaugeTex.get<Minus>(su3,dir,vid,dir);
      partial_trace_mul_Prop_G_Prop_meson_LIBE<ACC_ZERO,false>(R,prop1,prop2,su3);
      for(int s=0; s<9; s++){
	noeV[s*N_DIMS+dir] += trace_gamma_S<true>(gamma_dir[dir],NOROT,R[s/3][s%3]) - trace_gamma_S<true>(G5,NOROT,R[s/3][s%3]);
      }
    }    
  }
  
  #pragma unroll
  for(int i = 0; i < site_size; i++)
    noeV[i] *= 0.25;
  
  if(runFT){
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z}; 
    fourier_transform_3D(block2, noeV, shared_cache, site_size, sid3D, source_pos, moms, 0, +1, time_step, tid);
  } else{
    if (sid3D < DGC_localVolume3D)
      for(int i = 0; i < site_size; i++)
	block2[(tid*DGC_localVolume3D + sid3D)*site_size+i] = noeV[i];
  }
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
static void mesons_noe_LIBE_host(ProfileStruct &ps, Float2<FloatC> *result, PLEGMA_Correlator<FloatC> &corr,
			    PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatA>& prop2,
			    PLEGMA_Gauge<FloatG>& gauge){
  
  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  size_t volume = corr.getVolSize()/t_size;
  size_t size = corr.getTotalSize()/t_size*time_step;
  int site_size = corr.getSiteSize();
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("time_step = %d, t_size = %d, maxT = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step, t_size, maxT, ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true)? (size * (ps.tp.grid.x/time_step) ) : size;

  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(Float2<FloatC>) );
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatC>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto propTex2 = toTexture<propTex>(prop2);
  auto gaugetex = toTexture<gaugeTex>(gauge);

  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess || h_partial_block==NULL) goto exit;
  for(int it=0; it < t_size; it+=time_step) {
    int t_step = std::min(t_size-it, time_step);
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*t_step;
    mesons_noe_LIBE_device<FloatC,FloatA, FloatB, FloatG>
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (d_partial_block, *propTex1, *propTex2, *gaugetex, it, t_step, maxT, source, runFT, *moms); 
   
    error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;

    cudaMemcpy(h_partial_block , d_partial_block , (alloc_size/time_step)*t_step*sizeof(Float2<FloatC>) , cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;

    if(runFT==true){
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*t_step; v++)
	for(int i = 0 ; i < site_size; i++) {
	  result[(it*volume+v)*site_size+i] = 0;
	    for(int j = 0 ; j < accumX; j++)
	      result[(it*volume+v)*site_size+i] +=
		h_partial_block[(v*site_size+i)*accumX+j];
	}
    } else {
      for(size_t v = 0 ; v < volume*t_step; v++)
	for(int i = 0 ; i < site_size; i++)
	  result[(it*volume+v)*site_size+i] +=
	    h_partial_block[v*site_size+i];
    }
  }

 exit:
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  cudaFree(d_partial_block);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void mesons_noe_LIBE(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, PLEGMA_Gauge<FloatG>& gauge) {
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  int site_size = 9*N_DIMS;
  if(corr.getSiteSize() != site_size) {
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);
  }

  ProfileStruct ps(HGC_localVolume3D, (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;
  
  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.H_elem();
  
  tuneAndRun( ps, "mesons_noe_LIBE", mesons_noe_LIBE_host<FloatC,FloatA,FloatB,FloatG>,
	      ps, result, corr, prop1, prop2, gauge);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type(corr.H_elem()),
		  MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK\n");
#endif
}

template void mesons_noe_LIBE<float,float,float,float>(PLEGMA_Correlator<float> &corr, PLEGMA_Propagator<float>& prop1, PLEGMA_Propagator<float>& prop2, PLEGMA_Gauge<float>& gauge);
template void mesons_noe_LIBE<double,double,double,double>(PLEGMA_Correlator<double> &corr, PLEGMA_Propagator<double>& prop1, PLEGMA_Propagator<double>& prop2, PLEGMA_Gauge<double>& gauge);

