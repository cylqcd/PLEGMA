#include <PLEGMA_Correlator.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>
#include <PLEGMA_gammas.cuh>
#include <PLEGMA_threep.cuh>

using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

template<typename FloatC, typename FloatA, typename FloatB, typename FloatG>
__global__ void threep_threeD_part4_device(Float2<FloatC>* block2,
				     propTex<FloatA> prop1Tex, propTex<FloatB> prop2Tex,
				     gaugeTex<FloatG> gaugeTex, KernelArr<GAMMAS> listGammas,
				     int it, int time_step, int maxT, int4 source,
				     int signProps, bool runFT, tex_mom_list moms,
				     int dir1, int dir2, int dir3, int mu, int nu, int c1, int c2){
  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  bool notZfac = (mu<0) && (nu<0) && (c1<0) && (c2<0);
  
  Float2<FloatC> accum[N_SPINS*N_SPINS];
  #pragma unroll
  for(int i = 0; i < N_SPINS*N_SPINS; i++)
    accum[i]=0;

  if (sid3D < DGC_localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatC> R[N_SPINS][N_SPINS];
    Float2<FloatG> su3_1[N_COLS][N_COLS];
    Float2<FloatG> su3_2[N_COLS][N_COLS];
    Float2<FloatG> su3_3[N_COLS][N_COLS];

    // + term x-dir1, x-dir1, x, x+dir2, x+dir2+dir3
    prop1Tex.get<Minus>(prop1,vid,dir1); gaugeTex.get<Minus>(su3_1,dir1,vid,dir1); gaugeTex.get(su3_2,dir2,vid); gaugeTex.get<Plus>(su3_3,dir3,vid,dir2); prop2Tex.get<PlusPlus>(prop2,vid,dir2,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ZERO_PLUS,false,false,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1, x-dir1, x, x+dir2-dir3^, x+dir2-dir3
    /*prop1Tex.get<Minus>(prop1,vid,dir1);*/ /*gaugeTex.get<Minus>(su3_1,dir1,vid,dir1);*/ /*gaugeTex.get(su3_2,dir2,vid);*/ gaugeTex.get<PlusMinus>(su3_3,dir3,vid,dir2,dir3); prop2Tex.get<PlusMinus>(prop2,vid,dir2,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,false,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1+dir3, x-dir1+dir3, x+dir3, x+dir2^, x+dir2
    prop1Tex.get<MinusPlus>(prop1,vid,dir1,dir3); gaugeTex.get<MinusPlus>(su3_1,dir1,vid,dir1,dir3); gaugeTex.get<Plus>(su3_2,dir2,vid,dir3); gaugeTex.get<Plus>(su3_3,dir3,vid,dir2); prop2Tex.get<Plus>(prop2,vid,dir2);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,false,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1-dir3, x-dir1-dir3, x-dir3, x+dir2-dir3, x+dir2
    prop1Tex.get<MinusMinus>(prop1,vid,dir1,dir3); gaugeTex.get<MinusMinus>(su3_1,dir1,vid,dir1,dir3); gaugeTex.get<Minus>(su3_2,dir2,vid,dir3); gaugeTex.get<PlusMinus>(su3_3,dir3,vid,dir2,dir3); /*prop2Tex.get<Plus>(prop2,vid,dir2);*/
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,false,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1, x-dir1, x-dir2^, x-dir2, x-dir2+dir3
    prop1Tex.get<Minus>(prop1,vid,dir1); gaugeTex.get<Minus>(su3_1,dir1,vid,dir1); gaugeTex.get<Minus>(su3_2,dir2,vid,dir2); gaugeTex.get<Minus>(su3_3,dir3,vid,dir2); prop2Tex.get<MinusPlus>(prop2,vid,dir2,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,true,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1, x-dir1, x-dir2^, x-dir2-dir3^, x-dir2-dir3
    /*prop1Tex.get<Minus>(prop1,vid,dir1);*/ /*gaugeTex.get<Minus>(su3_1,dir1,vid,dir1);*/ /*gaugeTex.get<Minus>(su3_2,dir2,vid,dir2);*/ gaugeTex.get<MinusMinus>(su3_3,dir3,vid,dir2,dir3); prop2Tex.get<MinusMinus>(prop2,vid,dir2,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,true,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1+dir3, x-dir1+dir3, x-dir2+dir3^, x-dir2^, x-dir2
    prop1Tex.get<MinusPlus>(prop1,vid,dir1,dir3); gaugeTex.get<MinusPlus>(su3_1,dir1,vid,dir1,dir3); gaugeTex.get<MinusPlus>(su3_2,dir2,vid,dir2,dir3); gaugeTex.get<Minus>(su3_3,dir3,vid,dir2); prop2Tex.get<Minus>(prop2,vid,dir2);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,true,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1-dir3, x-dir1-dir3, x-dir2-dir3^, x-dir2-dir3, x-dir2
    prop1Tex.get<MinusMinus>(prop1,vid,dir1,dir3); gaugeTex.get<MinusMinus>(su3_1,dir1,vid,dir1,dir3); gaugeTex.get<MinusMinus>(su3_2,dir2,vid,dir2,dir3); gaugeTex.get<MinusMinus>(su3_3,dir3,vid,dir2,dir3); /*prop2Tex.get<Minus>(prop2,vid,dir2);*/
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,true,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1+dir2, x-dir1+dir2, x^, x, x+dir3
    prop1Tex.get<MinusPlus>(prop1,vid,dir1,dir2); gaugeTex.get<MinusPlus>(su3_1,dir1,vid,dir1,dir2); gaugeTex.get(su3_2,dir2,vid); gaugeTex.get(su3_3,dir3,vid); prop2Tex.get<Plus>(prop2,vid,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,true,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1+dir2, x-dir1+dir2, x^, x-dir3^, x-dir3
    /*prop1Tex.get<MinusPlus>(prop1,vid,dir1,dir2);*/ /*gaugeTex.get<MinusPlus>(su3_1,dir1,vid,dir1,dir2);*/ /*gaugeTex.get(su3_2,dir2,vid);*/ gaugeTex.get<Minus>(su3_3,dir3,vid,dir3); prop2Tex.get<Minus>(prop2,vid,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,true,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1-dir2, x-dir1-dir2, x-dir2, x-dir3^, x-dir3
    prop1Tex.get<MinusMinus>(prop1,vid,dir1,dir2); gaugeTex.get<MinusMinus>(su3_1,dir1,vid,dir1,dir2); gaugeTex.get<Minus>(su3_2,dir2,vid,dir2); /*gaugeTex.get<Minus>(su3_3,dir3,vid,dir3);*/ /*prop2Tex.get<Minus>(prop2,vid,dir3);*/
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,false,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1-dir2, x-dir1-dir2, x-dir2, x, x+dir3
    /*prop1Tex.get<MinusMinus>(prop1,vid,dir1,dir2);*/ /*gaugeTex.get<MinusMinus>(su3_1,dir1,vid,dir1,dir2);*/ /*gaugeTex.get<Minus>(su3_2,dir2,vid,dir2);*/ gaugeTex.get(su3_3,dir3,vid); prop2Tex.get<Plus>(prop2,vid,dir3);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,false,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1+dir2+dir3, x-dir1+dir2+dir3, x+dir3^, x^, x
    prop1Tex.get<MinusPlusPlus>(prop1,vid,dir1,dir2,dir3); gaugeTex.get<MinusPlusPlus>(su3_1,dir1,vid,dir1,dir2,dir3); gaugeTex.get<Plus>(su3_2,dir2,vid,dir3); /*gaugeTex.get(su3_3,dir3,vid);*/ prop2Tex.get(prop2,vid);
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,true,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1-dir2+dir3, x-dir1-dir2+dir3, x-dir2+dir3, x^, x
    prop1Tex.get<MinusMinusPlus>(prop1,vid,dir1,dir2,dir3); gaugeTex.get<MinusMinusPlus>(su3_1,dir1,vid,dir1,dir2,dir3); gaugeTex.get<MinusPlus>(su3_2,dir2,vid,dir2,dir3); /*gaugeTex.get(su3_3,dir3,vid);*/ /*prop2Tex.get(prop2,vid);*/
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,false,true>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // - term x-dir1+dir2-dir3, x-dir1+dir2-dir3, x-dir3^, x-dir3, x
    prop1Tex.get<MinusPlusMinus>(prop1,vid,dir1,dir2,dir3); gaugeTex.get<MinusPlusMinus>(su3_1,dir1,vid,dir1,dir2,dir3); gaugeTex.get<Minus>(su3_2,dir2,vid,dir3); gaugeTex.get<Minus>(su3_3,dir3,vid,dir3); /*prop2Tex.get(prop2,vid);*/
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_MINUS,false,true,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // + term x-dir1-dir2-dir3, x-dir1-dir2-dir3, x-dir2-dir3, x-dir3, x
    prop1Tex.get<MinusMinusMinus>(prop1,vid,dir1,dir2,dir3); gaugeTex.get<MinusMinusMinus>(su3_1,dir1,vid,dir1,dir2,dir3); gaugeTex.get<MinusMinus>(su3_2,dir2,vid,dir2,dir3); /*gaugeTex.get<Minus>(su3_3,dir3,vid,dir3);*/ /*prop2Tex.get(prop2,vid);*/
    partial_trace_mul_Prop_G1_G2_G3_Prop<true,ACC_PLUS,false,false,false>(R,prop1,prop2,su3_1,su3_2,su3_3,mu,nu,c1,c2);

    // END REGION
	
    for(int iop = 0; iop < listGammas.size; iop++){
      int opId=listGammas.array[iop];
      if(notZfac) accum[iop] = 0.015625*((signProps > 0) ? trace_gamma_S<true>(opId,TMP,R) : trace_gamma_S<true>(opId,TMM,R));
      else accum[iop] = trace_gamma_S<true>(opId,NOROT,R);
    }
  }    

  int site_size = listGammas.size;
  int source_pos[3] = {source.x, source.y, source.z}; 
  if(runFT){
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    fourier_transform_3D(block2, accum, shared_cache, site_size, sid3D, source_pos, moms, 0, +1, time_step, tid);
  } else{
    if (sid3D < DGC_localVolume3D)
      for(int iop = 0; iop < site_size; iop++)
	block2[(tid*DGC_localVolume3D + sid3D)*site_size+iop] = accum[iop];
  }
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
static void threep_threeD_part4_host(ProfileStruct &ps, Float2<FloatC> *result, PLEGMA_Correlator<FloatC> &corr,
			       PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatA>& prop2,
			       int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac){
  
  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);

  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  size_t volume = corr.getVolSize()/t_size;
  int extra=N_DIMS*(N_DIMS-1)*(N_DIMS-2);
  if(isZfac) extra*=N_SPINS*N_SPINS*N_COLS*N_COLS;
  size_t size = corr.getTotalSize()/extra/t_size*time_step;
  int site_size = corr.getSiteSize()/extra;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();

  KernelArr<GAMMAS> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);

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
    for(int et=0; et < extra; et++) {
      int dir1 = (et/(N_DIMS-1)/(N_DIMS-2)) % N_DIMS;
      int dir2 = (et/(N_DIMS-2)) % (N_DIMS-1);
      int dir3 = et % (N_DIMS-2);
      if(dir2>=dir1) dir2++;
      if(dir3>=dir1){
	dir3++;
	if(dir3>=dir2)
	  dir3++;
      } else if(dir3>=dir2){
	dir3++;
	if(dir3>=dir1)
	  dir3++;
      }

      int mu=-1, nu=-1, c1=-1, c2=-1;
      if(isZfac) {
	int tt = et/N_DIMS/(N_DIMS-1)/(N_DIMS-2);
	mu=tt/N_SPINS/N_COLS/N_COLS;
	nu=(tt/N_COLS/N_COLS)%N_SPINS;
	c1=(tt/N_COLS)%N_COLS;
	c2=tt%N_COLS;
      }	
      int t_step = std::min(t_size-it, time_step);
      dim3 grid = ps.tp.grid;
      grid.x = (grid.x/time_step)*t_step;
      threep_threeD_part4_device<FloatC,FloatA, FloatB, FloatG>
	<<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	(d_partial_block, *propTex1, *propTex2, *gaugetex, listGammas, it, t_step, maxT,
	 source, signProps, runFT, *moms, dir1,dir2,dir3,mu,nu,c1,c2);
      error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;
      
      cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*t_step*sizeof(Float2<FloatC>), cudaMemcpyDeviceToHost);
      error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;
      
      if(runFT==true){
	int accumX = ps.tp.grid.x/time_step;
	for(size_t v = 0 ; v < volume*t_step; v++)
	  for(int i = 0 ; i < site_size; i++) {
	    for(int j = 0 ; j < accumX; j++)
	      result[((it*volume+v)*extra+et)*site_size+i] +=
		h_partial_block[(v*site_size+i)*accumX+j];
	  }
      } else {
	for(size_t v = 0 ; v < volume*t_step; v++)
	  for(int i = 0 ; i < site_size; i++)
	    result[((it*volume+v)*extra+et)*site_size+i] +=
	      h_partial_block[v*site_size+i];
      }
    }
  }
  
 exit:
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  cudaFree(d_partial_block);
  cudaFree(listGammas.array);
}

template<typename FloatC,typename FloatA,typename FloatB,typename FloatG>
void threep_threeD_part4(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2,
		 int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  if(gammas.size() <= 0)
    PLEGMA_error("Error the container of gamma matrices cannot be zero");
  if(gammas.size() > N_SPINS*N_SPINS)
    PLEGMA_error("Error maximum number of gamma matrices is 16");

  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  int site_size = N_DIMS*(N_DIMS-1)*(N_DIMS-2)*gammas.size();
  
  if(isZfac)
    site_size *= N_SPINS*N_SPINS*N_COLS*N_COLS;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  site_size = gammas.size();
  ProfileStruct ps(HGC_localVolume3D, (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;
  
  Float2<FloatC> *result = NULL;
  if(runFT) {
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
    memcpy ( result, corr.H_elem(), corr.getTotalSize()*sizeof(Float2<FloatC>) );
  } else
    result = (Float2<FloatC> *) corr.H_elem();
 
  run( ps, "threep_threeD", threep_threeD_part4_host<FloatC,FloatA,FloatB,FloatG>,
       ps, result, corr, prop1, prop2, signProps, gauge, gammas,isZfac);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type(corr.H_elem()),
		  MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK\n");
#endif
}

template void threep_threeD_part4<float,float,float,float>(PLEGMA_Correlator<float> &corr, PLEGMA_Propagator<float>& prop1, PLEGMA_Propagator<float>& prop2, int signProps, PLEGMA_Gauge<float>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);
template void threep_threeD_part4<double,double,double,double>(PLEGMA_Correlator<double> &corr, PLEGMA_Propagator<double>& prop1, PLEGMA_Propagator<double>& prop2, int signProps, PLEGMA_Gauge<double>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

