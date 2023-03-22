#include <PLEGMA_Correlator.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_Fmunu.h>
#include "PLEGMA_kernel_utils.cuh"
#include "PLEGMA_kernel_getSet.cuh"
#include "PLEGMA_gammas.cuh"
#include "PLEGMA_threep.cuh"
#include <cmath>
#include <cfloat>
#include <malloc_quda.h>
using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};


template<typename Float>
__global__ void threep_qgq_kernel(Float2<Float>* block2,
				  prop2<Float> propl, prop2<Float> propr,
				  su3_2<Float> su3_l, su3_2<Float> fmunu, su3_2<Float> su3_r,
				  KernelArr<GAMMAS> listGammas,
				  int it, int time_step, int maxT, int4 source,
				  int signProps, bool runFT, tex_mom_list moms){
  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC->localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC->localVolume3D;
  
  Float2<Float> accum[N_SPINS*N_SPINS]; // max value of gammas
  #pragma unroll
  for(int i = 0; i < N_SPINS*N_SPINS; i++)
    accum[i]=0;

  if (sid3D < DGC->localVolume3D){

    Float2<Float> pp1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<Float> pp2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<Float> u1[N_COLS][N_COLS];
    Float2<Float> u2[N_COLS][N_COLS];
    Float2<Float> u3[N_COLS][N_COLS];
    Float2<Float> R[N_SPINS][N_SPINS];
    propl.get(pp1,vid);
    propr.get(pp2,vid);

    su3_l.get(u1,vid);
    fmunu.get(u2,vid);
    mul_G_G(u3,u1,u2);
    su3_r.get(u2,vid);
    mul_G_G(u1,u3,u2);
    
    partial_trace_mul_Prop_G_Prop<true,ACC_ZERO,false>(R,pp1,pp2,u1);

    for(int iop = 0; iop < listGammas.size; iop++){
      int opId=listGammas.array[iop];
      accum[iop]=((signProps > 0) ? trace_gamma_S<true>(opId,TMP,R) : trace_gamma_S<true>(opId,TMM,R));
    }
  }

  int source_pos[3] = {source.x, source.y, source.z}; 

  if(runFT){
    extern __shared__ int ext_shared_cache[];
    Float2<Float> *shared_cache = (Float2<Float> *) ext_shared_cache;
    fourier_transform_3D(block2, accum, shared_cache, listGammas.size, sid3D, source_pos, moms, 0, +1, time_step, tid);
  } else{
    if (sid3D < DGC->localVolume3D)
      for(int iop = 0; iop < listGammas.size; iop++)
	block2[(tid*DGC->localVolume3D + sid3D)*listGammas.size +iop] = accum[iop];
  }
}


template<typename Float>
static void threep_qgq_host(ProfileStruct &ps, Float2<Float> *result,
				   PLEGMA_Correlator<Float> &corr,
				   PLEGMA_Propagator<Float>& propl, PLEGMA_Propagator<Float>& propr,
				   int signProps, PLEGMA_Su3field<Float>& su3_l,
				   PLEGMA_Fmunu<Float>& Fmunu, std::pair<int,int> munu,
				   PLEGMA_Su3field<Float>& su3_r,
				   std::vector<GAMMAS>& gammas){
  
  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  size_t volume = corr.getVolSize()/t_size;
  size_t size = corr.getTotalSize()/t_size*time_step;
  int site_size = gammas.size();
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();

  KernelArr<GAMMAS> listGammas;
  listGammas.size = gammas.size();
  //cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
  listGammas.array=(GAMMAS*)device_malloc(gammas.size()*sizeof(GAMMAS));
  qudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), qudaMemcpyHostToDevice);

  if(HGC.verbosity > 2)
    if(corr.hasSource())
      printf("time_step = %d, t_size = %d, maxT = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step, t_size, maxT, ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true)? (size * (ps.tp.grid.x/time_step) ) : size;

  Float2<Float> *h_partial_block = NULL;
  Float2<Float> *d_partial_block = NULL;
 // cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(Float2<Float>) );
  d_partial_block=(Float2<Float>*)device_malloc( alloc_size * sizeof(Float2<Float>) );
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<Float>));


  long int shift = ((long int) Fmunu.munuToIndx(munu)) * N_COLS * N_COLS * Fmunu.Total_length();
  su3_2<Float> RFmunu((Float2<Float>*) Fmunu.D_elem()+shift, su3_l.Field_length(), Fmunu.is4D(), false);
  

  //cudaError_t error=cudaPeekAtLastError();
  //if(error != cudaSuccess || h_partial_block==NULL) goto exit;
  for(int it=0; it < t_size; it+=time_step) {
    int t_step = std::min(t_size-it, time_step);
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*t_step;
    threep_qgq_kernel<Float>
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (d_partial_block, toField2<prop2>(propl), toField2<prop2>(propr),
       toField2<su3_2>(su3_l), RFmunu, toField2<su3_2>(su3_r),
       listGammas, it, t_step, maxT,
       source, signProps, runFT, *moms);
//    error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;

    qudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*t_step*sizeof(Float2<Float>) , qudaMemcpyDeviceToHost);
//    error=cudaPeekAtLastError(); if(error != cudaSuccess) goto exit;

    if(runFT==true){
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*t_step; v++)
	for(int i = 0 ; i < site_size; i++) {
	  result[(it*volume+v)*site_size+i] = 0;
	  for(int j = 0 ; j < accumX; j++){
	      result[(it*volume+v)*site_size+i] +=
		h_partial_block[(v*site_size+i)*accumX+j];
	      if(std::isnan(h_partial_block[(v*site_size+i)*accumX+j].x)){
		PLEGMA_error("Got NaN");
	      }
	  }
	}
    } else {
      for(size_t v = 0 ; v < volume*t_step; v++)
	for(int i = 0 ; i < site_size; i++)
	  result[(it*volume+v)*site_size+i] +=
	    h_partial_block[v*site_size+i];
    }
  }

 exit:
  hostFree(h_partial_block, alloc_size*sizeof(Float));
  device_free(d_partial_block);
  device_free(listGammas.array);
}



template<typename Float>
void threep_qgq(PLEGMA_Correlator<Float> &corr,
	   PLEGMA_Propagator<Float>& prop1, PLEGMA_Propagator<Float>& prop2,
	   int signProps, PLEGMA_Su3field<Float>& su3_l,
	   PLEGMA_Fmunu<Float> &Fmunu, std::pair<int,int> munu,
	   PLEGMA_Su3field<Float>& su3_r,
	   std::vector<GAMMAS>& gammas){
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
  if(gammas.size() <= 0)
    PLEGMA_error("Error the container of gamma matrices cannot be zero");
  if(gammas.size() > N_SPINS*N_SPINS)
    PLEGMA_error("Error maximum number of gamma matrices is 16");

  bool runFT = (corr.getCorrSpace() == MOMENTUM_SPACE);
  int site_size = gammas.size();
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  ProfileStruct ps(HGC.localVolume3D, (runFT==true) ? site_size*sizeof(Float2<Float>) : 0);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC.fullComm);
  ps.max_volume = HGC.localVolume3D*maxLocalT;
  ps.tune_globally = true;
  
  Float2<Float> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<Float>));
  else
    result = (Float2<Float> *) corr.H_elem();

  
  tuneAndRun( ps, "threep_qgq", threep_qgq_host<Float>,
	      ps, result, corr, prop1, prop2, signProps, su3_l,
	      Fmunu, munu, su3_r, gammas);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type(corr.H_elem()),
		  MPI_SUM, HGC.spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<Float>));
  }
#else
  PLEGMA_error("You must enable PLEGMA_NUCLEON_3PF_FIX_SINK\n");
#endif
}


template void threep_qgq(PLEGMA_Correlator<float> &corr,
			 PLEGMA_Propagator<float>& prop1, PLEGMA_Propagator<float>& prop2,
			 int signProps, PLEGMA_Su3field<float>& su3_l,
			 PLEGMA_Fmunu<float> &Fmunu, std::pair<int,int> munu,
			 PLEGMA_Su3field<float>& su3_r,
			 std::vector<GAMMAS>& gammas);

template void threep_qgq(PLEGMA_Correlator<double> &corr,
			 PLEGMA_Propagator<double>& prop1, PLEGMA_Propagator<double>& prop2,
			 int signProps, PLEGMA_Su3field<double>& su3_l,
			 PLEGMA_Fmunu<double> &Fmunu, std::pair<int,int> munu,
			 PLEGMA_Su3field<double>& su3_r,
			 std::vector<GAMMAS>& gammas);
