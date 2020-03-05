#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas_scatt.cuh>

using namespace plegma;

template<typename FloatOut, typename FloatPhi>
__global__ void PhixGxPhi_kernel( vectorTex<FloatPhi> vectorPhi0, KernelArr<GAMMAS_SCATT> listGammas,
				  vectorTex<FloatPhi> vectorPhi1,
				  Float2<FloatOut> *d_partial_block, int it, int time_step, int maxT,
				  int4 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  //int site_size = N_SPINS*N_SPINS*N_SPINS*N_COLS;
  
  register Float2<FloatOut> accum[16];
  for(int i = 0 ; i < 16  ; i++){
    accum[i] = 0.;
  }
  

  if (sid3D < DGC_localVolume3D){
    Float2<FloatPhi> phi0[N_SPINS][N_COLS];
    Float2<FloatPhi> phi1[N_SPINS][N_COLS];
    vectorPhi0.get(phi0,vid);
    vectorPhi1.get(phi1,vid);
   
    const Float2<float> (*g)[4];
    const short (*gammasIdx)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma_scatt;
    gammasIdx = gammaInd_scatt;

    //phi0*_{alfa}^{a}(x) x Gamma_{alfa,beta} x phi1_{beta}^{a}(x)
    #pragma unroll
    for( int n_g=0; n_g<16; ++n_g){
      if( n_g < listGammas.size ){
	int gId=listGammas.array[n_g];
	#pragma unroll
	for( int nz_e=0; nz_e<4; ++nz_e){
	  int alfa = gammasIdx[gId][nz_e][0];
	  int beta = gammasIdx[gId][nz_e][1];
	  Float2<FloatOut> factor = g[gId][nz_e];
	  #pragma unroll
	  for(int a=0; a<N_COLS; ++a)
	    accum[n_g] = accum[n_g] + conj(phi0[alfa][a])*factor*phi1[beta][a];
	//****//     
	}
      }
    }
  }

  extern __shared__ int ext_shared_cache[];
  Float2<FloatOut> *shared_cache = (Float2<FloatOut> *) ext_shared_cache;
  int source_pos[3] = {source.x, source.y, source.z}; 

  fourier_transform_3D(d_partial_block, accum, shared_cache, listGammas.size, sid3D, source_pos, moms, 0, -1, time_step, tid);

}



template<typename FloatOut, typename FloatPhi>
static void PhixGxPhi_host( ProfileStruct &ps, PLEGMA_ScattCorrelator<FloatOut> &corr,
			    Float2<FloatOut>* result, std::vector<GAMMAS_SCATT> &gammas,
			    PLEGMA_Vector<FloatPhi>& Phi0, PLEGMA_Vector<FloatPhi>& Phi1){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = corr.getTotalSize()/t_size*time_step;//N_moms*site_size*time_step
  size_t N_moms = corr.getVolSize()/t_size;//N_moms
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = corr.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;
  
  if(HGC_verbosity > 2){
    printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
    PLEGMA_printf("size = %d, volume = %d, nblockxt = %d\n", size, N_moms, nblockspert);
  }
  
  size_t alloc_size = size * nblockspert; // N_moms*site_size*n_blocks

  Float2<FloatOut> *h_partial_block = NULL;
  Float2<FloatOut> *d_partial_block = NULL;
  
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    PLEGMA_printf("ERROR0\n");
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatOut>));

  KernelArr<GAMMAS_SCATT> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS_SCATT));
  checkCudaError();
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS_SCATT), cudaMemcpyHostToDevice);
  checkCudaError();
  if(HGC_verbosity > 2)
    PLEGMA_printf("site_size= %d\n", listGammas.size*N_SPINS*N_COLS);

  auto phiTex0 = toTexture<vectorTex>(Phi0);
  auto phiTex1 = toTexture<vectorTex>(Phi1);
  
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);

    PhixGxPhi_kernel<FloatOut, FloatPhi><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(*phiTex0, listGammas, *phiTex1, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, *moms);

    cudaDeviceSynchronize();

    error=cudaPeekAtLastError(); if(error != cudaSuccess) { PLEGMA_printf("ERROR1\n"); break;}

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatOut>), cudaMemcpyDeviceToHost);
    
    error=cudaPeekAtLastError(); if(error != cudaSuccess) { PLEGMA_printf("ERROR2\n"); break;}

    for(size_t tslicexmom = 0 ; tslicexmom< N_moms*std::min(t_size-it, time_step); tslicexmom++){
      for(int f = 0 ; f < site_size; f++) {
	result[(it*N_moms+tslicexmom)*site_size + f] = 0;
	for(int j = 0 ; j < nblockspert; j++)
	  result[(it*N_moms+tslicexmom)*site_size + f] += h_partial_block[(tslicexmom*site_size+f)*nblockspert+j];
      }
    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  cudaFree(d_partial_block);
  cudaFree(listGammas.array);
  
}

template<typename FloatOut,typename FloatPhi>
static void PhixGxPhi_k(PLEGMA_ScattCorrelator<FloatOut> &corr,
		  PLEGMA_Vector<FloatPhi> &Phi0, std::vector<GAMMAS_SCATT> &Gammas,
		  PLEGMA_Vector<FloatPhi> &Phi1){

  int site_size = Gammas.size();
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = Gammas.size()*sizeof(Float2<FloatOut>); //+
  PLEGMA_printf("site_size= %d\n", site_size);
  
  Float2<FloatOut> *result = NULL;
  hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  std::string kerName="PhixGxPhi_gammas_";
  for(auto const& G: Gammas) {kerName+="g";}

  tuneAndRun( ps, kerName, PhixGxPhi_host<FloatOut,FloatPhi>,
	      ps, corr, result, Gammas, Phi0, Phi1);

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

  hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatOut>));
}
