using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

#include <PLEGMA_scattreductionsT1.cu>
#include <PLEGMA_scattreductionsT2.cu>
#include <PLEGMA_scattreductionsWraps.cu>

template<VRED V,typename FloatOut, typename FloatV, typename ... Args>
static void V_reductions_host( ProfileStruct &ps, PLEGMA_ScattCorrelator<FloatOut> &Vout,
			       Float2<FloatOut>* result, std::vector<GAMMAS> &gammas,
			       FloatV *Phi, Args*... S_fields){

  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = Vout.getTotalSize()/HGC_localL[3]*time_step;//N_moms*site_size*time_step
  size_t N_moms = Vout.getVolSize()/HGC_localL[3];//N_moms
  int3 source = Vout.getSource3();
  tex_mom_list moms = Vout.getTexMomList();
  int site_size = Vout.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;
  
  if(HGC_verbosity > 2){
    PLEGMA_printf("time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
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

  KernelArr<GAMMAS> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
  checkCudaError();
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);
  checkCudaError();
  if(HGC_verbosity > 2)
    PLEGMA_printf("site_size= %d\n", listGammas.size*N_SPINS*N_COLS);

  for(int it=0; it < HGC_localL[3]; it+=time_step) {
    
    V_kernels_wrapper<V,FloatOut, FloatV, Args...>(ps, d_partial_block, it, time_step, source, moms, listGammas, Phi, S_fields... );

    cudaDeviceSynchronize();

    error=cudaPeekAtLastError(); if(error != cudaSuccess) { PLEGMA_printf("ERROR1\n"); break;}

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*MIN(HGC_localL[3]-it, time_step)*sizeof(Float2<FloatOut>), cudaMemcpyDeviceToHost);
    
    error=cudaPeekAtLastError(); if(error != cudaSuccess) { PLEGMA_printf("ERROR2\n"); break;}

    for(size_t tslicexmom = 0 ; tslicexmom< N_moms*MIN(HGC_localL[3]-it, time_step); tslicexmom++){
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

template<VRED V, typename FloatOut, typename FloatV, typename FloatP>
static void V_reductions(PLEGMA_ScattCorrelator<FloatOut> &Vout,
		 PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS> &Gammas,
		 PLEGMA_Propagator<FloatP> &S){

  int site_size = Gammas.size()*N_SPINS*N_COLS;
  
  if(Vout.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", Vout.getSiteSize(), site_size);

  int shared_size = N_SPINS*N_COLS*sizeof(Float2<FloatOut>); //+
  PLEGMA_printf("site_size= %d\n", site_size);
  
  Float2<FloatOut> *result = NULL;
  hostMalloc(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  ps.max_volume = HGC_localVolume;

  std::string kerName="V_reductions_V"+std::to_string(int(V))+"_gammas_";
  for(auto const& G: Gammas) {kerName+="g";}
      
  tuneAndRun( ps, kerName, V_reductions_host<V,FloatOut, FloatV, FloatP>,
	      ps, Vout, result, Gammas, Phi.D_elem(), S.D_elem() );

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Vout.getCorr(), Vout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

  hostFree(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>));
}

template<VRED V, typename FloatOut, typename FloatV, typename FloatP>
static void V_reductions(PLEGMA_ScattCorrelator<FloatOut> &Vout,
		  PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS> &Gammas,
		  PLEGMA_Propagator<FloatP> &S1,  PLEGMA_Propagator<FloatP> &S2){

  int site_size = Gammas.size()*N_SPINS*N_SPINS*N_SPINS*N_COLS;
  
  if(Vout.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", Vout.getSiteSize(), site_size);

  int shared_size = N_SPINS*N_COLS*sizeof(Float2<FloatOut>); //+
  PLEGMA_printf("site_size= %d\n", site_size);
  
  Float2<FloatOut> *result = NULL;
  hostMalloc(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  ps.max_volume = HGC_localVolume;

  std::string kerName="V_reductions_V"+std::to_string(int(V))+"_gammas_";
  for(auto const& G: Gammas) {kerName+="g";}

  tuneAndRun( ps, kerName, V_reductions_host<V,FloatOut,FloatV,FloatP,FloatP>,
	      ps, Vout, result, Gammas, Phi.D_elem(), S1.D_elem(), S2.D_elem());

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Vout.getCorr(), Vout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

 hostFree(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>));
 
}

template<TRED T, typename FloatOut, typename FloatP, typename ...Args>
void T_kernels_wrapper( ProfileStruct &ps, Float2<FloatOut> *block2,
			int it, int time_step, int3 source, tex_mom_list moms,
			KernelArr<GAMMAS> &listGammas_i, KernelArr<GAMMAS> &listGammas_f, FloatP* S1, FloatP* S2, FloatP* S3){
  // if(T==T_1)
  //   T1_kernel_wrapper( ps, block2, it, time_step, source, moms, listGammas_i, listGammas_f, S1, S2, S3 );
  // else if(T==T_2)
  //   T2_kernel_wrapper( ps, block2, it, time_step, source, moms, listGammas_i, listGammas_f, S1, S2, S3 );
  // else
  //   PLEGMA_error("Unrecognized T reduction type\n");
}

template<TRED T,typename FloatOut, typename ... Args>
static void T_reductions_host( ProfileStruct &ps, PLEGMA_ScattCorrelator<FloatOut> &Tout,
			       Float2<FloatOut>* result, std::vector<GAMMAS> &gammas_i, std::vector<GAMMAS> &gammas_f,
			       Args*... S_fields){

  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = Tout.getTotalSize()/HGC_localL[3]*time_step;//N_moms*site_size*time_step
  size_t N_moms = Tout.getVolSize()/HGC_localL[3];//N_moms
  int3 source = Tout.getSource3();
  tex_mom_list moms = Tout.getTexMomList();
  int site_size = Tout.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;
  
  if(HGC_verbosity > 2){
    PLEGMA_printf("time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
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

  KernelArr<GAMMAS> listGammas_i, listGammas_f;
  listGammas_i.size = gammas_i.size();
  listGammas_f.size = gammas_f.size();
  cudaMalloc((void**)&listGammas_i.array, gammas_i.size()*sizeof(GAMMAS));
  cudaMalloc((void**)&listGammas_f.array, gammas_f.size()*sizeof(GAMMAS));
  checkCudaError();
  cudaMemcpy(listGammas_i.array, gammas_i.data(), gammas_i.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);
  cudaMemcpy(listGammas_f.array, gammas_f.data(), gammas_f.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);
  checkCudaError();
  PLEGMA_printf("site_size= %d\n", listGammas_f.size*listGammas_i.size*N_SPINS*N_SPINS);
  PLEGMA_printf("OK till now\n");

  for(int it=0; it < HGC_localL[3]; it+=time_step) {
    
    T_kernels_wrapper<T,FloatOut, Args...>(ps, d_partial_block, it, time_step, source, moms, listGammas_i, listGammas_f, S_fields... );

    cudaDeviceSynchronize();

    error=cudaPeekAtLastError(); if(error != cudaSuccess) { PLEGMA_printf("ERROR1\n"); break;}

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*MIN(HGC_localL[3]-it, time_step)*sizeof(Float2<FloatOut>), cudaMemcpyDeviceToHost);
    
    error=cudaPeekAtLastError(); if(error != cudaSuccess) { PLEGMA_printf("ERROR2\n"); break;}

    for(size_t tslicexmom = 0 ; tslicexmom< N_moms*MIN(HGC_localL[3]-it, time_step); tslicexmom++){
      for(int f = 0 ; f < site_size; f++) {
	      result[(it*N_moms+tslicexmom)*site_size + f] = 0;
	      for(int j = 0 ; j < nblockspert; j++)
	        result[(it*N_moms+tslicexmom)*site_size + f] += h_partial_block[(tslicexmom*site_size+f)*nblockspert+j];
      }
    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  cudaFree(d_partial_block);
  cudaFree(listGammas_i.array);
  cudaFree(listGammas_f.array);
  
}


template<TRED T, typename FloatOut, typename FloatP>
static void T_reductions(PLEGMA_ScattCorrelator<FloatOut> &Tout,
		  std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, 
      PLEGMA_Propagator<FloatP> &S1, PLEGMA_Propagator<FloatP> &S2,
		  PLEGMA_Propagator<FloatP> &S3){

  int site_size = Gammas_i.size()*Gammas_f.size()*N_SPINS*N_SPINS;
  
  if(Tout.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", Tout.getSiteSize(), site_size);

  int shared_size = site_size*sizeof(Float2<FloatOut>);
  PLEGMA_printf("site_size= %d\n", site_size);
  
  Float2<FloatOut> *result = NULL;
  hostMalloc(result, Tout.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  ps.max_volume = HGC_localVolume;

  std::string kerName="T_reductions_T"+std::to_string(int(T))+"_gammas_i_";
  for(auto const& G: Gammas_i) {kerName+="g";}
  kerName+="_gammas_f_";
  for(auto const& G: Gammas_f) {kerName+="g";}

  tuneAndRun( ps, kerName, T_reductions_host<T,FloatOut,FloatP,FloatP,FloatP>,
	      ps, Tout, result, Gammas_i, Gammas_f, S1.D_elem(), S2.D_elem(), S3.D_elem());

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Tout.getCorr(), Tout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

 hostFree(result, Tout.getTotalSize()*sizeof(Float2<FloatOut>));
 
}
