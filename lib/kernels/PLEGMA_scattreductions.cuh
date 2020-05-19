#include <PLEGMA_kernel_utils.cuh>

using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

template<typename FloatOut, typename FloatV, typename FloatP>
void V_kernels_wrapper( ProfileStruct &ps, VRED V, Float2<FloatOut> *block2,
			int it, int time_step, int maxT, int4 source, tex_mom_list moms,
			KernelArr<GAMMAS_SCATT> &listGammas,
			vectorTex<FloatV> &Phi, propTex<FloatP>& S1, propTex<FloatP>& S2);

template<typename FloatOut, typename FloatP>
void T_kernels_wrapper( ProfileStruct &ps, TRED T, Float2<FloatOut> *block2,
			int it, int time_step, int maxT, int4 source, tex_mom_list moms,
			KernelArr<GAMMAS_SCATT> &listGammas_i, KernelArr<GAMMAS_SCATT> &listGammas_f,
			propTex<FloatP>& S1, propTex<FloatP>& S2, propTex<FloatP>& S3 );


// +++++++++++++++++++++++++++++++++++++++
// +++++++++++| V reductions |++++++++++++
// +++++++++++++++++++++++++++++++++++++++

template<typename FloatOut, typename FloatV, typename FloatP>
static void V_reductions_host( ProfileStruct &ps, VRED V, PLEGMA_ScattCorrelator<FloatOut> &Vout,
			       Float2<FloatOut>* result, std::vector<GAMMAS_SCATT> &gammas,
			       vectorTex<FloatV> &Phi, propTex<FloatP>& S1, propTex<FloatP>& S2){

  int t_size = Vout.localT(); if(t_size==0) return;
  int maxT = Vout.endT() - Vout.startT(); 
  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = Vout.getTotalSize()/t_size*time_step;//N_moms*site_size*time_step
  size_t N_moms = Vout.getVolSize()/t_size;//N_moms
  int4 source = Vout.getSource(); 
  auto moms = Vout.getTexMomList();
  int site_size = Vout.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;

  //value of some quantities
  if(HGC_verbosity > 2){
    PLEGMA_printf("t_size = %d, maxT = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
    PLEGMA_printf("size = %d, volume = %d, nblockxt = %d\n", size, N_moms, nblockspert);
  }

  //allocate partial_block on host and device
  size_t alloc_size = size * nblockspert; // N_moms*site_size*n_blocks
  Float2<FloatOut> *h_partial_block = NULL;
  Float2<FloatOut> *d_partial_block = NULL;
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    PLEGMA_printf("Error in allocating d_partial_block\n");
    cudaFree(d_partial_block);
    return;
  }

  //allocate list of Gammas that can be passed to the device (std::vector not recognized)
  KernelArr<GAMMAS_SCATT> listGammas;
  listGammas.size = gammas.size();
  cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS_SCATT));
  checkCudaError();
  cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS_SCATT), cudaMemcpyHostToDevice);
  checkCudaError();

  //loop over the bunches of timeslices passed to device
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    ps.tp.grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    
    //call the kernel wrapper
    V_kernels_wrapper<FloatOut, FloatV, FloatP>(ps, V, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, *moms, listGammas, Phi, S1, S2 );
    
    ps.tp.grid.x = grid.x;

    //Syncronize (maybe useles) and look for errors (without stopping)
    cudaDeviceSynchronize();
    error=cudaPeekAtLastError();
    if(error != cudaSuccess) { PLEGMA_printf("Error after V_kernels_wrapper, it=%d\n",it); break;}

    //copy partial summed 3dfourier back to host d_partial -> h_partial (device->host)
    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatOut>), cudaMemcpyDeviceToHost);
    
    error=cudaPeekAtLastError(); 
    if(error != cudaSuccess) { PLEGMA_printf("Error after copying back partial_block, it=%d\n",it); break;}

    //perform intermediate sum between results of different blocks with same timeslice
    for(size_t tslicexmom = 0 ; tslicexmom< N_moms*std::min(t_size-it, time_step); tslicexmom++){
      for(int f = 0 ; f < site_size; f++) {
	result[(it*N_moms+tslicexmom)*site_size + f] = 0;
	for(int j = 0 ; j < nblockspert; j++)
	  result[(it*N_moms+tslicexmom)*site_size + f] += h_partial_block[(tslicexmom*site_size+f)*nblockspert+j];
      }//f loop
    }//tslicexmom loop

  }//it loop

  //free allocated memory
  hostFree(h_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  cudaFree(d_partial_block);
  cudaFree(listGammas.array);
  
}

template<typename FloatOut, typename FloatV, typename FloatP>
static void V_reductions(VRED V, PLEGMA_ScattCorrelator<FloatOut> &Vout,
			 PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS_SCATT> &Gammas,
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
  int myLocalT = Vout.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;
  
  std::string kerName="V_reductions_V"+std::to_string(int(V))+"_gammas_";
  for(auto const& G: Gammas) {kerName+="g";}
      
  auto vectorPhi = toTexture<vectorTex>(Phi);
  auto propS = toTexture<propTex>(S);
  tuneAndRun( ps, kerName, V_reductions_host<FloatOut, FloatV, FloatP>,
	      ps, V, Vout, result, Gammas, *vectorPhi, *propS, *propS );

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Vout.H_elem(), Vout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

  hostFree(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>));
}

template<typename FloatOut, typename FloatV, typename FloatP>
static void V_reductions(VRED V, PLEGMA_ScattCorrelator<FloatOut> &Vout,
			 PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS_SCATT> &Gammas,
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
  int myLocalT = Vout.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  std::string kerName="V_reductions_V"+std::to_string(int(V))+"_gammas_";
  for(auto const& G: Gammas) {kerName+="g";}

  auto vectorPhi = toTexture<vectorTex>(Phi);
  auto propS1 = toTexture<propTex>(S1);
  auto propS2 = toTexture<propTex>(S2);
  tuneAndRun( ps, kerName, V_reductions_host<FloatOut,FloatV,FloatP>,
	      ps, V, Vout, result, Gammas, *vectorPhi, *propS1, *propS2);

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Vout.H_elem(), Vout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

 hostFree(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>));
 
}

// +++++++++++++++++++++++++++++++++++++++
// +++++++++++| T reductions |++++++++++++
// +++++++++++++++++++++++++++++++++++++++

template<typename FloatOut, typename FloatP>
static void T_reductions_host( ProfileStruct &ps, TRED T, PLEGMA_ScattCorrelator<FloatOut> &Tout,
			       Float2<FloatOut>* result, std::vector<GAMMAS_SCATT> &gammas_i,
			       std::vector<GAMMAS_SCATT> &gammas_f,
			       propTex<FloatP>& S1, propTex<FloatP>& S2, propTex<FloatP>& S3 ){

  int t_size = Tout.localT(); if(t_size==0) return;
  int maxT = Tout.endT() - Tout.startT(); 
  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = Tout.getTotalSize()/t_size*time_step;//N_moms*site_size*time_step
  size_t N_moms = Tout.getVolSize()/t_size;//N_moms
  int4 source = Tout.getSource(); 
  auto moms = Tout.getTexMomList();
  int site_size = Tout.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;
  
  if(HGC_verbosity > 2){
    PLEGMA_printf("t_size = %d, maxT = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
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

  KernelArr<GAMMAS_SCATT> listGammas_i, listGammas_f;
  listGammas_i.size = gammas_i.size();
  listGammas_f.size = gammas_f.size();
  cudaMalloc((void**)&listGammas_i.array, gammas_i.size()*sizeof(GAMMAS_SCATT));
  cudaMalloc((void**)&listGammas_f.array, gammas_f.size()*sizeof(GAMMAS_SCATT));
  checkCudaError();
  cudaMemcpy(listGammas_i.array, gammas_i.data(), gammas_i.size()*sizeof(GAMMAS_SCATT), cudaMemcpyHostToDevice);
  cudaMemcpy(listGammas_f.array, gammas_f.data(), gammas_f.size()*sizeof(GAMMAS_SCATT), cudaMemcpyHostToDevice);
  checkCudaError();
  if(HGC_verbosity > 2){
    PLEGMA_printf("site_size= %d\n", listGammas_f.size*listGammas_i.size*N_SPINS*N_SPINS);
  }
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    ps.tp.grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);

    T_kernels_wrapper<FloatOut, FloatP>(ps, T, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, *moms, listGammas_i, listGammas_f, S1, S2, S3 );

    ps.tp.grid.x = grid.x;
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
  cudaFree(listGammas_i.array);
  cudaFree(listGammas_f.array);
  
}


template<typename FloatOut, typename FloatP>
static void T_reductions(TRED T, PLEGMA_ScattCorrelator<FloatOut> &Tout,
			 std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, 
			 PLEGMA_Propagator<FloatP> &S1, PLEGMA_Propagator<FloatP> &S2,
			 PLEGMA_Propagator<FloatP> &S3){

  int site_size = Gammas_i.size()*Gammas_f.size()*N_SPINS*N_SPINS;
  
  if(Tout.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", Tout.getSiteSize(), site_size);

  int shared_size = N_SPINS*sizeof(Float2<FloatOut>);
  PLEGMA_printf("site_size= %d\n", site_size);
  
  Float2<FloatOut> *result = NULL;
  hostMalloc(result, Tout.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = Tout.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;

  std::string kerName="T_reductions_T"+std::to_string(int(T))+"_gammas_i_";
  for(auto const& G: Gammas_i) {kerName+="g";}
  kerName+="_gammas_f_";
  for(auto const& G: Gammas_f) {kerName+="g";}

  auto propS1 = toTexture<propTex>(S1);
  auto propS2 = toTexture<propTex>(S2);
  auto propS3 = toTexture<propTex>(S3);
  tuneAndRun( ps, kerName, T_reductions_host<FloatOut,FloatP>,
	      ps, T, Tout, result, Gammas_i, Gammas_f, *propS1, *propS2, *propS3);

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Tout.H_elem(), Tout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

 hostFree(result, Tout.getTotalSize()*sizeof(Float2<FloatOut>));
 
}
