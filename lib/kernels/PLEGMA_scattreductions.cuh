#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas.cuh>

using namespace plegma;
template<typename T>
struct KernelArr {T* array; int size;};

template<typename FloatOut, typename FloatV, typename FloatP, unsigned int N_GAMMAS>
__global__ void V3_kernel( FloatV *Phi, KernelArr<GAMMAS> listGammas,
			   FloatP *S, Float2<FloatOut> *block2,
			   int it, int time_step, int3 source, tex_mom_list moms){

  int grid3D = gridDim.x/time_step; //n_blocks x timeslice
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
  int tid = blockIdx.x/grid3D;
  int vid = sid3D + (it+tid)*DGC_localVolume3D;
  int site_size = N_GAMMAS*N_SPINS*N_COLS;

  //if (vid==0) {  printf("check0\n");}
  
  register Float2<FloatOut> accum[N_GAMMAS*N_SPINS*N_COLS];
  for(int i = 0 ; i <N_GAMMAS*N_SPINS*N_COLS  ; i++){
    accum[i] = 0.;
  }
  
  if (sid3D < DGC_localVolume3D){
    prop2<FloatP> propS(S);
    vector2<FloatV> vectorPhi(Phi);
    
    Float2<FloatP> s[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatV> phi[N_SPINS][N_COLS];
    propS.get(s,vid);
    vectorPhi.get(phi,vid);

    
    const Float2<float> (*g)[4];
    const short int (*gammasIdx)[4][2];
    g = (Float2<float> (*)[4]) plegma::gamma;
    gammasIdx = gammaInd;
    
    if (vid==0) {printf("check1\n");}
    
    #pragma unroll
    for(int i_g = 0 ; i_g < N_GAMMAS; i_g++){
      int gId=listGammas.array[i_g];
      if (vid==0) {printf("check2 - gId=%d\n", gId);}
      #pragma unroll //for loop over nonzero entries
      for(int nz_e = 0 ; nz_e < 4 ; nz_e++){
	int alpha0=gammasIdx[gId][nz_e][0];
	int alpha1=gammasIdx[gId][nz_e][1];
	Float2<FloatOut> factor=g[gId][nz_e];
	if (vid==0) {printf("check3 - n_ze=%d, a0-a1-f %d-%d-%f+i%f\n", nz_e, alpha0, alpha1, factor.x, factor.y);}
        #pragma unroll
	for(int beta = 0 ; beta < N_SPINS ; beta++){
          #pragma unroll
	  for(int a = 0 ; a < N_COLS ; a++){
            #pragma unroll
	    for(int b = 0 ; b < N_COLS ; b++){
	      accum[(i_g*N_SPINS + beta)*N_COLS+b] =
		accum[(i_g*N_SPINS + beta)*N_COLS+b]
		+ conj(phi[alpha0][a])*factor*s[alpha1][beta][a][b];
	    }
	  }
	}
      }
    }
  }
      
  extern __shared__ int ext_shared_cache[];
  Float2<FloatOut> *shared_cache = (Float2<FloatOut> *) ext_shared_cache;
  int source_pos[3] = {source.x, source.y, source.z}; 
  fourier_transform_3D(block2, accum, shared_cache, site_size, sid3D, source_pos, moms, 0, -1, time_step, tid);
    
}

template<typename FloatOut, typename FloatV, typename FloatP>
void V3_kernel_wr( ProfileStruct &ps, FloatV *Phi, KernelArr<GAMMAS> &listGammas,
		FloatP *S, Float2<FloatOut> *block2,
		int it, int time_step, int3 source, tex_mom_list moms){
  dim3 grid = ps.tp.grid;
  grid.x = (grid.x/time_step)*MIN(HGC_localL[3]-it, time_step);

  switch(listGammas.size){
  case(1): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)1><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(2): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)2><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(3): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)3><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(4): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)4><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(5): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)5><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(6): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)6><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(7): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)7><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(8): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)8><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(9): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)9><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(10): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)10><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(11): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)11><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(12): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)12><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(13): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)13><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(14): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)14><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(15): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)15><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  case(16): V3_kernel<FloatOut,FloatV,FloatP,(unsigned int)16><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms); break;
  }
  
}

template<typename FloatOut, typename FloatV, typename FloatP>
static void V3_k_host( ProfileStruct &ps,
		       PLEGMA_ScattCorrelator<FloatOut> &Vout,
		       PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS> &gammas,
		       PLEGMA_Propagator<FloatP> &S, Float2<FloatOut>* result){

  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = Vout.getTotalSize()/HGC_localL[3]*time_step;//N_moms*site_size*time_step
  size_t N_moms = Vout.getVolSize()/HGC_localL[3];//N_moms
  int3 source = Vout.getSource3();
  tex_mom_list moms = Vout.getTexMomList();
  int site_size = Vout.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;
  
  if(HGC_verbosity > 2){
    PLEGMA_printf("time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);
    PLEGMA_printf("size = %d, N_moms = %d, nblockxt = %d\n", size, N_moms, nblockspert);
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
  PLEGMA_printf("site_size= %d\n", listGammas.size*N_SPINS*N_COLS);
  
  for(int it=0; it < HGC_localL[3]; it+=time_step) {
    
    V3_kernel_wr(ps, Phi.D_elem(), listGammas, S.D_elem(), d_partial_block, it,  time_step, source, moms);

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

template<typename FloatOut, typename FloatV, typename FloatP>
static void V3_k(PLEGMA_ScattCorrelator<FloatOut> &Vout,
		 PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS> &Gammas,
		 PLEGMA_Propagator<FloatP> &S){

  int site_size = Gammas.size()*N_SPINS*N_COLS;
  
  if(Vout.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", Vout.getSiteSize(), site_size);

  int shared_size = site_size*sizeof(Float2<FloatOut>);
  PLEGMA_printf("site_size= %d\n", site_size);
  
  Float2<FloatOut> *result = NULL;
  hostMalloc(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  ps.max_volume = HGC_localVolume;

  std::string kerName="V3_k_gammas_";
  for(auto const& G: Gammas) {kerName+=GAMMAS_STR[int(G)];}
  
  tuneAndRun( ps, kerName, V3_k_host<FloatOut, FloatV, FloatP>,
	      ps, Vout, Phi, Gammas, S, result);

  //reduction between spaceComm for the sum of Fourier transformation between nodes
  MPI_Allreduce(result, Vout.getCorr(), Vout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);

  hostFree(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>));
}

