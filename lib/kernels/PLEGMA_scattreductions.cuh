#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
template<typename T>

// struct KernelArr {T* array; int size;};
// KernelArr<GAMMAS> listGammas;
//   listGammas.size = gammas.size();
//   cudaMalloc((void**)&listGammas.array, gammas.size()*sizeof(GAMMAS));
//   checkCudaError();
//   cudaMemcpy(listGammas.array, gammas.data(), gammas.size()*sizeof(GAMMAS), cudaMemcpyHostToDevice);
//   checkCudaError();

// template<typename FloatOut, typename FloatV, typename FloatP>
// __global__ void V3_kernel( FloatV *Phi, /*Passgammastodevicesomehow*/ vector<GAMMAS> Gammas,
// 			   FloatP *S, Float2<FloatOut> *block2, int site_size,
// 			   int it, int time_step, int3 source, tex_mom_list moms){

//   int grid3D = gridDim.x/time_step; //n_blocks x timeslice
//   int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;//id of thread
//   int tid = blockIdx.x/grid3D;
//   int vid = sid3D + (it+tid)*DGC_localVolume3D;
    
//   register Float2<FloatC> accum[2*N_MESONS];
//   for(int i = 0 ; i < 2*N_MESONS ; i++){
//     accum[i] = 0.;
//   }

//   if (sid3D < DGC_localVolume3D){
//     prop2<FloatP> propS(S);
//     vector2<FloatV> vectorPhi(Phi);
    
//     Float2<FloatP> s[N_SPINS][N_SPINS][N_COLS][N_COLS];
//     Float2<FloatV> phi[N_SPINS][N_COLS];
//     propS.get(s,vid);
//     vectorPhi.get(phi,vid);

//     for(int i_g = 0 ; i_g < n_gammas ; i_g++){
//       #pragma unroll
//       for(int i_ss = 0 ; i_ss < N_SPINS*N_SPINS ; i_ss++){
// 	int alpha=i_ss/N_SPINS;
// 	int beta=i_ss%N_SPINS;
//         #pragma unroll
// 	for(int a = 0 ; a < N_COLS ; a++){
//           #pragma unroll
// 	  for(int b = 0 ; b < N_COLS ; b++){
// 	    accum[(i_g*N_SPINS + beta)*N_COLS+b] = [(i_g*N_SPINS + beta)*N_COLS+b] + gamma[i_g][alpha][beta] * prop1[alpha][beta][a][b] * conj(prop1[delta][gamma][a][b]);
// 	  }
// 	}
//       }
//     }
//   }

  
//   if(runFT) {
//     extern __shared__ int ext_shared_cache[];
//     Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
//     int source_pos[3] = {source.x, source.y, source.z}; 
//     fourier_transform_3D(block2, accum, shared_cache, 2*N_MESONS, sid3D, source_pos, moms, 0, -1, time_step, tid);
//   } else {
//     if(block2 != NULL)
//       for(int ip = 0 ; ip < 2*N_MESONS ; ip++){
// 	block2[(tid*DGC_localVolume3D + sid3D)*2*N_MESONS + ip] = accum[ip];
//       }
//   }
// }

template<typename FloatOut, typename FloatV, typename FloatP>
static void V3_k_host( ProfileStruct &ps,
		       PLEGMA_ScattCorrelator<FloatOut> &Vout,
		       PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS> &Gammas,
		       PLEGMA_Propagator<FloatP> &S, Float2<FloatOut>* result){

  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;//size of bunch of timeslices passed to the device
  size_t size = Vout.getTotalSize()/HGC_localL[3]*time_step;//N_moms*site_size*time_step
  size_t volume = Vout.getVolSize()/HGC_localL[3];//N_moms
  int3 source = Vout.getSource3();
  tex_mom_list moms = Vout.getTexMomList();
  int site_size = Vout.getSiteSize();
  int nblockspert = ps.tp.grid.x/time_step;
  
  if(HGC_verbosity > 2)
    PLEGMA_printf("time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = size * nblockspert; // N_moms*site_size*n_blocks

  Float2<FloatOut> *h_partial_block = NULL;
  Float2<FloatOut> *d_partial_block = NULL;
  
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatOut>));
  
  for(int it=0; it < HGC_localL[3]; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*MIN(HGC_localL[3]-it, time_step);
    V3_kernel<<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (,, d_partial_block, it, MIN(HGC_localL[3]-it, time_step), source, moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*MIN(HGC_localL[3]-it, time_step)*sizeof(Float2<FloatOut>), cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      

    for(size_t v = 0 ; v < volume*MIN(HGC_localL[3]-it, time_step); v++)
      for(int f = 0 ; f < site_size; f++) {
	result[(f*HGC_localL[3] + it)*volume+v] = 0;
	for(int j = 0 ; j < nblockspert; j++)
	  result[(f*HGC_localL[3] + it)*volume+v] += h_partial_block[(v*site_size+f)*accumXnblockspert+j];
      }
    
  }
  hostFree(h_partial_block, alloc_size*sizeof(FloatOut));
  cudaFree(d_partial_block);
  
}

template<typename FloatOut, typename FloatV, typename FloatP>
static void V3_k(PLEGMA_ScattCorrelator<FloatOut> &Vout,
		 PLEGMA_Vector<FloatV> &Phi, std::vector<GAMMAS> &Gammas,
		 PLEGMA_Propagator<FloatP> &S){

  int site_size = Gammas.size()*N_SPINS*N_COLS;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", Vout.getSiteSize(), site_size);

  int shared_size = site_size*sizeof(Float2<FloatOut>);

  Float2<FloatOut> *result = NULL;
  hostMalloc(result, Vout.getTotalSize()*sizeof(Float2<FloatOut>)); //N.B N_moms*Tlocal*site_size

  //allocation of a number of threads multiple of local3DVolume. the profiler will decide how much.
  ProfileStruct ps(HGC_localVolume3D, shared_size);
  ps.max_volume = HGC_localVolume;
  
  tuneAndRun( ps, "V3_k", V3_k_host<FloatOut, FloatV, FloatP>,
	      ps, Vout, Phi, Gammas, S);

  //reduction between spaceComm for the sum of Fourier transformaton between nodes
  MPI_Allreduce(result, Vout.getCorr(), Vout.getTotalSize()*2, MPI_Type<FloatOut>(), MPI_SUM, HGC_spaceComm);
}

