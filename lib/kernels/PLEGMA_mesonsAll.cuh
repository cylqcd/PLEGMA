#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_mesons.cuh>
using namespace plegma;

const int N_PAIRS=34;
  
template<typename FloatA, typename FloatB, typename FloatC>
__global__ void contract_mesons_all_device( propTex<FloatA> texProp1,
					propTex<FloatB> texProp2,
					Float2<FloatC> *block2,
					int it, int time_step, int maxT, int4 source,
					bool runFT, tex_mom_list moms){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  register Float2<FloatC> accum[N_PAIRS];
  for(int i = 0 ; i < N_PAIRS ; i++){
    accum[i] = 0.;
  }

  if (sid3D < DGC_localVolume3D){
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatB> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    short int beta, zeta, delta, alpha;
    Float2<FloatC> value1, value2;
#pragma unroll
    for(int ip = 0 ; ip < 2 ; ip++){
      // For 1 and g5
#pragma unroll
      for(int is = 0 ; is < N_SPINS*N_SPINS ; is++){
	beta  = mesons_indices[ip][is][0];
        zeta = mesons_indices[ip][is][1];
        delta = mesons_indices[ip][is][2];
        alpha = mesons_indices[ip][is][3];
        FloatC value = mesons_values[ip][is];
#pragma unroll
	for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
	  for(int b = 0 ; b < N_COLS ; b++){
	    accum[ip] = accum[ip] + value * prop1[alpha][beta][a][b] * conj(prop2[delta][zeta][a][b]);
	  }
	}
      }
      // For all pairs of (g_mu,g_nu) & (g5g_mu,g5g_nu)
#pragma unroll
      for(int ip1 = 0 ; ip1 < 4 ; ip1++){
#pragma unroll
	for(int ip2 = 0 ; ip2 < 4 ; ip2++){
#pragma unroll
	  for(int is1 = 0 ; is1 < N_SPINS ; is1++){
#pragma unroll
	    for(int is2 = 0 ; is2 < N_SPINS ; is2++){
	      // \Gamma1 prop1 \Gamma2 (prop2)^\dagger
	      // \Gamma1_{delta,alpha}: sink gamma
	      // \Gamma2_{beta,zeta}: src gamma
	      // alpha, gamma can be chosen to run independently from 0 to 3
	      beta  = gammaInd[1+ip2+5*ip][is2][0];
	      zeta  = gammaInd[1+ip2+5*ip][is2][1];
	      delta = gammaInd[1+ip1+5*ip][is1][0];
	      alpha = gammaInd[1+ip1+5*ip][is1][1];
	      value1 = Float2<FloatC>(plegma::gamma[1+ip1+5*ip][is1][0],plegma::gamma[1+ip1+5*ip][is1][1]);
	      value2 = Float2<FloatC>(plegma::gamma[1+ip2+5*ip][is2][0],plegma::gamma[1+ip2+5*ip][is2][1]);//plegma::gamma[1+ip2+6*ip][is2];
#pragma unroll
	      for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
		for(int b = 0 ; b < N_COLS ; b++){ //minus from value2 cancels overall minus sign of Tr
		  accum[2+ip1*4+ip2+16*ip] = accum[2+ip1*4+ip2+16*ip] + value1 * prop1[alpha][beta][a][b] * value2 * conj(prop2[delta][zeta][a][b]); 
		}
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
    fourier_transform_3D(block2, accum, shared_cache, N_PAIRS, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      for(int ip = 0 ; ip < N_PAIRS ; ip++){
	block2[(tid*DGC_localVolume3D + sid3D)*N_PAIRS + ip] = accum[ip];
      }
  }
}

template<typename FloatA, typename FloatB, typename FloatC>
void contract_mesons_all_host( ProfileStruct &ps,
			   PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2,
			   PLEGMA_Correlator<FloatC>& corr, Float2<FloatC> *result){

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = ps.tp.grid.x*ps.tp.block.x/HGC_localVolume3D;
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int site_size = N_PAIRS;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size * (ps.tp.grid.x/time_step)) : size;

  Float2<FloatC> *h_partial_block = NULL;
  Float2<FloatC> *d_partial_block = NULL;
  cudaMalloc((void**)&d_partial_block, alloc_size*sizeof(Float2<FloatC>));
  // Checking for allocation error. In case we return and let the tuner handle the error.
  cudaError_t error=cudaPeekAtLastError();
  if(error != cudaSuccess) {
    cudaFree(d_partial_block);
    return;
  }
  hostMalloc(h_partial_block, alloc_size*sizeof(Float2<FloatC>));

  auto propTex1 = toTexture<propTex>(prop1);
  auto propTex2 = toTexture<propTex>(prop2);
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    contract_mesons_all_device
      <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
      (*propTex1, *propTex2, d_partial_block, it, std::min(t_size-it, time_step), maxT, source, runFT, *moms);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;

    cudaMemcpy(h_partial_block, d_partial_block, (alloc_size/time_step)*std::min(t_size-it, time_step)*sizeof(Float2<FloatC>), cudaMemcpyDeviceToHost);
    error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
    if(runFT==true) {
      int accumX = ps.tp.grid.x/time_step;
      for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++)
	for(int f = 0 ; f < site_size; f++) {
	  result[(it*volume+v)*site_size+f] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(it*volume+v)*site_size+f] += h_partial_block[(v*site_size+f)*accumX+j];
	}
    } else {
      for(size_t v = 0 ; v < volume; v++)
	for(int f = 0 ; f < site_size; f++) {
	  result[(it*volume+v)*site_size+f] = h_partial_block[v*site_size+f];
	}
    }
  }
  hostFree(h_partial_block, alloc_size*sizeof(FloatC));
  cudaFree(d_partial_block);
}

template<typename FloatA, typename FloatB, typename FloatC>
static void contract_mesons_all(PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2,
			    PLEGMA_Correlator<FloatC>& corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int site_size = N_PAIRS;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? site_size*sizeof(Float2<FloatC>) : 0;

  Float2<FloatC> *result = NULL;
  if(runFT)
    hostMalloc(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  else
    result = (Float2<FloatC> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;
  
  tuneAndRun( ps, "contract_mesons_all", contract_mesons_all_host<FloatA,FloatB,FloatC>,
	      ps, prop1, prop2, corr, result);

  if(runFT) {
    MPI_Allreduce(result, corr.H_elem(), corr.getTotalSize()*2, MPI_Type<FloatC>(), MPI_SUM, HGC_spaceComm);
    hostFree(result, corr.getTotalSize()*sizeof(Float2<FloatC>));
  }
}
