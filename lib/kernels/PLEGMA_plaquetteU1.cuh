#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_tuner.cuh>
using namespace plegma;

template<typename Float, typename FloatG>
static __global__ void calculatePlaquetteU1_device(gaugeU1Tex<FloatG> gaugeU1Tex, Float *partial_plaq, float phase) {
  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  
  if (sid < gaugeU1Tex.volume()) {
    Float2<FloatG> G1, G2, G3, G4;    
    Float trace = 0.;

    // Loop over xy, xz, xt, yz, yt, zt
    #pragma unroll
    for(int dir1=0; dir1<N_DIMS-1; dir1++) {
      #pragma unroll
      for(int dir2=dir1+1; dir2<N_DIMS; dir2++) {
	// term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
	gaugeU1Tex.get(G1,dir1,sid);
	gaugeU1Tex.get<Plus>(G2,dir2,sid,dir1);
	gaugeU1Tex.get<Plus>(G3,dir1,sid,dir2);
	gaugeU1Tex.get(G4,dir2,sid);

	trace += cos(phase*(G1.x+G2.x-G3.x-G4.x));
      }
    }
    shared_cache[cacheIndex] = trace;
  } else {
    shared_cache[cacheIndex] = 0.;
  }
  reduce(shared_cache, 1);

  // now on the first element of the shared memory we have the reduction of block threads
  if(cacheIndex == 0 && partial_plaq!=NULL)
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory
}

template<typename Float, typename FloatG>
static void calculatePlaquetteU1_host(ProfileStruct& ps, gaugeU1Tex<FloatG> gaugeU1Tex, Float& plaquette, Float phase){

  Float *d_partial_plaq = NULL;
  int gridDimX = ps.tp.grid.x;
  cudaMalloc((void**)&d_partial_plaq, gridDimX * sizeof(Float));
  calculatePlaquetteU1_device<<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(gaugeU1Tex, d_partial_plaq, phase);

  Float *h_partial_plaq = NULL;
  hostMalloc(h_partial_plaq, gridDimX * sizeof(Float) );
  if(h_partial_plaq == NULL) PLEGMA_error("Error allocate memory for host partial plaq");
  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDimX * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_plaq);
  //checkCudaError();

  plaquette = 0.;
  for(int i = 0 ; i < gridDimX ; i++)
    plaquette += h_partial_plaq[i];
  hostFree(h_partial_plaq, gridDimX * sizeof(Float) );
}

template<typename Float, typename FloatG>
static Float calculatePlaquetteU1(gaugeU1Tex<FloatG> gaugeU1Tex, Float phase){

  assert(gaugeU1Tex.is4D); // TODO: For 3D we should not compute the plaquette in T
  ProfileStruct ps(gaugeU1Tex.volume(),sizeof(Float));
  Float plaquette;
  tuneAndRun(ps, "calculatePlaquetteU1", calculatePlaquetteU1_host<Float,FloatG>, ps, gaugeU1Tex, plaquette, phase);

  Float globalPlaquetteU1 = 0.;
  MPI_Allreduce(&plaquette , &globalPlaquetteU1 , 1 , MPI_Type(plaquette) , MPI_SUM , HGC_fullComm);  
  return globalPlaquetteU1/(HGC_totalVolume*6);
}
