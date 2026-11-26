#include <PLEGMA_kernel_utils.cuh>
#include <utils/PLEGMA_auxiliary.h>
#pragma once
using namespace plegma;
using namespace std;

template<typename Float>
__global__ void baryons_EEE_device( vectorTex<Float> texVec1,
				    vectorTex<Float> texVec2,
				    vectorTex<Float> texVec3,
				    Float2<Float> *block2,
				    int it, int time_step, int maxT, int4 source,
				    bool runFT, tex_mom_list moms, int is){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  register Float2<Float> accum = 0;

  const Float2<float> Cg5[4] = {{0,-1},{0,1},{0,-1},{0,1}};
  const short int Cg5idx[4][2] = {{0,1},{1,0},{2,3},{3,2}};
  
  if (sid3D < DGC_localVolume3D){
    Float2<Float> vec1[N_SPINS][N_COLS];
    Float2<Float> vec2[N_SPINS][N_COLS];
    Float2<Float> vec3[N_SPINS][N_COLS];
    texVec1.get(vec1,vid);
    texVec2.get(vec2,vid);
    texVec3.get(vec3,vid);
    #pragma unroll
    for(int nz = 0 ; nz < N_SPINS ; nz++){
      int mu = Cg5idx[nz][0];
      int nu = Cg5idx[nz][1];
      Float2<Float> val = Cg5[nz];
#pragma unroll
      for(int ei = 0 ; ei < 6 ; ei++){
	int a = eps[ei][0];
	int b = eps[ei][1];
	int c = eps[ei][2];
	
	accum = accum + sgn_eps[ei] * val * vec3[nu][b] * (vec2[mu][a] * vec1[is][c] - vec1[mu][a] * vec2[is][c]);
      }
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<Float> *shared_cache = (Float2<Float> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z}; 
    fourier_transform_3D(block2, &accum, shared_cache, 1, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      block2[tid*DGC_localVolume3D + sid3D] = accum;
  }
}

template<typename Float>
void baryons_EEE_host( ProfileStruct &ps,
		       PLEGMA_Vector<Float>& vec1, PLEGMA_Vector<Float>& vec2, PLEGMA_Vector<Float>& vec3,
		       PLEGMA_Correlator<Float>& corr, Float2<Float> *result){
  
  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  size_t size = corr.getTotalSize()/t_size*time_step;
  size_t volume = corr.getVolSize()/t_size;
  int4 source = corr.getSource();
  auto moms = corr.getTexMomList();
  int full_site_size = corr.getSiteSize();
  const int site_size = N_SPINS;

  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("t_size = %d, maxT = %d, source.w = %d, time_step = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n", t_size, maxT, source.w, time_step,  ps.tp.grid.x, ps.tp.block.x, ps.tp.shared_bytes);

  size_t alloc_size = (runFT==true) ? (size/full_site_size * (ps.tp.grid.x/time_step)) : (size/full_site_size);

  Float2<Float> *h_block[site_size];
  Float2<Float> *d_block[site_size];
  qudaStream_t stream[site_size];
  for(int k=0; k<site_size; k++) {
    d_block[k]=(Float2<Float>)device_malloc(alloc_size*sizeof(Float2<Float>));
    hostMallocPinned(h_block[k], alloc_size*sizeof(Float2<Float>));
    //qudaStreamCreate(stream+k) ;
    // Checking for allocation error. In case we return and let the tuner handle the error.
    cudaError_t error=cudaPeekAtLastError();
    if(error != cudaSuccess) {
      for(int i=0; i<k; i++){
	device_free(d_block[i]);
	hostFreePinned(h_block[i], alloc_size*sizeof(Float));
	//cudaStreamDestroy(stream[i]);
      }
      return;
    }
  }
  
  auto vectorTex1 = toTexture<vectorTex>(vec1);
  auto vectorTex2 = toTexture<vectorTex>(vec2);
  auto vectorTex3 = toTexture<vectorTex>(vec3);
  for(int it=0; it < t_size; it+=time_step) {
    dim3 grid = ps.tp.grid;
    grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
    for(int ip=0; ip < site_size; ip++) {
      //baryons_EEE_device<Float>
      //	<<<grid,ps.tp.block,ps.tp.shared_bytes,stream[ip]>>>
      //	    (*vectorTex1, *vectorTex2, *vectorTex3, d_block[ip], it, std::min(t_size-it, time_step), maxT, source, runFT, *moms, ip);
      qudaLaunchKernel(
           baryons_EEE_device<Float>,          // kernel
           grid,                               // dim3 grid
           ps.tp.block,                        // dim3 block
           ps.tp.shared_bytes,                 // shared memory
           stream[ip],                         // qudaStream_t
           *vectorTex1, *vectorTex2, *vectorTex3,
           d_block[ip], it,
           std::min(t_size-it, time_step),
           maxT, source, runFT, *moms, ip
      );
      cudaError_t error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      qudaMemcpyAsync(h_block[ip], d_block[ip], alloc_size*sizeof(Float2<Float>), qudaMemcpyDeviceToHost, stream[ip]);
    }
    cudaError_t error=cudaPeekAtLastError(); if(error != cudaSuccess) break;
      
    for(int ip=0; ip < site_size; ip++) {
      qudaStreamSynchronize(stream[ip]);
      if(runFT==true) {
	int accumX = ps.tp.grid.x/time_step;
	for(size_t v = 0 ; v < volume*std::min(t_size-it, time_step); v++) {
	  result[(it*volume+v)*site_size+ip] = 0;
	  for(int j = 0 ; j < accumX; j++)
	    result[(it*volume+v)*site_size+ip] += h_block[ip][v*accumX+j];
	}
      } else {
	  for(size_t v = 0 ; v < volume; v++)
	    result[(it*volume+v)*site_size+ip] = h_block[ip][v];
      }
    }
  }
  for(int k=0; k<site_size; k++) {
    hostFreePinned(h_block[k], alloc_size*sizeof(Float));
    device_free(d_block[k]);
    //qudaStreamDestroy(stream[k]);
  }
}


template<typename Float>
static void contract_baryons_EEE(Float* evecs, Float* evals, int nvecs, size_t vec_size, bool dev_ptr,
			PLEGMA_Correlator<Float>& corr){
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  assert(runFT);
  int site_size = 16*2;
  
  if(corr.getSiteSize() != site_size)
    PLEGMA_error("Correlator siteSize do not match: %d != %d\n", corr.getSiteSize(), site_size);

  int shared_size = (runFT==true) ? sizeof(Float2<Float>) : 0;

  Float2<Float> *result = NULL;
  Float2<Float> *finalE = NULL;
  if(runFT){
    hostMalloc(result, corr.getTotalSize()/8*sizeof(Float2<Float>));
    hostMalloc(finalE, corr.getTotalSize()/8*HGC_nProc[DIM_T]*sizeof(Float2<Float>));
  } else
    result = (Float2<Float> *) corr.H_elem();

  ProfileStruct ps(HGC_localVolume3D, shared_size);
  int myLocalT = corr.localT();
  int maxLocalT = myLocalT;
  MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
  ps.max_volume = HGC_localVolume3D*maxLocalT;
  ps.tune_globally = true;


  memset(corr.H_elem(),0,corr.getTotalSize()*2*sizeof(Float));
  
  const int ils = dev_ptr ? 1 : 4;
  PLEGMA_Vector<Float> vec1(dev_ptr ? NONE : DEVICE);
  PLEGMA_Vector<Float> vec2(dev_ptr ? NONE : DEVICE);
  std::vector<PLEGMA_Vector<Float>*> vec3;
  size_t vec_bytes = vec_size*sizeof(Float);

  qudaStream_t stream[ils];
  for(int k=0; k<ils; k++) {
    vec3.push_back(new PLEGMA_Vector<Float>(dev_ptr ? NONE : DEVICE));
    //cudaStreamCreate(stream+k) ;
  }
  
  for(int h=0; h<nvecs; h++) {
    PLEGMA_printf("### Contractions for exact_exact vector %d, %s\n", h, getDateAndTime().c_str());
    
    if(dev_ptr) {
      vec1.D_elem(evecs+h*vec_size);
    } else {
      qudaMemcpy(vec1.D_elem(), evecs+h*vec_size, vec_bytes, qudaMemcpyHostToDevice);
      auto error= qudaGetLastError();
      if(error != QUDA_SUCCESS) {
        errorQuda("Failed to copy from host to device %s\n", qudaGetLastErrorString().c_str());
        return;
      }
    }
    
    for(int i=0; i<h; i++) {
      for(int k=0; k<ils and k<nvecs; k++) {
	if(dev_ptr) {
	  vec3[k]->D_elem(evecs+k*vec_size);
	} else {
	  qudaMemcpyAsync(vec3[k]->D_elem(), evecs+k*vec_size, vec_bytes, qudaMemcpyHostToDevice, stream[k]);
	}
      }
    
      if(dev_ptr) {
	vec2.D_elem(evecs+i*vec_size);
      } else {
	qudaMemcpy(vec2.D_elem(), evecs+i*vec_size, vec_bytes, qudaMemcpyHostToDevice);
        auto error= qudaGetLastError();
        if(error != QUDA_SUCCESS) {
          errorQuda("Failed to copy from host to device %s\n", qudaGetLastErrorString().c_str());
          return;
        }
	
      }
    
      for(int j=0; j<nvecs; j++) {
	int k = j % ils;
	if(not dev_ptr) {
	  qudaStreamSynchronize(stream[k]);
	  
	}
	tuneAndRun( ps, "baryons_EEE", baryons_EEE_host<Float>,
		    ps, vec1, vec2, *vec3[k], corr, result);
	// Start copying next vector to use
	if(j+ils<nvecs) {
	  if(dev_ptr) {
	    vec3[k]->D_elem(evecs+(j+ils)*vec_size);
	  } else {
	    qudaMemcpyAsync(vec3[k]->D_elem(), evecs+(j+ils)*vec_size, vec_bytes, qudaMemcpyHostToDevice, stream[k]);
	  }
	}

	if(runFT) {
	  MPI_Allreduce(MPI_IN_PLACE, result, corr.getTotalSize()/8*2, MPI_Type<Float>(), MPI_SUM, HGC_spaceComm);
	}

	// Convolute and average
	// we got in result Eijk(lt,p,g) for local time (lt)
	MPI_Allgather(result, corr.getTotalSize()/8*2, MPI_Type<Float>(),
		      finalE, corr.getTotalSize()/8*2, MPI_Type<Float>(), HGC_timeComm);

	std::complex<Float>* out = (complex<Float>*) corr.H_elem();
	std::complex<Float>* in = (complex<Float>*) finalE;
	std::complex<Float>* evs = (complex<Float>*) evals;
	int t_size = corr.localT();
	size_t volume = corr.getVolSize()/t_size;
	size_t size = corr.getTotalSize()/2;

	std::complex<Float> eval = 1;
	eval /= (evs[h]*evs[i]*conj(evs[j]));
	// const int g5Idx[4] = {2,3,0,1}; // (is+2)%4
	for(int t0=0; t0<HGC_totalL[DIM_T]; t0++) {
	  for(int t1=0; t1<t_size; t1++) {
	    int t0pt = (t0+t1+HGC_procPosition[DIM_T] * HGC_localL[DIM_T])%HGC_totalL[DIM_T];
	    for(int v=0; v<volume; v++)
	      for(int is=0; is<site_size/2; is++) {
		complex<Float> tmp = in[(t0*volume+v)*4+is/4]*conj(in[(t0pt*volume+v)*4+(is+2)%4]);
		out[0*size+(t1*volume+v)*site_size/2+is] += eval * tmp;
		out[1*size+(t1*volume+v)*site_size/2+is] += conj(eval) * tmp;
	      }
	  }
	}
	
      }
    }
  }
  
  hostFree(result, corr.getTotalSize()/8*sizeof(Float2<Float>));
  hostFree(finalE, corr.getTotalSize()/8*HGC_nProc[DIM_T]*sizeof(Float2<Float>));
  for(int k=0; k<ils; k++) {
    delete vec3[k];
//    cudaStreamDestroy(stream[k]);
  }
}
