#include "PLEGMA_kernel_utils.cuh"
#include "PLEGMA_baryons_udsc.cuh"

template<typename FloatA, typename FloatC>
__global__ void create_prop_product(genericTex<FloatC> *propProd,
				    propTex<FloatA> texProp1, propTex<FloatA> texProp2, propTex<FloatA> texProp3,
				    int it, int time_step, int maxT, int4 source){
  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  
  if (sid3D < DGC_localVolume3D){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop3[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    texProp3.get(prop3,vid);
    int mu[6];
    for(mu[0] = 0 ; mu[0] < N_SPINS ; mu[0]++)
      for(mu[1] = 0 ; mu[1] < N_SPINS ; mu[1]++)
	for(mu[2] = 0 ; mu[2] < N_SPINS ; mu[2]++)
	  for(mu[3] = 0 ; mu[3] < N_SPINS ; mu[3]++)
	    for(mu[4] = 0 ; mu[4] < N_SPINS ; mu[4]++)
	      for(mu[5] = 0 ; mu[5] < N_SPINS ; mu[5]++) {
		Float2<FloatC> accum = 0;
#pragma unroll
		for(int cc1 = 0 ; cc1 < 6 ; cc1++){
		  int a = eps[cc1][0];
		  int b = eps[cc1][1];
		  int c = eps[cc1][2];
#pragma unroll
		  for(int cc2 = 0 ; cc2 < 6 ; cc2++){
		    int a1 = eps[cc2][0];
		    int b1 = eps[cc2][1];
		    int c1 = eps[cc2][2];
		    FloatC factor = sgn_eps[cc1] * sgn_eps[cc2];
		    accum += factor *
		      prop1[mu[0]][mu[1]][a][a1] * prop2[mu[2]][mu[3]][b][b1] * prop3[mu[4]][mu[5]][c][c1];
		  }
		}
		int i = 0;
#pragma unroll
		for(int j = 0; j < 6; j++)
		  i=i*N_SPINS+mu[j];
		propProd[tid].set(i,sid3D,accum);
	      }
  }
}

template<typename FloatC>
__global__ void contract_prop_prod(genericTex<FloatC> *texPropProd, Float2<FloatC>* block,
				   int size, short *idxs, Float2<float>*vals,
				   int4 source, bool runFT, tex_mom_list moms, int it, int time_step, int maxT){

  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  Float2<FloatC> accum=0;

  if (sid3D < DGC_localVolume3D){ // I work only on the spatial volume
    for(int i = 0; i < size; i++) {
      int mu = 0;
#pragma unroll
      for(int j = 0; j < 6; j++)
	mu=mu*N_SPINS+idxs[i*6+j];
      accum += texPropProd[tid].get(mu, sid3D)*((Float2<FloatC>) vals[i]);
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block, &accum, shared_cache, 1, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if(block!=NULL)
      block[tid*DGC_localVolume3D + sid3D] = accum;
  }  
}

template<typename FloatA, typename FloatC>
__global__ void contract_props(propTex<FloatA> texProp1, propTex<FloatA> texProp2, propTex<FloatA> texProp3,
			       Float2<FloatC>* block, int size, short *idxs, Float2<float>*vals,
			       int4 source, bool runFT, tex_mom_list moms, int it, int time_step, int maxT){
  
  int grid3D = gridDim.x/time_step;
  int sid3D = (blockIdx.x % grid3D)*blockDim.x + threadIdx.x;
  int tid = blockIdx.x/grid3D;
  // this takes into account the case where the source is in the local lattice
  // and we need to start from it when we go over maxT
  int t=it+tid; if(t>=maxT) t=(source.w%DGC_localL[DIM_T])+t-maxT;
  int vid = sid3D + t*DGC_localVolume3D;
  Float2<FloatC> accum=0;

  if (sid3D < DGC_localVolume3D){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop3[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    texProp3.get(prop3,vid);
    for(int i = 0; i < size; i++) {
      short *mu = idxs + i*6;
#pragma unroll
      for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	int a = eps[cc1][0];
	int b = eps[cc1][1];
	int c = eps[cc1][2];
#pragma unroll
	for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	  int a1 = eps[cc2][0];
	  int b1 = eps[cc2][1];
	  int c1 = eps[cc2][2];
	  Float2<FloatC> factor = sgn_eps[cc1] * sgn_eps[cc2] * vals[i];
	  accum += factor *
	    prop1[mu[0]][mu[1]][a][a1] * prop2[mu[2]][mu[3]][b][b1] * prop3[mu[4]][mu[5]][c][c1];
	}
      }
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block, &accum, shared_cache, 1, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      block[tid*DGC_localVolume3D + sid3D] = accum;
  }  
}

template<typename FloatA, typename FloatC>
void contract_baryons_udsc_host(ProfileStruct &ps,
				PLEGMA_Propagator<FloatA>**props,
				PLEGMA_Correlator<FloatC> &corr,
				Float2<FloatC> *result, int i) {

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x, ps.tp.block.x);

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int4 source = corr.getSource();
  size_t volume3D = corr.getVolSize()/t_size;
  size_t volume = volume3D*time_step;
  auto moms = corr.getTexMomList();
  
  if(HGC_verbosity > 2)
    if(corr.hasSource())
      printf("time_step = %d, ps.tp.aux.x = %d, ps.tp.grid.x = %d, ps.tp.block.x = %d, ps.tp.shared_bytes = %d\n",
	      time_step,      ps.tp.aux.x,      ps.tp.grid.x,      ps.tp.block.x,      ps.tp.shared_bytes);
  
  auto propTex1 = toTexture<propTex>(*(props[0]));
  auto propTex2 = toTexture<propTex>(*(props[1]));
  auto propTex3 = toTexture<propTex>(*(props[2]));

  Float2<FloatC> *h_partial_block = NULL;        
  Float2<FloatC> *d_partial_block = NULL;
  size_t alloc_size = (runFT==true) ? (volume * (ps.tp.grid.x/time_step)):volume;
  hostMalloc(h_partial_block, alloc_size * sizeof(Float2<FloatC>));
  d_partial_block=(Float2<FloatC>*)device_malloc(alloc_size * sizeof(Float2<FloatC>) );
  
  short *idxs;
  Float2<float> *vals;
  int size = 0;
  for(int j=0; j<BP_prop_prods_count[i].size(); j++)
    size += BP_prop_prods_count[i][j];
  idxs=(short*)device_malloc(6*size*sizeof(short));
  vals=(Float2<float>*)device_malloc(size*sizeof(Float2<float>));
  int shift = 0;
  for(int j=0; j<BP_prop_prods_count[i].size(); j++) {
    qudaMemcpy(idxs+6*shift, BP_prop_prods_idxs[i][j], 6*BP_prop_prods_count[i][j]*sizeof(short), qudaMemcpyHostToDevice);
    qudaMemcpy(vals+shift, BP_prop_prods_vals[i][j], BP_prop_prods_count[i][j]*sizeof(Float2<float>), qudaMemcpyHostToDevice);
    shift += BP_prop_prods_count[i][j];
  }
    
  // in case of ps.tp.aux == 1 we create a propProd which has open indeces. Otherwise we contract directly the props
  std::vector<std::shared_ptr<genericTex<FloatC>>> holder;
  PLEGMA_Field3D<FloatC> *propProd[time_step];
  genericTex<FloatC> *texPropProd = NULL;
  if (ps.tp.aux.x == 2) {
    texPropProd=(genericTex<FloatC>*)device_malloc( time_step * sizeof(genericTex<FloatC>) );
    for(int t=0; t<time_step; t++) {
      propProd[t] = new PLEGMA_Field3D<FloatC>(DEVICE, N_SPINS*N_SPINS*N_SPINS*N_SPINS*N_SPINS*N_SPINS, NO_GHOSTS, false, false);
    }
    //cudaError_t error=cudaPeekAtLastError();
    //if(error != cudaSuccess) { goto exit; }
    for(int t=0; t<time_step; t++) {
      holder.push_back(toTexture<genericTex>(*(propProd[t])));
      qudaMemcpy(texPropProd+t, holder.back().get(), sizeof(genericTex<FloatC>), qudaMemcpyHostToDevice);
    }
  } else {
    //cudaError_t error=cudaPeekAtLastError();
    //if(error != cudaSuccess) { goto exit; }
  }

  for(int it=0; it < t_size; it+=time_step) {
    
    if(ps.tp.aux.x == 2) {
      // Would be nice to run another tuner here but not possible right now with quda_tune. 
      //ProfileStruct ps2((ps.volume/time_step)*std::min(t_size-it, time_step),0);
      //tuneAndRun(ps2, "create_prop_product", create_prop_product<FloatA,FloatC>, propProd2, props[0], props[1], props[2], it, std::min(t_size-it, time_step)); 
      dim3 grid = ps.tp.grid;
      grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
      create_prop_product
	<<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	(texPropProd, *propTex1, *propTex2, *propTex3, it, std::min(t_size-it, time_step), maxT, source);
    }
    
    shift = 0;
    for(int j=0; j<BP_prop_prods_count[i].size(); j++) {
      dim3 grid = ps.tp.grid;
      grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
      if(ps.tp.aux.x == 2) {
	contract_prop_prod
	  <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	  (texPropProd, d_partial_block, BP_prop_prods_count[i][j], idxs+6*shift, vals+shift,
	   source, runFT, *moms, it, std::min(t_size-it, time_step), maxT);
      } else {
	contract_props
	  <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	  (*propTex1, *propTex2, *propTex3, d_partial_block, BP_prop_prods_count[i][j], idxs+6*shift, vals+shift,
	   source, runFT, *moms, it, std::min(t_size-it, time_step), maxT);
      }
      qudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(Float2<FloatC>), qudaMemcpyDeviceToHost);
      if(runFT==true){
	int accumX = ps.tp.grid.x/time_step;
	Float2<FloatC> *reduction = result + (j*t_size + it)*volume3D;
	for(size_t k = 0 ; k < volume3D*std::min(t_size-it, time_step); k++) {
	  reduction[k] = 0;
	  for(int l = 0 ; l < accumX; l++) {
	    reduction[k] += h_partial_block[k*accumX + l];
	  }
	}
      } else {
	for(size_t k = 0 ; k < volume3D*std::min(t_size-it, time_step); k++)
	  result[(j*t_size + it)*volume3D + k] = h_partial_block[k];    
      }
      shift += BP_prop_prods_count[i][j];
    }
  }
 exit:
  hostFree(h_partial_block, alloc_size*sizeof(Float2<FloatC>));
  device_free(d_partial_block); d_partial_block=NULL;
  
  device_free(idxs);
  device_free(vals);
  
  if (ps.tp.aux.x == 2) {
    device_free(texPropProd);
    holder.clear();
    for(int t=0; t<time_step; t++) {
      delete propProd[t];
    }
  }
}

template<typename FloatA, typename FloatC>
void contract_baryons_udsc(PLEGMA_Propagator<FloatA>& propUP, PLEGMA_Propagator<FloatA>& propDN,
			   PLEGMA_Propagator<FloatA>& propST, PLEGMA_Propagator<FloatA>& propCH,
			   PLEGMA_Correlator<FloatC> &corr, std::vector<int> &todo){

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int shift = 0;
  for(auto i: todo) {
    Float2<FloatC> *result = NULL;
    if(runFT)
      hostMalloc(result, BP_prop_prods_count[i].size()*corr.getVolSize()*sizeof(Float2<FloatC>));
    else
      result = ((Float2<FloatC> *) corr.H_elem()) + shift*corr.getVolSize();

    PLEGMA_Propagator<FloatA>* props[3];
    for (int j=0; j<3; j++) {
      if(BP_prop_prods[i][j] == 'u')
	props[j] = &propUP;
      else if(BP_prop_prods[i][j] == 'd')
	props[j] = &propDN;
      else if(BP_prop_prods[i][j] == 's')
	props[j] = &propST;
      else if(BP_prop_prods[i][j] == 'c')
	props[j] = &propCH;
      else
	PLEGMA_error("Unknown propagator %c", BP_prop_prods[i][j]);
    }
    
    ProfileStruct ps(HGC_localVolume3D, sizeof(Float2<FloatC>));
    int myLocalT = corr.localT();
    int maxLocalT = myLocalT;
    MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
    ps.max_volume = HGC_localVolume3D*maxLocalT;
    ps.tune_globally = true;
    ps.aux_range.x = 2;
    
    if(HGC_verbosity>2) PLEGMA_printf("Running for %s\n", BP_prop_prods[i].c_str());
    tuneAndRun(ps, "contract_baryons_size"+std::to_string(BP_prop_prods_count[i].size()), contract_baryons_udsc_host<FloatA,FloatC>, ps, props, corr, result, i);

    if(runFT) {
      FloatC *corr_ip = corr.H_elem() + shift*corr.getVolSize()*2;
      MPI_Allreduce(result, corr_ip, BP_prop_prods_count[i].size()*corr.getVolSize()*2, MPI_Type(corr_ip),
		    MPI_SUM, HGC_spaceComm);
      hostFree(result, BP_prop_prods_count[i].size()*corr.getVolSize()*sizeof(Float2<FloatC>));
    }
    shift+=BP_prop_prods_count[i].size();
  }
}

template
void contract_baryons_udsc<float,float>(PLEGMA_Propagator<float>& propUP, PLEGMA_Propagator<float>& propDN,
					PLEGMA_Propagator<float>& propST, PLEGMA_Propagator<float>& propCH,
					PLEGMA_Correlator<float> &corr, std::vector<int> &todo);

template
void contract_baryons_udsc<double,double>(PLEGMA_Propagator<double>& propUP, PLEGMA_Propagator<double>& propDN,
					  PLEGMA_Propagator<double>& propST, PLEGMA_Propagator<double>& propCH,
					  PLEGMA_Correlator<double> &corr, std::vector<int> &todo);
