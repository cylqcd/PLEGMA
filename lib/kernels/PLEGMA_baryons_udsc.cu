#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_baryons_udsc.cuh>

template<typename FloatA, typename FloatC>
__global__ void create_prop_product(generic2<FloatC> propProd,
				    propTex<FloatA> texProp1, propTex<FloatA> texProp2, propTex<FloatA> texProp3, int it){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t vid = sid + it*DGC_localVolume3D;
  sidStride ss(sid, DGC_localVolume3D);
  
  if (sid < DGC_localVolume3D){ // I work only on the spatial volume
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
		propProd.set(i,ss,accum);
	      }
  }
}

template<typename FloatC>
__global__ void contract_prop_prod(genericTex<FloatC> texPropProd, Float2<FloatC>* block,
				   int size, int *idxs, Float2<float>*vals,
				   int3 source, bool runFT, tex_mom_list moms){

  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<FloatC> accum=0;

  if (sid < DGC_localVolume3D){ // I work only on the spatial volume
    for(int i = 0; i < size; i++) {
      int mu = 0;
#pragma unroll
      for(int j = 0; j < 6; j++)
	mu=mu*N_SPINS+idxs[i*6+j];
      sidStride ss(sid, DGC_localVolume3D);
      accum += texPropProd.get(mu, ss)*((Float2<FloatC>) vals[i]);
    }
  }
  if(runFT) {
    extern __shared__ int ext_shared_cache[];
    Float2<FloatC> *shared_cache = (Float2<FloatC> *) ext_shared_cache;
    int source_pos[3] = {source.x, source.y, source.z};
    fourier_transform_3D(block, &accum, shared_cache, 1, sid, source_pos, moms);
  } else {
    if(block!=NULL)
      block[sid] = accum;
  }  
}

template<typename FloatA, typename FloatC>
void contract_baryons_udsc(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			   propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			   PLEGMA_Correlator<FloatC> &corr, int it, std::vector<int> &todo){

  int SpVol = HGC_localVolume3D;
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int3 source = corr.getSource3();
  size_t volume = corr.getVolSize()/HGC_localL[3];
  tex_mom_list moms = corr.getTexMomList();
  size_t shift = 0;
  Float2<FloatC> *h_partial_block = NULL;        
  Float2<FloatC> *d_partial_block = NULL;
  
  PLEGMA_Field<FloatC> propProd(DEVICE, N_SPINS*N_SPINS*N_SPINS*N_SPINS*N_SPINS*N_SPINS, NO_GHOSTS, true);
  genericTex<FloatC> texPropProd(propProd.createTexObject());
  generic2<FloatC> propProd2(propProd.D_elem());

  for(auto i: todo) {
    propTex<FloatA> props[3];
    for (int j=0; j<3; j++) {
      if(BP_prop_prods[i][j] == 'u')
	props[j] = texPropUP;
      else if(BP_prop_prods[i][j] == 'd')
	props[j] = texPropDN;
      else if(BP_prop_prods[i][j] == 's')
	props[j] = texPropST;
      else if(BP_prop_prods[i][j] == 'c')
	props[j] = texPropCH;
      else
	PLEGMA_error("Unknown propagator %c", BP_prop_prods[i][j]);
    }
    ProfileStruct ps(SpVol,0);
    if(HGC_verbosity>2) PLEGMA_printf("Running for %s\n", BP_prop_prods[i].c_str());
    tuneAndRun(ps, "create_prop_product", create_prop_product<FloatA,FloatC>, propProd2, props[0], props[1], props[2], it);
    
    for(int j=0; j<BP_prop_prods_count[i].size(); j++) {
      int size = BP_prop_prods_count[i][j];
      ProfileStruct ps2(SpVol,sizeof(Float2<FloatC>));
      int *idxs;
      Float2<float> *vals;
      cudaMalloc((void**)&idxs, 6*size*sizeof(int));
      cudaMalloc((void**)&vals, size*sizeof(Float2<float>));
      cudaMemcpy(idxs, BP_prop_prods_idxs[i][j], 6*size*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(vals, BP_prop_prods_vals[i][j], size*sizeof(Float2<float>), cudaMemcpyHostToDevice);
      checkCudaError();
      tune(ps2, "contract_prop_prod_"+std::to_string(size), contract_prop_prod<FloatC>, texPropProd, d_partial_block,
	   size, idxs, vals, source, runFT, moms);

      size_t alloc_size = (runFT==true) ? (volume * ps2.tp.grid.x * 2):(volume * 2);
      hostMalloc(h_partial_block, alloc_size*sizeof(FloatC));
      cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(FloatC) );
      checkCudaError();

      run(ps2, "contract_prop_prod_"+std::to_string(size), contract_prop_prod<FloatC>, texPropProd, d_partial_block,
	  size, idxs, vals, source, runFT, moms);
      cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
      checkCudaError();
      if(runFT==true){
	int gridDimX = ps2.tp.grid.x;
	Float2<FloatC> *reduction;
	hostMalloc(reduction, volume*2*sizeof(FloatC));
	for(size_t k = 0 ; k < volume; k++) {
	  reduction[k] = 0;
	  for(int l = 0 ; l < gridDimX; l++) {
	    reduction[k] += h_partial_block[k*gridDimX + l];
	  }
	}
	MPI_Allreduce(reduction, h_partial_block, volume*2, MPI_Type<FloatC>(), MPI_SUM, HGC_spaceComm);
	hostFree(reduction, volume*2*sizeof(FloatC));
      }
      
      Float2<FloatC> *corr2 = (Float2<FloatC> *) corr.getCorr() + shift + it*volume;
      for(size_t v = 0 ; v < volume; v++)
	corr2[v] = h_partial_block[v];
      
      shift += HGC_localL[3]*volume;
      
      hostFree(h_partial_block, alloc_size*sizeof(FloatC));
      cudaFree(d_partial_block); d_partial_block=NULL;
      cudaFree(idxs);
      cudaFree(vals);
      checkCudaError();
    }
  }
  propProd.destroyTexObject(texPropProd.tex);
}

template
void contract_baryons_udsc<float,float>(propTex<float> texPropUP, propTex<float> texPropDN,
					propTex<float> texPropST, propTex<float> texPropCH,
					PLEGMA_Correlator<float> &corr, int it, std::vector<int> &todo);

template
void contract_baryons_udsc<double,double>(propTex<double> texPropUP, propTex<double> texPropDN,
					  propTex<double> texPropST, propTex<double> texPropCH,
					  PLEGMA_Correlator<double> &corr, int it, std::vector<int> &todo);
