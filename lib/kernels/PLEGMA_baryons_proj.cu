#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_baryons_proj.cuh>

template<typename FloatA, typename FloatC>
__global__ void create_prop_product(Float2<FloatC> *propProd,
				    propTex<FloatA> texProp1, propTex<FloatA> texProp2, propTex<FloatA> texProp3, int it){
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t vid = sid + it*DGC_localVolume3D;
  
  if (sid < DGC_localVolume3D){ // I work only on the spatial volume
    Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop3[N_SPINS][N_SPINS][N_COLS][N_COLS];
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    texProp3.get(prop3,vid);
    for(int mu1 = 0 ; mu1 < N_SPINS ; mu1++)
      for(int mu2 = 0 ; mu2 < N_SPINS ; mu2++)
	for(int mu3 = 0 ; mu3 < N_SPINS ; mu3++)
	  for(int mu4 = 0 ; mu4 < N_SPINS ; mu4++)
	    for(int mu5 = 0 ; mu5 < N_SPINS ; mu5++)
	      for(int mu6 = 0 ; mu6 < N_SPINS ; mu6++) {
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
		      (prop1[mu1][mu2][a][a1] + prop2[mu3][mu4][b][b1] + prop3[mu5][mu6][c][c1]);
		  }
		}
		propProd[(((((mu1*N_SPINS+mu2)*N_SPINS+mu3)*N_SPINS+mu4)*N_SPINS+mu5)*N_SPINS+mu6)*DGC_localVolume3D + sid] = accum;		
	      }
  }
}

template<typename FloatC>
__global__ void contract_prop_prod(genericTex<FloatC> texPropProd, Float2<FloatC>* block,
				   int size, int *idxs, Float2<float>*vals,
				   int3 source, bool runFT){

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
    fourier_transform_3D(block, &accum, shared_cache, 1, sid, source_pos);
  } else {
    if(block!=NULL)
      block[sid] = accum;
  }  
}

template<typename FloatA, typename FloatC>
void contract_baryons_proj(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			   propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			   PLEGMA_Correlator<FloatC> &corr, int it, std::vector<int> &todo){

  int SpVol = HGC_localVolume3D;
  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int3 source = corr.getSource3();
  PLEGMA_Field<FloatC> propProd(DEVICE, N_SPINS*N_SPINS*N_SPINS*N_SPINS*N_SPINS*N_SPINS, NO_GHOSTS, true);
  genericTex<FloatC> texPropProd;
  texPropProd.tex = propProd.createTexObject();
  
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
    tuneAndRun(ps, "create_prop_product", create_prop_product<FloatA,FloatC>, (Float2<FloatC>*) propProd.D_elem(), props[0], props[1], props[2], it);
    
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
      tune(ps2, "contract_prop_prod_"+std::to_string(size), contract_prop_prod<FloatC>, texPropProd, (Float2<FloatC>*) NULL,
	   size, idxs, vals, source, runFT);

      //TODO
      cudaFree(idxs);
      cudaFree(vals);
    }
    
    
    
  }
  propProd.destroyTexObject(texPropProd.tex);
}


template
void contract_baryons_proj<float,float>(propTex<float> texPropUP, propTex<float> texPropDN,
					propTex<float> texPropST, propTex<float> texPropCH,
					PLEGMA_Correlator<float> &corr, int it, std::vector<int> &todo);

template
void contract_baryons_proj<double,double>(propTex<double> texPropUP, propTex<double> texPropDN,
					  propTex<double> texPropST, propTex<double> texPropCH,
					  PLEGMA_Correlator<double> &corr, int it, std::vector<int> &todo);
