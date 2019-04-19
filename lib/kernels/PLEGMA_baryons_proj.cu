#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_all_baryons_arrays.cuh>

template<typename FloatA, typename FloatC>
static __device__ void create_prop_product(Float2<FloatC> propProd[N_SPINS][N_SPINS][N_SPINS][N_SPINS][N_SPINS][N_SPINS],
					   propTex<FloatA> texProp1, propTex<FloatA> texProp2, propTex<FloatA> texProp3, size_t vid){
  Float2<FloatA> prop1[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatA> prop2[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatA> prop3[N_SPINS][N_SPINS][N_COLS][N_COLS];
  texProp1.get(prop1,vid);
  texProp2.get(prop2,vid);
  texProp3.get(prop2,vid);
#pragma unroll
  for(int mu1 = 0 ; mu1 < 4 ; mu1++)
#pragma unroll
    for(int mu2 = 0 ; mu2 < 4 ; mu2++)
#pragma unroll
      for(int mu3 = 0 ; mu3 < 4 ; mu3++)
#pragma unroll
	for(int mu4 = 0 ; mu4 < 4 ; mu4++)
#pragma unroll
	  for(int mu5 = 0 ; mu5 < 4 ; mu5++)
#pragma unroll
	    for(int mu6 = 0 ; mu6 < 4 ; mu6++) {
	      propProd[mu1][mu2][mu3][mu4][mu5][mu6] = 0.;
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
		    propProd[mu1][mu2][mu3][mu4][mu5][mu6] += factor *
		      (prop1[mu1][mu2][a][a1] + prop2[mu3][mu4][b][b1] + prop3[mu5][mu6][c][c1]);
		  }
		}

	    }
}

template<typename FloatA, typename FloatC, bool runFT>
__global__ void contract_all_baryons_kernel(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
					    propTex<FloatA> texPropST, propTex<FloatA> texPropCH, FloatC* block,
					    int it, int x0, int y0, int z0){

  Float2<FloatC> *block2 = (Float2<FloatC> *)block;
  size_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t vid = sid + it*c_stride_spatial;
  int source_pos[3] = {x0, y0, z0}; 
  
  Float2<FloatC> accum[all_baryons_size];
  for(int i=0; i<all_baryons_size; i++)
    accum[i] = 0;

  if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
    int inc = 0;
    // loop over the prop. products
    for(int p=0; p<all_prop_prod_length; p++) {
      Float2<FloatC> propProd[N_SPINS][N_SPINS][N_SPINS][N_SPINS][N_SPINS][N_SPINS];
      propTex<FloatA> props[3];
      for (int i=0; i<3; i++) {
	if(prop_prod[p][i] == 'u')
	  props[i] = texPropUP;
	else if(prop_prod[p][i] == 'd')
	  props[i] = texPropDN;
	else if(prop_prod[p][i] == 's')
	  props[i] = texPropST;
	else if(prop_prod[p][i] == 'c')
	  props[i] = texPropCH;
      }
      printf("Running for %c %c %c\n", prop_prod[p][0], prop_prod[p][1], prop_prod[p][2]);
      create_prop_product(propProd, props[0], props[1], props[2], vid);
      printf("created prop product\n", prop_prod[p][0], prop_prod[p][1], prop_prod[p][2]);

  
      // loop over baryons with the same prop. product
      for(int b=0; b<all_baryon_length[p]; b++) {
	
	// loop over the gamma insertions within the baryon
	for(int g=0; g<all_gamma_length[p][b]; g++) {

#pragma unroll
	  for(int id = 0 ; id < all_gamma_comp_length[p][b][g]; id++) {
	    int mu1=all_idxs[p][b][g][id][0];
	    int mu2=all_idxs[p][b][g][id][1];
	    int mu3=all_idxs[p][b][g][id][2];
	    int mu4=all_idxs[p][b][g][id][3];
	    int mu5=all_idxs[p][b][g][id][4];
	    int mu6=all_idxs[p][b][g][id][5];
	    
	    accum[inc] += all_vals[p][b][g][id]*propProd[mu1][mu2][mu3][mu4][mu5][mu6];
	  }
	  inc++;
	}
      }
    }
  }
  if(runFT) {
    printf("Running FT\n");
    const int shared_size = 10;
    __shared__ Float2<FloatC> shared_cache[shared_size*THREADS_PER_BLOCK];
    for(int i=0; i < (all_baryons_size+shared_size-1)/shared_size; i++) {
      Float2<FloatC> *accum_i = accum+i*shared_size;
      int reduce = MIN(all_baryons_size-i*shared_size, shared_size);
      int out_pad = all_baryons_size-reduce;
      int out_shift = i*shared_size;
      fourier_transform_3D(block2, accum_i, shared_cache, reduce, sid, source_pos, out_pad, out_shift);
    }
  } else {
    for(int i = 0 ; i < all_baryons_size; i++){
      block2[ sid*all_baryons_size + i] = accum[i];
    }
  }
}

template<typename FloatA, typename FloatC, bool runFT>
static void contract_all_baryons(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
				 propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
				 PLEGMA_Correlator<FloatC> &corr, int it){

  int SpVol = GK_localVolume/GK_localL[3];

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (SpVol + blockDim.x -1)/blockDim.x , 1 , 1); // spawn threads only for the spatial volume

  FloatC *h_partial_block = NULL;
  FloatC *d_partial_block = NULL;

  int site_size=all_baryons_size;
  size_t volume;
  size_t size;
  size_t alloc_size;
  if(runFT==true){
    volume = GK_Nmoms;
    size = site_size*volume;
    alloc_size = size * gridDim.x;
  } else {
    volume = SpVol;
    size = site_size*volume;
    alloc_size = size; 
  }
  h_partial_block = (FloatC*)malloc(alloc_size*sizeof(FloatC));
  if(h_partial_block == NULL) errorQuda("contract_baryons_kernel: Cannot allocate host block.\n");
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(FloatC) );
  checkCudaError();

  if(runFT)
    cudaFuncSetCacheConfig(contract_all_baryons_kernel<FloatA,FloatC,runFT>, cudaFuncCachePreferShared);

  int isource = corr.getIdSource();
  contract_all_baryons_kernel<FloatA,FloatC,runFT><<<gridDim,blockDim>>>( texPropUP,texPropDN,texPropST,texPropCH,
									  d_partial_block, it,
									  GK_sourcePosition[isource][0],
									  GK_sourcePosition[isource][1],
									  GK_sourcePosition[isource][2]);
  checkCudaError();
  
  cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(FloatC) , cudaMemcpyDeviceToHost);
  checkCudaError();
    
  if(runFT==true){
    FloatC *reduction =(FloatC*) calloc(size,sizeof(FloatC));
    for(size_t i = 0 ; i < size; i++)
      for(int j = 0 ; j < gridDim.x; j++) {
	reduction[i+0] += h_partial_block[(i*gridDim.x + j)+0];
	reduction[i+1] += h_partial_block[(i*gridDim.x + j)+1];
      }
    MPI_Allreduce(reduction, h_partial_block, size, MPI_Type(reduction), MPI_SUM, GK_spaceComm);
    free(reduction);
  }
  
  FloatC *corr_it = corr.getCorr() + it*size;
  for(size_t v = 0 ; v < size; v++)
    corr_it[v] = h_partial_block[v];
  
  free(h_partial_block);
  cudaFree(d_partial_block);
  checkCudaError();
}

template<typename FloatA, typename FloatC>
void contract_all_baryons(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			  propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			  PLEGMA_Correlator<FloatC> &corr, int it){
  if (corr.getCorrSpace()==POSITION_SPACE){
    contract_all_baryons<FloatA,FloatC,false>(texPropUP,texPropDN,texPropST,texPropCH,corr,it);
  }
  else if(corr.getCorrSpace()==MOMENTUM_SPACE) {
    contract_all_baryons<FloatA,FloatC,true>(texPropUP,texPropDN,texPropST,texPropCH,corr,it);
  }
  else
    errorQuda("run_contract_baryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");
  checkCudaError();
}

template
void contract_all_baryons<float,float>(propTex<float> texPropUP, propTex<float> texPropDN,
				       propTex<float> texPropST, propTex<float> texPropCH,
				       PLEGMA_Correlator<float> &corr, int it);

template
void contract_all_baryons<double,double>(propTex<double> texPropUP, propTex<double> texPropDN,
					 propTex<double> texPropST, propTex<double> texPropCH,
					 PLEGMA_Correlator<double> &corr, int it);
