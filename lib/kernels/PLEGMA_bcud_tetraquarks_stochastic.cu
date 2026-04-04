#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_bcud_tetraquarks.cuh>


template<typename FloatA, typename FloatC>
__global__ void contract_props_bcud_stochastic(propTex<FloatA> texProp1, propTex<FloatA> texProp2, propTex<FloatA> texProp3,
			       propTex<FloatA> texProp4,propTex<FloatA> texProp5, propTex<FloatA> texProp6, propTex<FloatA> texProp7,
			       propTex<FloatA> texProp8, Float2<FloatC>* block, int size, short *idxs, short *col_contr, Float2<float>*vals,
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
    Float2<FloatA> prop4[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop5[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop6[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop7[N_SPINS][N_SPINS][N_COLS][N_COLS];
    Float2<FloatA> prop8[N_SPINS][N_SPINS][N_COLS][N_COLS];
    
    texProp1.get(prop1,vid);
    texProp2.get(prop2,vid);
    texProp3.get(prop3,vid);
    texProp4.get(prop4,vid);
    texProp5.get(prop5,vid);
    texProp6.get(prop6,vid);
    texProp7.get(prop7,vid);
    texProp8.get(prop8,vid);
    
    Float2<FloatA> (*PropPointer[8])[N_SPINS][N_COLS][N_COLS];
    PropPointer[0]=prop1;
    PropPointer[1]=prop2;
    PropPointer[2]=prop3;
    PropPointer[3]=prop4;
    PropPointer[4]=prop5;
    PropPointer[5]=prop6;
    PropPointer[6]=prop7;
    PropPointer[7]=prop8;
    
    for(int i = 0; i < size; i++) {
      short *mu = idxs + i*8;
      short *c = col_contr + i*8;
    int a[4];  
    
    for (a[0]=0; a[0] < 3; a[0]++){
        for (a[1]=0; a[1] < 3; a[1]++){
            for (a[2]=0; a[2] < 3; a[2]++){
                for (a[3]=0; a[3] < 3; a[3]++){
    

      Float2<FloatC> factor = vals[i];
        accum += factor *
                PropPointer[0+((c[0])%2)][mu[0]][mu[1]][a[c[0]]][a[c[1]]]  *                  PropPointer[2+((c[2])%2)][mu[2]][mu[3]][a[c[2]]][a[c[3]]]
        * conj( PropPointer[4+((c[4])%2)][(mu[5]+2)%4][(mu[4]+2)%4][a[c[5]]][a[c[4]]]) * conj(PropPointer[6+((c[6])%2)][(mu[7]+2)%4][(mu[6]+2)%4][a[c[7]]][a[c[6]]]);
        
//         accum += factor *
//                 PropPointer[c[0]][mu[0]][mu[1]][a[c[0]]][a[c[1]]]  *                  PropPointer[c[2]][mu[2]][mu[3]][a[c[2]]][a[c[3]]]
//         * conj( PropPointer[c[4]][(mu[5]+2)%4][(mu[4]+2)%4][a[c[5]]][a[c[4]]]) * conj(PropPointer[c[6]][(mu[7]+2)%4][(mu[6]+2)%4][a[c[7]]][a[c[6]]]);
        
/*        
	  accum += factor *
                prop1[mu[0]][mu[1]][a[c[0]]][a[c[1]]]  *        prop2[mu[2]][mu[3]][a[c[2]]][a[c[3]]]
        * conj( prop3[(mu[5]+2)%4][(mu[4]+2)%4][a[c[5]]][a[c[4]]]) * conj(prop4[(mu[7]+2)%4][(mu[6]+2)%4][a[c[7]]][a[c[6]]]);*/

      // we apply gamma5 hermiticity for the two bottom props prop3 and prop4. 
      // The offset of 2 with modulo 4 is the hard-coded application of the multiplication with g5 matrices.
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
    fourier_transform_3D(block, &accum, shared_cache, 1, sid3D, source_pos, moms, 0, -1, time_step, tid);
  } else {
    if (sid3D < DGC_localVolume3D)
      block[tid*DGC_localVolume3D + sid3D] = accum;
  }  
}

template<typename FloatA, typename FloatC>
void contract_tetraquarks_bcud_stochastic_host(ProfileStruct &ps,
				PLEGMA_Propagator<FloatA>**props,
				PLEGMA_Correlator<FloatC> &corr,
				Float2<FloatC> *result, int i) {

  int t_size = corr.localT(); if(t_size==0) return;
  int maxT = corr.endT() - corr.startT(); 
  int time_step = get_time_step(ps.tp.grid.x,ps.tp.block.x);
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
  auto propTex4 = toTexture<propTex>(*(props[3]));
  auto propTex5 = toTexture<propTex>(*(props[4]));
  auto propTex6 = toTexture<propTex>(*(props[5]));
  auto propTex7 = toTexture<propTex>(*(props[6]));
  auto propTex8 = toTexture<propTex>(*(props[7]));

  Float2<FloatC> *h_partial_block = NULL;        
  Float2<FloatC> *d_partial_block = NULL;
  size_t alloc_size = (runFT==true) ? (volume * (ps.tp.grid.x/time_step)):volume;
  hostMalloc(h_partial_block, alloc_size * sizeof(Float2<FloatC>));
  cudaMalloc((void**)&d_partial_block, alloc_size * sizeof(Float2<FloatC>) );
  
  short *idxs, *col_contr;
  Float2<float> *vals;
  int size = 0;
  for(int j=0; j<TETRA_bcud_stoch_prop_prods_count[i].size(); j++)
    size += TETRA_bcud_stoch_prop_prods_count[i][j];
  cudaMalloc((void**)&idxs, 8*size*sizeof(short));
  cudaMalloc((void**)&col_contr, 8*size*sizeof(short));
  cudaMalloc((void**)&vals, size*sizeof(Float2<float>));
  int shift = 0;
  for(int j=0; j<TETRA_bcud_stoch_prop_prods_count[i].size(); j++) {
    cudaMemcpy(idxs+8*shift, TETRA_bcud_stoch_prop_prods_idxs[i][j], 8*TETRA_bcud_stoch_prop_prods_count[i][j]*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(col_contr+8*shift, TETRA_bcud_stoch_prop_prods_col_contr[i][j], 8*TETRA_bcud_stoch_prop_prods_count[i][j]*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(vals+shift, TETRA_bcud_stoch_prop_prods_vals[i][j], TETRA_bcud_stoch_prop_prods_count[i][j]*sizeof(Float2<float>), cudaMemcpyHostToDevice);
    shift += TETRA_bcud_stoch_prop_prods_count[i][j];
  }

    cudaError_t error=cudaPeekAtLastError();
    if(error != cudaSuccess) { goto exit; }

  
  for(int it=0; it < t_size; it+=time_step) {
    
    shift = 0;
    for(int j=0; j<TETRA_bcud_stoch_prop_prods_count[i].size(); j++) {
      dim3 grid = ps.tp.grid;
      grid.x = (grid.x/time_step)*std::min(t_size-it, time_step);
      
          
	contract_props_bcud_stochastic
	  <<<grid,ps.tp.block,ps.tp.shared_bytes>>>
	  (*propTex1, *propTex2, *propTex3, *propTex4, *propTex5, *propTex6, *propTex7, *propTex8, d_partial_block, TETRA_bcud_stoch_prop_prods_count[i][j], idxs+8*shift, col_contr+8*shift, vals+shift,
	   source, runFT, *moms, it, std::min(t_size-it, time_step), maxT);
      
      cudaMemcpy(h_partial_block , d_partial_block , alloc_size*sizeof(Float2<FloatC>), cudaMemcpyDeviceToHost);
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
      shift += TETRA_bcud_stoch_prop_prods_count[i][j];
    }
  }
 exit:
  hostFree(h_partial_block, alloc_size*sizeof(Float2<FloatC>));
  cudaFree(d_partial_block); d_partial_block=NULL;
  
  cudaFree(idxs);
  cudaFree(col_contr);
  cudaFree(vals);
  
 // if (ps.tp.aux.x == 2) {
//     cudaFree(texPropProd);
//     holder.clear();
//     for(int t=0; t<time_step; t++) {
//       delete propProd[t];
//     }
  //}
}

template<typename FloatA, typename FloatC>
void contract_tetraquarks_bcud_stochastic(  PLEGMA_Propagator<FloatA>&propLT1, PLEGMA_Propagator<FloatA>&propLT2, 
                                            PLEGMA_Propagator<FloatA>&propST1, PLEGMA_Propagator<FloatA>&propST2,
                                            PLEGMA_Propagator<FloatA>&propCH1, PLEGMA_Propagator<FloatA>&propCH2,
                                            PLEGMA_Propagator<FloatA>&propBT1, PLEGMA_Propagator<FloatA>&propBT2,
                                            PLEGMA_Correlator<FloatC> &corr, std::vector<int> &todo){

  bool runFT = (corr.getCorrSpace()==MOMENTUM_SPACE);
  int shift = 0;
  for(auto i: todo) {
    Float2<FloatC> *result = NULL;
    if(runFT)
      hostMalloc(result, TETRA_bcud_stoch_prop_prods_count[i].size()*corr.getVolSize()*sizeof(Float2<FloatC>));
    else
      result = ((Float2<FloatC> *) corr.H_elem()) + shift*corr.getVolSize();

    PLEGMA_Propagator<FloatA>* props[8];
//     props[0]=&propLT1;
//     props[1]=&propLT2;
//     props[2]=&propBT1;
//     props[3]=&propBT2;
    
    for (int j=0; j<4; j++) {
      if(TETRA_bcud_stoch_prop_prods[i][j] == 'u'){
	props[2*j] = &propLT1;
    props[2*j+1] = &propLT2;}
      else if(TETRA_bcud_stoch_prop_prods[i][j] == 'd'){
    props[2*j] = &propLT1;
    props[2*j+1] = &propLT2;}
      else if(TETRA_bcud_stoch_prop_prods[i][j] == 's'){
    props[2*j] = &propST1;
    props[2*j+1] = &propST2;}
      else if(TETRA_bcud_stoch_prop_prods[i][j] == 'c'){
    props[2*j] = &propCH1;
    props[2*j+1] = &propCH2;}
      else if(TETRA_bcud_stoch_prop_prods[i][j] == 'b'){
	props[2*j] = &propBT1;
    props[2*j+1] = &propBT2;}
      else{
	PLEGMA_error("Unknown propagator %c", TETRA_bcud_stoch_prop_prods[i][j]);}
    }
    
    ProfileStruct ps(HGC_localVolume3D, sizeof(Float2<FloatC>));
    int myLocalT = corr.localT();
    int maxLocalT = myLocalT;
    MPI_Allreduce( &myLocalT, &maxLocalT, 1, MPI_Type(maxLocalT), MPI_MAX, HGC_fullComm);
    ps.max_volume = HGC_localVolume3D*maxLocalT;
    ps.tune_globally = true;
    ps.aux_range.x = 1;
    
    if(HGC_verbosity>2) PLEGMA_printf("Running for %s\n", TETRA_bcud_stoch_prop_prods[i].c_str());
    tuneAndRun(ps, "contract_tetraquarks_bcud_stochastic_size"+std::to_string(TETRA_bcud_stoch_prop_prods_count[i].size()), contract_tetraquarks_bcud_stochastic_host<FloatA,FloatC>, ps, props, corr, result, i);

    if(runFT) {
      FloatC *corr_ip = corr.H_elem() + shift*corr.getVolSize()*2;
      MPI_Allreduce(result, corr_ip, TETRA_bcud_stoch_prop_prods_count[i].size()*corr.getVolSize()*2, MPI_Type(corr_ip),
		    MPI_SUM, HGC_spaceComm);
      hostFree(result, TETRA_bcud_stoch_prop_prods_count[i].size()*corr.getVolSize()*sizeof(Float2<FloatC>));
    }
    shift+=TETRA_bcud_stoch_prop_prods_count[i].size();
  }
}

template
void contract_tetraquarks_bcud_stochastic<float,double>(PLEGMA_Propagator<float>&propLT1, PLEGMA_Propagator<float>&propLT2, 
                                                   PLEGMA_Propagator<float>&propST1, PLEGMA_Propagator<float>&propST2,
                                                   PLEGMA_Propagator<float>&propCH1, PLEGMA_Propagator<float>&propCH2,
                                       PLEGMA_Propagator<float>&propBT1, PLEGMA_Propagator<float>&propBT2,
                                        PLEGMA_Correlator<double> &corr, std::vector<int> &todo);

template
void contract_tetraquarks_bcud_stochastic<float,float>(PLEGMA_Propagator<float>&propLT1, PLEGMA_Propagator<float>&propLT2, 
                                                  PLEGMA_Propagator<float>&propST1, PLEGMA_Propagator<float>&propST2,
                                                  PLEGMA_Propagator<float>&propCH1, PLEGMA_Propagator<float>&propCH2,
                                       PLEGMA_Propagator<float>&propBT1, PLEGMA_Propagator<float>&propBT2,
					PLEGMA_Correlator<float> &corr, std::vector<int> &todo);

template
void contract_tetraquarks_bcud_stochastic<double,double>(PLEGMA_Propagator<double>&propLT1, PLEGMA_Propagator<double>&propLT2, 
                                                    PLEGMA_Propagator<double>&propST1, PLEGMA_Propagator<double>&propST2,
                                                    PLEGMA_Propagator<double>&propCH1, PLEGMA_Propagator<double>&propCH2,
                                         PLEGMA_Propagator<double>&propBT1, PLEGMA_Propagator<double>&propBT2,
					  PLEGMA_Correlator<double> &corr, std::vector<int> &todo);
