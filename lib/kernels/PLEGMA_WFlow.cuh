#pragma once
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_getSet.cuh>

using namespace plegma;

template<typename FloatG>
__device__ void calculatestaples(Float2<FloatG> U[N_COLS][N_COLS], gauge2<FloatG> &gaugep, int dir,  int sid) {

  Float2<FloatG> aux[N_COLS][N_COLS], u1[N_COLS][N_COLS], u2[N_COLS][N_COLS]; 

  init_to_zero<FloatG>( U ); //U=0
  
  // Loop over directions nu!=dir
  #pragma unroll
  for(int i=1; i<N_DIMS; i++){
    int nu = (dir+i)%4;
      
    //fwd
    gaugep.get<Plus>(u1,dir,sid,nu);  // u1 = U(i+\nu)^(\mu)
    gaugep.get(u2, nu, sid);          // u2 = U(i)^(\nu)
    mul_Gdag_Gdag(aux,u1,u2);         // aux = u1+*u2+
    
    gaugep.get<Plus>(u2,nu,sid,dir); // u2 = U(i+\mu)^(\nu)
    mul_G_G(u1,u2,aux);              // u1 = U(i+\mu)^(\nu)*U(i+\nu)^(+\mu)*U(i)^(+\nu) = staple_fwd 

    G_plus_aG(U, u1, 1.);//U += u1
    //bwd
    //nu+=4;
        
    gaugep.get<PlusMinus>(u1,nu,sid,dir,nu);  // u1 = U(i+\mu-\nu)^(\nu)
    gaugep.get<Minus>(u2, dir, sid, nu);      // u2 = U(i-\nu)^(\mu)
    mul_Gdag_Gdag(aux,u1,u2);                 // aux =  u1+*u2+ 

    gaugep.get<Minus>(u2, nu, sid, nu);       // u2 = U(i-\nu)^(\nu)
    mul_G_G(u1,aux,u2);                       // u1 = U(i+\mu-\nu)^(+\nu)*U(i-\nu)^(+\mu)*U(i-\nu)^(\nu) = staple_bwd

    G_plus_aG( U, u1, 1.);  //U += staple
  }
  
}

template<typename FloatG, typename FloatE>
__global__ void ZUpdate( gauge2<FloatG> W_in, gauge2<FloatG> Z, FloatE e_work, FloatE e_save ){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  if (sid < W_in.volume()) {
    Float2<FloatG> V_i[N_COLS][N_COLS], staple[N_COLS][N_COLS], aux[N_COLS][N_COLS];
      
    #pragma unroll
    for( int dir=0; dir<N_DIMS; dir++)
      {
	//read W_in(x,mu) an calculate its staples (maybe I have to modify)
	W_in.get(V_i, dir, sid);

	calculatestaples( staple, W_in, dir, sid);        //tested
	
	//compute derivative Z=antihermtraceless(link*staple)
	mul_G_G(aux, V_i, staple);
	
	AntiHermTrless_G( aux );                          //aux = - Z(V_i)
	
	//printf("Z=antiH(W*staples)_11=%f\n",aux[1][1].y);
	
	//compute e_work*Z + e_save*Z_previous
	  
	scaleG( aux, -e_work);                   //aux = eps* Z(V_i)
	
	if( sqrt(e_save*e_save) > 1e-14 ){
	  Z.get( V_i, dir, sid);                           //I  can put it into V_i, change
	  G_plus_aG(aux, V_i, e_save);
	}
	//store Z_new
	Z.set(aux, dir, sid);                            //Write aux on Z
	  
      }
  }
}

template<typename FloatG>
__global__ void WUpdate( gauge2<FloatG> W, gauge2<FloatG> Z ){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  if (sid < W.volume()) {
    Float2<FloatG> Z_i[N_COLS][N_COLS], W_i[N_COLS][N_COLS], W_f[N_COLS][N_COLS];
      
    #pragma unroll
    for( int dir=0; dir<N_DIMS; dir++)
      {
	Z.get(Z_i, dir, sid);
	exp_G_Taylor( Z_i );
	
	W.get(W_i, dir, sid);

	mul_G_G(W_f, Z_i, W_i);                                   

	W.set(W_f, dir, sid);
      }
  }
}

template<typename FloatG, typename FloatE>
__inline__ void GFlow_substep( gauge2<FloatG> W, gauge2<FloatG> Z, FloatE e_work, FloatE e_save )
{
  dim3 blockDim( THREADS_PER_BLOCK, 1, 1 );
  dim3 gridDim( (W.volume() + blockDim.x -1)/blockDim.x, 1, 1);
  //FloatG* debug_array;
  
  ZUpdate<FloatG,FloatE><<<gridDim,blockDim>>>( W, Z, e_work, e_save );
  cudaDeviceSynchronize();

  WUpdate<FloatG><<<gridDim,blockDim>>>( W, Z );
  checkQudaError();
}


template<typename FloatG>
__global__ void unitarize_dev_kernel( gauge2<FloatG> gauge ){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  if (sid < gauge.volume()) {
    Float2<FloatG> U[N_COLS][N_COLS];
      
    #pragma unroll
    for( int dir=0; dir<N_DIMS; dir++)
      {
	gauge.get(U, dir, sid);
	enforce_unitarity( U );
      	gauge.set(U, dir, sid);
      }
  }
}


template<typename FloatG>
__inline__ void unitarize_dev( gauge2<FloatG> gauge )
{
  dim3 blockDim( THREADS_PER_BLOCK, 1, 1 );
  dim3 gridDim( (gauge.volume() + blockDim.x -1)/blockDim.x, 1, 1);
  
  unitarize_dev_kernel<<<gridDim,blockDim>>>( gauge );
  checkQudaError();
}

//#####################################################################################
//##############                   Xcheck Functions                   #################
//#####################################################################################

template<typename FloatG>
static __global__ void calcPlaqStaplesDef_kernel( gauge2<FloatG> gaugep, FloatG* partial_plaq ){
  __shared__ FloatG shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  if (sid < gaugep.volume()) {
    FloatG trace = 0.;
    
    Float2<FloatG> S[N_COLS][N_COLS], U[N_COLS][N_COLS];

    for( int dir=0; dir<N_DIMS; dir++){
      gaugep.get( U, dir, sid );
      calculatestaples<FloatG>( S, gaugep, dir, sid );
      FloatG tmp_dbug = real_trace_mul_G_G<FloatG,FloatG>( U, S);
      //printf("%f \n", tmp_dbug);
      trace += tmp_dbug;
    }
    
    shared_cache[cacheIndex] = trace;
  }
  else {
    shared_cache[cacheIndex] = 0;
  }
  
  reduce( shared_cache, 1);
  if(cacheIndex == 0 && partial_plaq!=NULL){
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory  
  }
}

template<typename FloatG>
static FloatG calcPlaqStaplesDef(gauge2<FloatG> gaugep){

  FloatG plaquette = 0.;
  FloatG globalPlaquette = 0.;
  FloatG *d_partial_plaq = NULL;
  FloatG *h_partial_plaq = NULL;
  dim3 blockDim( THREADS_PER_BLOCK, 1, 1 );
  dim3 gridDim( (gaugep.volume() + blockDim.x -1)/blockDim.x, 1, 1);
   
  h_partial_plaq = (FloatG*) malloc(gridDim.x * sizeof(FloatG) );
  if(h_partial_plaq == NULL) errorQuda("Error allocate memory for host partial plaq");
  cudaMalloc((void**)&d_partial_plaq, gridDim.x * sizeof(FloatG));

  calcPlaqStaplesDef_kernel<FloatG><<<gridDim,blockDim>>>( gaugep, d_partial_plaq );

  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDim.x * sizeof(FloatG) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_plaq);
  checkQudaError();
  
  for(int i = 0 ; i < gridDim.x ; i++)
    plaquette += h_partial_plaq[i];
  free(h_partial_plaq);

  MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_Type(plaquette) , MPI_SUM , HGC_fullComm);  
  return globalPlaquette/(HGC_totalVolume*N_COLS*24);
}
