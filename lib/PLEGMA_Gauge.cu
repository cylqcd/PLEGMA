#include <PLEGMA_Gauge.h>
#include <PLEGMA_plaquette.cuh>
using namespace plegma;

//--------------------------//
// class PLEGMA_Gauge //
//--------------------------//

template<typename Float>
PLEGMA_Gauge<Float>::PLEGMA_Gauge(ALLOCATION_FLAG alloc_flag): 
  PLEGMA_Field<Float>(alloc_flag, GAUGE){ ; }

template<typename Float>
void PLEGMA_Gauge<Float>::packGauge(double **p_gauge){
  
  for(int dir = 0 ; dir < N_DIMS ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++)
	  for(int part = 0 ; part < 2 ; part++){
	    PLEGMA_Field<Float>::h_elem[dir*N_COLS*N_COLS*GK_localVolume*2 + 
		       c1*N_COLS*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       iv*2 + part] = 
	      (Float) p_gauge[dir][iv*N_COLS*N_COLS*2 + 
				   c1*N_COLS*2 + c2*2 + part];
	  }
}

template<typename Float>
void PLEGMA_Gauge<Float>::packGaugeToBackup(void **gauge){
  double **p_gauge = (double**) gauge;
  if(PLEGMA_Field<Float>::h_elem_backup != NULL){
    for(int dir = 0 ; dir < N_DIMS ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++)
    for(int c2 = 0 ; c2 < N_COLS ; c2++)
    for(int part = 0 ; part < 2 ; part++){
      PLEGMA_Field<Float>::h_elem_backup[dir*N_COLS*N_COLS*GK_localVolume*2 + 
			c1*N_COLS*GK_localVolume*2 + 
			c2*GK_localVolume*2 + 
			iv*2 + part] = 
	(Float) p_gauge[dir][iv*N_COLS*N_COLS*2 + 
			     c1*N_COLS*2 + 
			     c2*2 + part];
    }
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }

}

template<typename Float>
void PLEGMA_Gauge<Float>::justDownloadGauge(){
  cudaMemcpy(PLEGMA_Field<Float>::h_elem,PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Gauge<Float>::loadGauge(){
  cudaMemcpy(PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::h_elem,PLEGMA_Field<Float>::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void PLEGMA_Gauge<Float>::loadGaugeFromBackup(){
  if(PLEGMA_Field<Float>::h_elem_backup != NULL){
    cudaMemcpy(PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::h_elem_backup, PLEGMA_Field<Float>::bytes_total_length, 
	       cudaMemcpyHostToDevice );
    checkCudaError();
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }
}

// gpu collect ghost and send it to host
template<typename Float>
void PLEGMA_Gauge<Float>::ghostToHost(){   

  // direction x 
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    // number of blocks that we need
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3];
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    position = GK_localL[0]-1;
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++){
	  d_elem_offset = (PLEGMA_Field<Float>::d_elem + 
			   i*N_COLS*N_COLS*GK_localVolume*2 + 
			   c1*N_COLS*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);
	  h_elem_offset = (PLEGMA_Field<Float>::h_elem + 
			   GK_minusGhost[0]*N_DIMS*N_COLS*N_COLS*2 + 
			   i*N_COLS*N_COLS*GK_surface3D[0]*2 + 
			   c1*N_COLS*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++){
	  d_elem_offset = (PLEGMA_Field<Float>::d_elem + 
			   i*N_COLS*N_COLS*GK_localVolume*2 + 
			   c1*N_COLS*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (PLEGMA_Field<Float>::h_elem + 
			   GK_plusGhost[0]*N_DIMS*N_COLS*N_COLS*2 + 
			   i*N_COLS*N_COLS*GK_surface3D[0]*2 + 
			   c1*N_COLS*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction y 
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    // number of blocks that we need
    int height = GK_localL[2] * GK_localL[3];
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++){
	  d_elem_offset = (PLEGMA_Field<Float>::d_elem + 
			   i*N_COLS*N_COLS*GK_localVolume*2 + 
			   c1*N_COLS*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (PLEGMA_Field<Float>::h_elem + 
			   GK_minusGhost[1]*N_DIMS*N_COLS*N_COLS*2 + 
			   i*N_COLS*N_COLS*GK_surface3D[1]*2 + 
			   c1*N_COLS*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++){
	  d_elem_offset = (PLEGMA_Field<Float>::d_elem + 
			   i*N_COLS*N_COLS*GK_localVolume*2 + 
			   c1*N_COLS*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (PLEGMA_Field<Float>::h_elem + 
			   GK_plusGhost[1]*N_DIMS*N_COLS*N_COLS*2 + 
			   i*N_COLS*N_COLS*GK_surface3D[1]*2 + 
			   c1*N_COLS*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  
  // direction z 
  if( GK_localL[2] < GK_totalL[2]){

    int position;
    // number of blocks that we need
    int height = GK_localL[3]; 
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++){
	  d_elem_offset = (PLEGMA_Field<Float>::d_elem + 
			   i*N_COLS*N_COLS*GK_localVolume*2 + 
			   c1*N_COLS*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + position*2);  
	  h_elem_offset = (PLEGMA_Field<Float>::h_elem + 
			   GK_minusGhost[2]*N_DIMS*N_COLS*N_COLS*2 + 
			   i*N_COLS*N_COLS*GK_surface3D[2]*2 + 
			   c1*N_COLS*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++){
	  d_elem_offset = (PLEGMA_Field<Float>::d_elem + 
			   i*N_COLS*N_COLS*GK_localVolume*2 + 
			   c1*N_COLS*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (PLEGMA_Field<Float>::h_elem + 
			   GK_plusGhost[2]*N_DIMS*N_COLS*N_COLS*2 + 
			   i*N_COLS*N_COLS*GK_surface3D[2]*2 + 
			   c1*N_COLS*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = N_DIMS*N_COLS*N_COLS;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset=PLEGMA_Field<Float>::d_elem+position*2;
    h_elem_offset=PLEGMA_Field<Float>::h_elem+GK_minusGhost[3]*N_DIMS*N_COLS*N_COLS*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
    // set minus points to plus area
    position = 0;
    d_elem_offset=PLEGMA_Field<Float>::d_elem+position*2;
    h_elem_offset=PLEGMA_Field<Float>::h_elem+GK_plusGhost[3]*N_DIMS*N_COLS*N_COLS*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
  }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Gauge<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < N_DIMS; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = 
	  GK_surface3D[idim]*N_COLS*N_COLS*N_DIMS*2*sizeof(Float);
	// send to plus
	pointer_receive = PLEGMA_Field<Float>::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*N_COLS*N_COLS*N_DIMS*2;
	pointer_send = PLEGMA_Field<Float>::h_elem + GK_minusGhost[idim]*N_COLS*N_COLS*N_DIMS*2;
	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = PLEGMA_Field<Float>::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*N_COLS*N_COLS*N_DIMS*2;
	pointer_send = PLEGMA_Field<Float>::h_elem + GK_plusGhost[idim]*N_COLS*N_COLS*N_DIMS*2;
	mh_from_fwd[idim] = comm_declare_receive_relative(pointer_receive,idim,1,nbytes);
	mh_send_back[idim] = comm_declare_send_relative(pointer_send,idim,-1,nbytes);
	comm_start(mh_from_fwd[idim]);
	comm_start(mh_send_back[idim]);
	comm_wait(mh_send_back[idim]);
	comm_wait(mh_from_fwd[idim]);
		
	pointer_receive = NULL;
	pointer_send = NULL;

      }
    }

    for(int idim = 0 ; idim < N_DIMS ; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	comm_free(mh_send_fwd[idim]);
	comm_free(mh_from_fwd[idim]);
	comm_free(mh_send_back[idim]);
	comm_free(mh_from_back[idim]);
      }
    }
    
  }
}

template<typename Float>
void PLEGMA_Gauge<Float>::ghostToDevice(){
  if(comm_size() > 1){
    Float *host = PLEGMA_Field<Float>::h_ext_ghost;
    Float *device = PLEGMA_Field<Float>::d_elem+GK_localVolume*N_COLS*N_COLS*N_DIMS*2;
    cudaMemcpy(device,host,PLEGMA_Field<Float>::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void PLEGMA_Gauge<Float>::calculatePlaq(){
  
  ghostToHost();
  cpuExchangeGhost();
  ghostToDevice();
  
  gaugeTex<Float> tex;
  tex.tex = this->createTexObject();
  printfQuda("Calculated plaquette is %f\n",calculatePlaquette<Float>(tex));
  this->destroyTexObject(tex.tex);
}

template class PLEGMA_Gauge<float>;
template class PLEGMA_Gauge<double>;
