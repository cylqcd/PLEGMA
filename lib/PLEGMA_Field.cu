#include <PLEGMA_Field.h> 
#include <PLEGMA_shifts.cuh>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vector>
#include <algorithm>
using namespace plegma;
 
#define DEVICE_MEMORY_REPORT
#define CMPLX_FLOAT std::complex<Float>

//--------------------------//
// class PLEGMA_Field //
//--------------------------//

// This is is a class which allocates memory on either the
// the device, or host, or both for the structures:
// Field: one complex number per spacetime point.
// Gauge: one SU(3) link variable per spacetime point X spacetime dimension.
// Vector: 12 complex numbers (3colour x 4spin) per spacetime point.
// Vector3D: as above, but only defined on a single timeslice.
// Propagator: 12x12 complex matrix per sink position 
// (usually a whole spacetime volume.)
// Propagtor3D: as above, but with sinks only at one timeslice.

template<typename Float>
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT):
  h_elem(NULL), d_elem(NULL), h_ext_ghost(NULL), h_ext_ghost_corner(NULL),h_elem_backup(NULL), 
  allocation(alloc_flag), isAllocHost(false), isAllocDevice(false), isAllocHostBackup(false)

{
  if(GK_init_PLEGMA_flag == false) 
    errorQuda("You must initialize init_PLEGMA first");
  
  switch(classT){
  case FIELD:
    field_length = 1;
    total_length = GK_localVolume;
    break;
  case SU3FIELD:
    field_length = N_COLS * N_COLS;
    total_length = GK_localVolume;
    break;
  case GAUGE:
    field_length = N_DIMS * N_COLS * N_COLS;
    total_length = GK_localVolume;
    break;    
  case VECTOR:
    field_length = N_SPINS * N_COLS;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR:
    field_length = N_SPINS * N_COLS * N_SPINS * N_COLS;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR3D:
    field_length = N_SPINS * N_COLS * N_SPINS * N_COLS;
    total_length = GK_localVolume/GK_localL[3];
    break;
  case VECTOR3D:
    field_length = N_SPINS * N_COLS;
    total_length = GK_localVolume/GK_localL[3];
    break;
  }
  ghost_length = 0;
  ghost_corner_length = 0;
  
  for(int i = 0 ; i < N_DIMS ; i++){
    ghost_length += 2*GK_surface3D[i];
    for(int j = 0; j < N_DIMS; j++){
      ghost_corner_length += 2*GK_surface2D[i][j];
    }
  }
  total_plus_ghost_length = total_length + ghost_length;
  total_plus_ghost_corner_length = total_plus_ghost_length + ghost_corner_length;
  
  bytes_total_length = total_length*field_length*2*sizeof(Float);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(Float);
  bytes_total_plus_ghost_length = total_plus_ghost_length*field_length*2*sizeof(Float);
  bytes_ghost_corner_length = ghost_corner_length*field_length*2*sizeof(Float);
  bytes_total_plus_ghost_corner_length = total_plus_ghost_corner_length*field_length*2*sizeof(Float);

  if( alloc_flag == BOTH ){
    create_host();
    create_device();
  }
  else if (alloc_flag == HOST){
    create_host();
  }
  else if (alloc_flag == DEVICE){
    create_device();
  }
  else if (alloc_flag == BOTH_EXTRA){
    create_host();
    create_host_backup();
    create_device();    
  }

}

//Destructor
template<typename Float>
PLEGMA_Field<Float>::~PLEGMA_Field(){
  if(h_elem != NULL) destroy_host();
  if(h_elem_backup != NULL) destroy_host_backup();
  if(d_elem != NULL) destroy_device();
}

template<typename Float>
void PLEGMA_Field<Float>::pack( Float *topack ){
  for(int i=0; i<field_length; i++){
    for(int j=0; j<total_length; j++){
      for(int part=0; part<2; part++)
	h_elem[i*total_length*2 + j*2 + part] = topack[j*field_length*2 + i*2 + part];
    }
  }
}

template<typename Float>
void PLEGMA_Field<Float>::unpack(Float *out){
  for(int i=0; i<field_length; i++){
    for(int j=0; j<total_length; j++){
      for(int part=0; part<2; part++)
	out[j*field_length*2 + i*2 + part] = h_elem[i*total_length*2 + j*2 + part];
    }
  }  
}



template<typename Float>
void PLEGMA_Field<Float>::load(){
  cudaMemcpy(d_elem, h_elem, bytes_total_length, cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::unload(){
  cudaMemcpy(h_elem, d_elem, bytes_total_length, cudaMemcpyDeviceToHost);
  checkCudaError();
}


template<typename Float>
void PLEGMA_Field<Float>::create_host(){
  h_elem = (Float*) malloc(bytes_total_plus_ghost_length);
  h_ext_ghost = (Float*) malloc(bytes_ghost_length);
  h_ext_ghost_corner = (Float*) malloc(bytes_ghost_corner_length);
  if(h_elem == NULL || h_ext_ghost == NULL || h_ext_ghost_corner)
    errorQuda("Error with allocation host memory");
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void PLEGMA_Field<Float>::create_host_backup(){
  h_elem_backup = (Float*) malloc(bytes_total_plus_ghost_length);
  if(h_elem_backup == NULL) errorQuda("Error with allocation host memory");
  isAllocHostBackup = true;
  zero_host_backup();
}

template<typename Float>
void PLEGMA_Field<Float>::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_plus_ghost_length);
  checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  // device memory in MB
  GK_deviceMemory += bytes_total_length/(1024.*1024.);          
  printfQuda("Device memory in use is %f MB A PLEGMA \n",GK_deviceMemory);
#endif
  isAllocDevice = true;
  zero_device();
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_host(){
  free(h_elem);
  free(h_ext_ghost);
  free(h_ext_ghost_corner);
  h_elem=NULL;
  h_ext_ghost = NULL;
  h_ext_ghost_corner = NULL;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_host_backup(){
  free(h_elem_backup);
  h_elem=NULL;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_device(){
  cudaFree(d_elem);
  checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  GK_deviceMemory -= bytes_total_length/(1024.*1024.);
  printfQuda("Device memory in use is %f MB D PLEGMA\n",GK_deviceMemory);
#endif
}

template<typename Float>
void PLEGMA_Field<Float>::zero_host(){
  memset(h_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void PLEGMA_Field<Float>::zero_host_backup(){
  memset(h_elem_backup,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void PLEGMA_Field<Float>::zero_device(){
  cudaMemset(d_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void PLEGMA_Field<Float>::zero_where(ALLOCATION_FLAG alloc_flag){
  
  if( alloc_flag == BOTH ){
    zero_host();
    zero_device();
  }
  else if (alloc_flag == HOST){
    zero_host();
  }
  else if (alloc_flag == DEVICE){
    zero_device();
  }
  else if (alloc_flag == BOTH_EXTRA){
    zero_host();
    zero_host_backup();
    zero_device();    
  }
}

template<typename Float>
cudaTextureObject_t PLEGMA_Field<Float>::createTexObject(){
  cudaTextureObject_t tex;
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  int precision = PLEGMA_Field<Float>::Precision();
  if(precision == 4) desc.f = cudaChannelFormatKindFloat;
  else desc.f = cudaChannelFormatKindSigned;

  if(precision == 4){
    desc.x = 8*precision;
    desc.y = 8*precision;
    desc.z = 0;
    desc.w = 0;
  }
  else if(precision == 8){
    desc.x = 8*precision/2;
    desc.y = 8*precision/2;
    desc.z = 8*precision/2;
    desc.w = 8*precision/2;
  }

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_elem;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = bytes_total_plus_ghost_length;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
  checkCudaError();
  return tex;
}


template<typename Float>
void PLEGMA_Field<Float>::destroyTexObject(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
}

template<typename Float>
void PLEGMA_Field<Float>::printInfo(){
  printfQuda("This object has precision %d\n",Precision());
  printfQuda("This object needs %f Mb\n",
	     bytes_total_plus_ghost_length/(1024.*1024.));
  printfQuda("The flag for the host allocation is %d\n",(int) isAllocHost);
  printfQuda("The flag for the device allocation is %d\n",(int) isAllocDevice);
}

template<typename Float>
static void getOffsets(int dirOr, int field_length,int *pos,
		       int *height, size_t *width,
		       size_t *spitch, size_t *dpitch,int *ghostOffset){
  if( dirOr > -1 && dirOr < 2*N_DIMS ){
    *pos = (dirOr > N_DIMS) ? 0 : (GK_localL[dirOr]-1);
    *height = 1;
    *width = 2*sizeof(Float);
    *ghostOffset = ( dirOr < N_DIMS ) ? GK_minusGhost[dirOr%N_DIMS] : GK_plusGhost[dirOr%N_DIMS];
    for(int i=0; i<N_DIMS; i++){
      if( dirOr % N_DIMS > i ){
	*pos *= GK_localL[i];
	*width *= GK_localL[i];
      }
      else if( dirOr % N_DIMS < i ){
	*height *= GK_localL[i];
      }
    }
    *spitch = GK_localL[dirOr%N_DIMS] * (*width);
    *dpitch = *width;
    if(*height == 1) *height = field_length;
  }
  else{
    errorQuda("Direction should be in [0,%d] range",2*N_DIMS-1);
  }
}

template<typename Float>
static void getOffsetsCorner(int dir1, int dir2, int field_length,int *pos,
			     int *height, size_t *width, size_t *spitch,
			     size_t *dpitch,int *ghostOffset){
  if( (dir1 > -1 && dir1 < 2*N_DIMS) && (dir2 > -1 && dir2 < 2*N_DIMS) ){
    if( dir1 == dir2 ){ errorQuda("Directions must be different"); }
    *pos = (dir1 > N_DIMS && dir2 > N_DIMS) ? 0 :
      ( ( dir1 > N_DIMS ) ? (GK_localL[dir1]-1) :
	( dir2 > N_DIMS ) ? (GK_localL[dir2]-1) :
	(GK_localL[dir2])*(GK_localL[dir1])-1 );
    *height = 1;
    *width = 2*sizeof(Float);
    *ghostOffset = GK_cornerGhost[dir1][dir2];
    for(int i=0; i<N_DIMS; i++){
      for(int j=0; j<N_DIMS; j++){
	if( dir1 % N_DIMS > i && dir2 % N_DIMS > j ){
	  *pos *= GK_localL[i]*GK_localL[j];
	  *width *= GK_localL[i]*GK_localL[j];
	}
	else if( dir1 % N_DIMS > i && dir2 % N_DIMS < j ){
	  *pos *= GK_localL[i];
	  *width *= GK_localL[i];
	  *height *= GK_localL[j];
	}
	else if( dir1 % N_DIMS < i && dir2 % N_DIMS > j ){
	  *pos *= GK_localL[j];
	  *width *= GK_localL[j];
	  *height *= GK_localL[i];
	}
	else if( dir1 % N_DIMS < i && dir2 % N_DIMS < j ){
	  *height *= GK_localL[i]*GK_localL[j];
	}
      }
    }
    *spitch = GK_localL[dir1%N_DIMS] * GK_localL[dir2%N_DIMS] * (*width);
    *dpitch = *width;
    if(*height == 1) *height = field_length;
  }
  else{
    errorQuda("Directions should be in [0,%d] range",2*N_DIMS-1);
  }
}



template<typename Float>
void PLEGMA_Field<Float>::ghostToHost(int dirOr){
  if(dirOr<-1 || dirOr>7)
    errorQuda("Directions should be in [0,7] range with -1 all directions");
  bool isAll=(dirOr<0)?true:false;

  int position=0;
  int height=0;
  size_t width=0;
  size_t spitch=0;
  size_t dpitch=0;
  int ghostOffset=0;
  for(int ir = 0 ; ir <= 7 ; ir++)
    if(dirOr == ir || isAll){
      if(GK_localL[ir%4] < GK_totalL[ir%4]){
	Float *h_elem_offset = NULL;
	Float *d_elem_offset = NULL;
	getOffsets<Float>(ir,field_length,&position,&height,&width,&spitch,&dpitch,&ghostOffset);
	int N=(ir==3 || ir==4+3)?1:field_length;
	for(int i = 0 ; i < N; i++){
	  d_elem_offset = d_elem + i*total_length*2 + position*2;
	  h_elem_offset = h_elem + ghostOffset*field_length*2 + i*GK_surface3D[ir%4]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset, spitch,width,height,cudaMemcpyDeviceToHost);
	  checkCudaError();  
	}
      }
    }
}

template<typename Float>
void PLEGMA_Field<Float>::ghostCornerToHost(int dirOr){
  if(dirOr<-1 || dirOr>2*N_DIMS-1)
    errorQuda("Directions should be in [0,%d] range with -1 all directions",2*N_DIMS-1);
  bool isAll = (dirOr<0) ? true:false;

  int position=0;
  int height=0;
  size_t width=0;
  size_t spitch=0;
  size_t dpitch=0;
  int ghostOffset=0;

 for(int i=0; i<2*N_DIMS; i++){
   for(int j=i+1; j<2*N_DIMS; j++){
     if( (i%N_DIMS != j%N_DIMS ) && GK_dimBreak[i%N_DIMS] && GK_dimBreak[j%N_DIMS] ){
       if(dirOr == i || dirOr == j || isAll){
	 Float *h_elem_offset = NULL;
	 Float *d_elem_offset = NULL;
	 getOffsetsCorner<Float>(i,j,field_length,&position,&height,&width,&spitch,&dpitch,&ghostOffset);
	 int Ni = ( i==N_DIMS-1 || i==2*N_DIMS-1 ) ? 1 : field_length;
	 int Nj = ( j==N_DIMS-1 || j==2*N_DIMS-1 ) ? 1	: field_length;
	 for(int ii=0; ii<Ni; ii++){
	   for(int jj=0; jj<Nj; jj++){
	     d_elem_offset = d_elem + ii*Nj*total_length*2 + jj*total_length*2 + position*2;
	     h_elem_offset = h_elem + ghostOffset*field_length*2 + ii*Nj*GK_surface2D[i%N_DIMS][j%N_DIMS]*2 + jj*GK_surface2D[i%N_DIMS][j%N_DIMS]*2;
	     cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset, spitch,width,height,cudaMemcpyDeviceToHost);
	     checkCudaError();
	   }
	 }
       }
     }
   }
 }
}

template<typename Float>
void PLEGMA_Field<Float>::cpuExchangeGhost(int dirOr){

  if(dirOr<-1 || dirOr>7)
    errorQuda("Directions should be in [0,7] range with -1 all directions");
  bool isAll = (dirOr<0) ? true:false;

  MsgHandle *mh_send_fwd[4];
  MsgHandle *mh_from_back[4];
  MsgHandle *mh_from_fwd[4];
  MsgHandle *mh_send_back[4];

  Float *pointer_receive = NULL;
  Float *pointer_send = NULL;

  for(int ir = 0 ; ir <= 7 ; ir++)
    if(dirOr == ir || isAll){
      if(GK_localL[ir%4] < GK_totalL[ir%4]){
	size_t nbytes = GK_surface3D[ir%4]*field_length*2*sizeof(Float);
	int ghost = ir<4?GK_minusGhost[ir%4]:GK_plusGhost[ir%4];
	pointer_receive=h_ext_ghost + (ghost-total_length)*field_length*2;
	pointer_send=h_elem + ghost*field_length*2;
	if(ir<4){
	  mh_from_back[ir%4] = comm_declare_receive_relative(pointer_receive,ir%4,-1,nbytes);
	  mh_send_fwd[ir%4] = comm_declare_send_relative(pointer_send,ir%4,1,nbytes);
	  comm_start(mh_from_back[ir%4]);
	  comm_start(mh_send_fwd[ir%4]);
	  comm_wait(mh_send_fwd[ir%4]);
	  comm_wait(mh_from_back[ir%4]);

	  comm_free(mh_from_back[ir%4]);
	  comm_free(mh_send_fwd[ir%4]);
	}
	else{
	  mh_from_fwd[ir%4] = comm_declare_receive_relative(pointer_receive,ir%4,1,nbytes);
	  mh_send_back[ir%4] = comm_declare_send_relative(pointer_send,ir%4,-1,nbytes);
	  comm_start(mh_from_fwd[ir%4]);
	  comm_start(mh_send_back[ir%4]);
	  comm_wait(mh_send_back[ir%4]);
	  comm_wait(mh_from_fwd[ir%4]);

	  comm_free(mh_from_fwd[ir%4]);
	  comm_free(mh_send_back[ir%4]);
	}
      }
    }    
}

template<typename Float>
void PLEGMA_Field<Float>::cpuExchangeGhostCorner(int dirOr){

  if(dirOr<-1 || dirOr>2*N_DIMS-1)
    errorQuda("Directions should be in [0,%d] range with -1 all directions",2*N_DIMS-1);
  bool isAll = (dirOr<0) ? true:false;

  MsgHandle *mh_send;
  MsgHandle *mh_recv;
    
  Float *pointer_receive = NULL;
  Float *pointer_send = NULL;

  int disp[N_DIMS] = {0};
  
  for(int i=0; i<2*N_DIMS; i++){
    for(int j=i+1; j<2*N_DIMS; j++){
      if( (i%N_DIMS != j%N_DIMS ) && GK_dimBreak[i%N_DIMS] && GK_dimBreak[j%N_DIMS] ){
	if(dirOr == i || dirOr == j || isAll){
	  size_t nbytes = GK_surface2D[i][j]*field_length*2*sizeof(Float);
	  // negative second direction communication
	  int ghost =  GK_cornerGhost[i][j];
	  pointer_receive = h_ext_ghost_corner + (ghost-total_length)*field_length*2;
	  pointer_send = h_elem + ghost*field_length*2;
	  disp[i%N_DIMS] = (i<N_DIMS) ? -1 : 1;
	  disp[j%N_DIMS] = (j<N_DIMS) ? -1 : 1;
	  mh_recv = comm_declare_receive_displaced(pointer_receive,disp,nbytes); 
	  disp[i%N_DIMS] *= -1;
	  disp[j%N_DIMS] *= -1;
	  mh_send = comm_declare_send_displaced(pointer_send,disp,nbytes);
	  comm_start(mh_recv);
	  comm_start(mh_send);
	  comm_wait(mh_send);
	  comm_wait(mh_recv);
	  comm_free(mh_recv);
	  comm_free(mh_send);
	  disp[i%N_DIMS] = 0; disp[j%N_DIMS] = 0;	  
	}
      }
    }
  }
}

template<typename Float>
void PLEGMA_Field<Float>::ghostToDevice(){
  if(comm_size() > 1){
    Float *host = h_ext_ghost;
    Float *device = d_elem+GK_localVolume*field_length*2;
    cudaMemcpy(device,host,bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void PLEGMA_Field<Float>::ghostCornerToDevice(){
  if(comm_size() > 1){
    Float *hostCorner = h_ext_ghost_corner;
    Float *device = d_elem+GK_localVolume*field_length*2+bytes_ghost_length;
    cudaMemcpy(device,hostCorner,bytes_ghost_corner_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}


template<typename Float>
void PLEGMA_Field<Float>::shift(PLEGMA_Field<Float> &Fin, int dirOr){
  // we have to make sure that we have the ghost
  Fin.ghostToHost(dirOr);
  Fin.cpuExchangeGhost(dirOr);
  Fin.ghostToDevice();
  shiftField(Fin,*this,dirOr);
}

template<typename Float>
void PLEGMA_Field<Float>::setUnit(std::vector<int> indDiag){
  /* 
   * Set specific indices of Field to one as provided from indOne
   * Example: For Su3 field indOne ={0,4,8};
   */
  for(int i = 0 ; i < Field_length(); i++){
    std::vector<int>::iterator it = std::find(indDiag.begin(), indDiag.end(), i);
    thrust::device_ptr<Float2<Float> > dev_ptr( (Float2<Float>*) (this->D_elem() + i*(this->Total_length())*2));
    Float2<Float> value;
    value.y=0.;
    value.x=(it != indDiag.end() )?1.:0.;
    thrust::fill(dev_ptr, dev_ptr + this->Total_length(), value);
  }
}

template class PLEGMA_Field<float>;
template class PLEGMA_Field<double>;
