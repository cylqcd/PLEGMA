#include <PLEGMA_Field.h> 
#include <PLEGMA_field_utils.cuh>
#include <PLEGMA_shifts.cuh>
#include <PLEGMA_Random.h>
#include <PLEGMA_Thrust.h>
#include <vector>
#include <algorithm>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_FT.cuh>

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
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag):
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag), isAllocHost(false), isAllocDevice(false)

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
    case QLOOPS:
      field_length = N_SPINS * N_SPINS;
      total_length = GK_localVolume;
      break;
  }
  ghost_length = 0;
  ghost_corner_length = 0;



  for(int i = 0 ; i < N_DIMS ; i++){
    if(ghost_flag >= FIRST_SIDE) ghost_length += 2*GK_surface3D[i];
    for(int j = i+1; j < N_DIMS; j++){
      if(ghost_flag >= FIRST_CORNER) ghost_corner_length += 4*GK_surface2D[i][j];
    }
  }
  total_plus_ghost_length = total_length + ghost_length + ghost_corner_length;

  bytes_total_length = total_length*field_length*2*sizeof(Float);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(Float);
  bytes_ghost_corner_length = ghost_corner_length*field_length*2*sizeof(Float);
  bytes_total_plus_ghost_length = total_plus_ghost_length*field_length*2*sizeof(Float);

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
  else{
    errorQuda("Error not supported %d\n",alloc_flag);
  }
}

//Destructor
template<typename Float>
PLEGMA_Field<Float>::~PLEGMA_Field(){
  if(isAllocHost) destroy_host();
  if(isAllocDevice) destroy_device();
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
  if(h_elem == NULL)
    errorQuda("Error with allocation host memory");
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void PLEGMA_Field<Float>::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_plus_ghost_length);
  checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  // device memory in MB
  GK_deviceMemory += bytes_total_plus_ghost_length/(1024.*1024.);          
  printfQuda("Device memory in use is %f MB A PLEGMA \n",GK_deviceMemory);
#endif
  zero_device();
  if(ghost_flag >= FIRST_SIDE){
    cudaMallocHost((void**)&h_ext_ghost_r, bytes_ghost_length);
    cudaMallocHost((void**)&h_ext_ghost_s, bytes_ghost_length);
  }
  if(ghost_flag == FIRST_CORNER){
    cudaMallocHost((void**)&h_ext_ghost_corner_r, bytes_ghost_corner_length);
    cudaMallocHost((void**)&h_ext_ghost_corner_s, bytes_ghost_corner_length);
  }
  checkCudaError();
  isAllocDevice = true;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_host(){
  free(h_elem);
  h_elem=NULL;
  isAllocHost=false;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_device(){
  cudaFree(d_elem);
  checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  GK_deviceMemory -= bytes_total_plus_ghost_length/(1024.*1024.);
  printfQuda("Device memory in use is %f MB D PLEGMA\n",GK_deviceMemory);
#endif
  if(ghost_flag >= FIRST_SIDE){
    cudaFreeHost(h_ext_ghost_r); h_ext_ghost_r=NULL;
    cudaFreeHost(h_ext_ghost_s); h_ext_ghost_s=NULL;
  }
  if(ghost_flag == FIRST_CORNER){
    cudaFreeHost(h_ext_ghost_corner_r); h_ext_ghost_corner_r=NULL;
    cudaFreeHost(h_ext_ghost_corner_s); h_ext_ghost_corner_s=NULL;
  }
  checkCudaError();
  isAllocDevice=false;
}

template<typename Float>
void PLEGMA_Field<Float>::zero_host(){
  if(isAllocHost)memset(h_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void PLEGMA_Field<Float>::zero_device(){
  if(isAllocDevice)cudaMemset(d_elem,0,bytes_total_plus_ghost_length);
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
  else{
    errorQuda("Not supported %d\n",alloc_flag);
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
void PLEGMA_Field<Float>::communicateSideGhost(int dirOr){
  if(comm_size() == 1)
    return;
  if(ghost_flag < FIRST_SIDE)
    errorQuda("First side ghosts have not been allocated.\n");
  if(dirOr<-1 || dirOr>2*N_DIMS-1)
    errorQuda("Directions should be in [-1,%d] range with -1 all directions",2*N_DIMS-1);

  bool isAll = (dirOr<0) ? true:false;

  std::vector<MsgHandle *> mh_recv;
  std::vector<MsgHandle *> mh_send;

  for(int i=0; i<2*N_DIMS; i++){
    if( GK_dimBreak[i%N_DIMS] ){
      if(dirOr == i || isAll){
        Float *pointer_receive = h_ext_ghost_r + (GK_sideGhost[i]-total_length)*field_length*2;
        Float *pointer_send = h_ext_ghost_s + (GK_sideGhost[i]-total_length)*field_length*2;
        Float *pointer_device = d_elem + GK_sideGhost[i]*field_length*2;
        int disp[N_DIMS] = {0};
        size_t nbytes = GK_surface3D[i%N_DIMS]*field_length*2*sizeof(Float);

        // collecting elements from device
        copy_side_to_ghost(*this, i);
        cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
        checkCudaError();

        // communicating
        disp[i%N_DIMS] = (i<N_DIMS) ? +1 : -1;
        mh_recv.push_back(comm_declare_receive_displaced(pointer_receive,disp,nbytes)); 
        disp[i%N_DIMS] *= -1;
        mh_send.push_back(comm_declare_send_displaced(pointer_send,disp,nbytes));
        disp[i%N_DIMS] = 0;
        comm_start(mh_recv.back());
        comm_start(mh_send.back());
      }
    }
  }
  // waiting for communications
  while (! mh_recv.empty()) {
    comm_wait(mh_recv.back());
    comm_wait(mh_send.back());
    comm_free(mh_recv.back());
    comm_free(mh_send.back());
    mh_recv.pop_back();
    mh_send.pop_back();
  }
  //copying to device
  if(isAll) {
    Float *host = h_ext_ghost_r;
    Float *device = d_elem+total_length*field_length*2;
    cudaMemcpy(device, host, bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  } else {
    if( GK_dimBreak[dirOr%N_DIMS] ){
      Float *host = h_ext_ghost_r + (GK_sideGhost[dirOr]-total_length)*field_length*2;
      Float *device = d_elem + GK_sideGhost[dirOr]*field_length*2;
      cudaMemcpy(device, host, GK_surface3D[dirOr%N_DIMS]*field_length*2*sizeof(Float),
          cudaMemcpyHostToDevice);
      checkCudaError();
    }
  }
}


template<typename Float>
void PLEGMA_Field<Float>::communicateCornerGhost(int dirOr){
  if(comm_size() == 1)
    return;
  if(ghost_flag < FIRST_CORNER)
    errorQuda("First corner ghosts have not been allocated.\n");
  if(dirOr<-1 || dirOr>2*N_DIMS-1)
    errorQuda("Directions should be in [-1,%d] range with -1 all directions",2*N_DIMS-1);

  bool isAll = (dirOr<0) ? true:false;

  std::vector<MsgHandle *> mh_recv;
  std::vector<MsgHandle *> mh_send;

  for(int i=0; i<2*N_DIMS; i++){
    for(int j=i+1; j<2*N_DIMS; j++){
      if( (i%N_DIMS != j%N_DIMS ) && GK_dimBreak[i%N_DIMS] && GK_dimBreak[j%N_DIMS] ){
        if(dirOr == i || dirOr == j || isAll){
          Float *pointer_receive = h_ext_ghost_corner_r + (GK_cornerGhost[i][j]-total_length-ghost_length)*field_length*2;
          Float *pointer_send = h_ext_ghost_corner_s + (GK_cornerGhost[i][j]-total_length-ghost_length)*field_length*2;
          Float *pointer_device = d_elem + GK_cornerGhost[i][j]*field_length*2;
          int disp[N_DIMS] = {0};
          size_t nbytes = GK_surface2D[i%N_DIMS][j%N_DIMS]*field_length*2*sizeof(Float);

          // collecting elements from device
          copy_corner_to_ghost(*this, i, j);
          cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
          checkCudaError();

          // communicating
          disp[i%N_DIMS] = (i<N_DIMS) ? +1 : -1;
          disp[j%N_DIMS] = (j<N_DIMS) ? +1 : -1;
          mh_recv.push_back(comm_declare_receive_displaced(pointer_receive,disp,nbytes)); 
          disp[i%N_DIMS] *= -1;
          disp[j%N_DIMS] *= -1;
          mh_send.push_back(comm_declare_send_displaced(pointer_send,disp,nbytes));
          disp[i%N_DIMS] = 0; disp[j%N_DIMS] = 0;	  
          comm_start(mh_recv.back());
          comm_start(mh_send.back());
        }
      }
    }
  }
  // waiting for communications
  while (! mh_recv.empty()) {
    comm_wait(mh_recv.back());
    comm_wait(mh_send.back());
    comm_free(mh_recv.back());
    comm_free(mh_send.back());
    mh_recv.pop_back();
    mh_send.pop_back();
  }
  //copying to device
  if(isAll) {
    Float *hostCorner = h_ext_ghost_corner_r;
    Float *device = d_elem+(total_length+ghost_length)*field_length*2;
    cudaMemcpy(device,hostCorner,bytes_ghost_corner_length,cudaMemcpyHostToDevice);
    checkCudaError();
  } else {
    for(int i=0; i<2*N_DIMS; i++){
      for(int j=i+1; j<2*N_DIMS; j++){
        if( (i%N_DIMS != j%N_DIMS ) && GK_dimBreak[i%N_DIMS] && GK_dimBreak[j%N_DIMS] ){
          if(dirOr == i || dirOr == j || isAll){
            Float *hostCorner = h_ext_ghost_corner_r + (GK_cornerGhost[i][j]-total_length-ghost_length)*field_length*2;
            Float *device = d_elem+GK_cornerGhost[i][j]*field_length*2;
            cudaMemcpy(device, hostCorner, GK_surface2D[i%N_DIMS][j%N_DIMS]*field_length*2*sizeof(Float),
                cudaMemcpyHostToDevice);
            checkCudaError();
          }
        }
      }
    }
  }
}

template<typename Float>
void PLEGMA_Field<Float>::communicateGhost(int dirOr, GHOST_FLAG which_ghost){
  if(ghost_flag < which_ghost) {
    errorQuda("Asking to communicate ghost but they have not been allocated.\n");
  }
  if(which_ghost >= FIRST_SIDE){
    communicateSideGhost(dirOr);
  }
  if(which_ghost >= FIRST_CORNER){
    communicateCornerGhost(dirOr);
  }
}

template<typename Float>
void PLEGMA_Field<Float>::communicateGhost(int dirOr){
  this->communicateGhost(dirOr, ghost_flag);
}

template<typename Float>
void PLEGMA_Field<Float>::shift(PLEGMA_Field<Float> &Fin, int dirOr){
  // we have to make sure that we have the ghost
  Fin.communicateSideGhost((dirOr+N_DIMS)%(2*N_DIMS));
  shiftField(Fin,*this,dirOr);
}

template<typename Float>
void PLEGMA_Field<Float>::randInit(int seed){

  randstate_ptr = new PLEGMA_RNG(seed, total_length);
  checkCudaError();  
}

template<typename Float>
void PLEGMA_Field<Float>::stochastic_Z(int n){
  this->zero_device();

  //printf("Array of random numbers not allocated, array size: %d !\nExiting...\n",this->field_length * this->total_length);
  int rng_size = this->total_length;
  switch( n ){
    case 2:
      set_stochastic<Float, 2>( *randstate_ptr, *this, this->field_length, rng_size);
      break;
    case 3:
      set_stochastic<Float, 3>( *randstate_ptr, *this, this->field_length, rng_size);
      break;
    case 4:
      set_stochastic<Float, 4>( *randstate_ptr, *this, this->field_length, rng_size);
      break;
    default:
      errorQuda("This value of n has not been compiled. Come here to add it");
  }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::random(DIST sampling){
  this->zero_device();
  //printf("Array of random numbers not allocated, array size: %d !\nExiting...\n",this->field_length * this->total_length);
  int rng_size = this->total_length;
  set_random<Float>( *randstate_ptr, *this, this->field_length, rng_size, sampling);    
}


template<typename Float>
void PLEGMA_Field<Float>::setUnit(std::vector<int> indDiag){
  /* 
   * Set specific indices of Field to one as provided from indOne
   * Example: For Su3 field indOne ={0,4,8};
   */
  if(!isAllocDevice) errorQuda("This function needs allocation on the device to work\n");
  for(int i = 0 ; i < Field_length(); i++){
    std::vector<int>::iterator it = std::find(indDiag.begin(), indDiag.end(), i);
    thrust::device_ptr<Float2<Float> > dev_ptr( (Float2<Float>*) (this->D_elem() + i*(this->Total_length())*2));
    Float2<Float> value;
    value.y=0.;
    value.x=(it != indDiag.end() )?1.:0.;
    thrust::fill(dev_ptr, dev_ptr + this->Total_length(), value);
  }
}

template<typename Float>
void PLEGMA_Field<Float>::mulMomentumPhases(std::vector<int> mom, int sign){
  if(!isAllocDevice) errorQuda("This function needs allocation on the device to work\n");
  if(sign != +1 && sign != -1) errorQuda("Sign should be either +1 or -1\n");
  if(mom.size() != 3 && mom.size() != 4) errorQuda("Momentum size vector should be either 3 or 4\n");
  if(total_length == GK_localVolume && mom.size() != 4 ) errorQuda("A 4D field needs a 4D momentum vector\n");
  if( (total_length == GK_localVolume/GK_localL[3]) && mom.size() != 3 ) errorQuda("A 3D field needs a 3D momentum vector\n");
  int D3D4 = mom.size();
  int V = D3D4 == 3 ? GK_localVolume/GK_localL[3] : GK_localVolume;
  Float2<Float> *x;
  cudaMalloc((void**)&x, V*2*sizeof(Float));
  cudaMemset((void*) x,0,V*2*sizeof(Float));
  checkCudaError();
  createMomField(x, mom, D3D4, sign);
  for(int dof = 0; dof < field_length; dof++)
    plegma::elemWiseMul(V,(Float*) x, d_elem + dof*total_length*2);
  cudaFree(x);
}

template<typename Float>
void PLEGMA_Field<Float>::dot(Float res[2], PLEGMA_Field<Float> &fieldIn){
  
  cuBLAS::dot(res, total_length*field_length, PLEGMA_Field<Float>::d_elem, fieldIn.D_elem(), MPI_COMM_WORLD);
  printfQuda("Vector dot product is %e %e\n",res[0], res[1]);  
}
template class PLEGMA_Field<float>;
template class PLEGMA_Field<double>;
