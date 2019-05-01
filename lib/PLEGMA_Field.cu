#include <PLEGMA_Field.h> 
#include <PLEGMA_Thrust.h>
#include <PLEGMA_field_utils.cuh>
#include <PLEGMA_shifts.cuh>
#include <PLEGMA_Random.h>
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
void PLEGMA_Field<Float>::
initialize(ALLOCATION_FLAG alloc_flag, int field_l, size_t vol_l, GHOST_FLAG ghost_flag) {
  if(HGC_init_PLEGMA_flag == false) 
    PLEGMA_error("You must initialize init_PLEGMA first");

  field_length = field_l;
  total_length = vol_l;
  
  ghost_length = 0;
  ghost_corner_length = 0;

  for(int i = 0 ; i < N_DIMS ; i++){
    if(ghost_flag >= FIRST_SIDE) ghost_length += 2*HGC_surface3D[i];
    for(int j = i+1; j < N_DIMS; j++){
      if(ghost_flag >= FIRST_CORNER) ghost_corner_length += 4*HGC_surface2D[i][j];
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
    PLEGMA_error("Error not supported %d\n",alloc_flag);
  }
}

template<typename Float>
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, int site_size, GHOST_FLAG ghost_flag, bool D3):
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag), isAllocHost(false), isAllocDevice(false), field_type(CUSTOM)
{
  initialize(alloc_flag, site_size, D3 ? HGC_localVolume3D : HGC_localVolume, ghost_flag);
}

template<typename Float>
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag):
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag), isAllocHost(false), isAllocDevice(false), field_type(classT)
{
  if(HGC_init_PLEGMA_flag == false) 
    PLEGMA_error("You must initialize init_PLEGMA first");

  switch(classT){
    case SCALAR:
      initialize(alloc_flag, 1, HGC_localVolume, ghost_flag);
      break;
    case SU3FIELD:
      initialize(alloc_flag, N_COLS * N_COLS, HGC_localVolume, ghost_flag);
      break;
    case GAUGE:
      initialize(alloc_flag, N_DIMS * N_COLS * N_COLS, HGC_localVolume, ghost_flag);
      break;    
    case VECTOR:
      initialize(alloc_flag, N_SPINS * N_COLS, HGC_localVolume, ghost_flag);
      break;
    case PROPAGATOR:
      initialize(alloc_flag, N_SPINS * N_COLS * N_SPINS * N_COLS, HGC_localVolume, ghost_flag);
      break;
    case PROPAGATOR3D:
      initialize(alloc_flag, N_SPINS * N_COLS * N_SPINS * N_COLS, HGC_localVolume3D, ghost_flag);
      break;
    case VECTOR3D:
      initialize(alloc_flag, N_SPINS * N_COLS, HGC_localVolume3D, ghost_flag);
      break;
    case QLOOPS:
      initialize(alloc_flag, N_SPINS * N_SPINS, HGC_localVolume, ghost_flag);
      break;
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
  if(allocation != BOTH) PLEGMA_error("Load from Host to Device needs BOTH allocation");
  cudaMemcpy(d_elem, h_elem, bytes_total_length, cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::unload(){
  if(allocation != BOTH) PLEGMA_error("Load from Host to Device needs BOTH allocation");
  cudaMemcpy(h_elem, d_elem, bytes_total_length, cudaMemcpyDeviceToHost);
  checkCudaError();
}


template<typename Float>
void PLEGMA_Field<Float>::create_host(){
  hostMalloc(h_elem, bytes_total_plus_ghost_length);
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void PLEGMA_Field<Float>::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_plus_ghost_length);
  checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  // device memory in MB
  HGC_deviceMemory += bytes_total_plus_ghost_length/(1024.*1024.);          
  if(HGC_verbosity>1) PLEGMA_printf("Device memory in use is %f MB A PLEGMA \n",HGC_deviceMemory);
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
  hostFree(h_elem, bytes_total_plus_ghost_length);
  isAllocHost=false;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_device(){
  cudaFree(d_elem);
  checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  HGC_deviceMemory -= bytes_total_plus_ghost_length/(1024.*1024.);
  if(HGC_verbosity>1) PLEGMA_printf("Device memory in use is %f MB D PLEGMA\n",HGC_deviceMemory);
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
    PLEGMA_error("Not supported %d\n",alloc_flag);
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
  PLEGMA_printf("This object has precision %d\n",Precision());
  PLEGMA_printf("This object needs %f Mb\n",
      bytes_total_plus_ghost_length/(1024.*1024.));
  PLEGMA_printf("The flag for the host allocation is %d\n",(int) isAllocHost);
  PLEGMA_printf("The flag for the device allocation is %d\n",(int) isAllocDevice);
}

template<typename Float>
void PLEGMA_Field<Float>::communicateSideGhost(int dirOr){
  if(comm_size() == 1)
    return;
  if(ghost_flag < FIRST_SIDE)
    PLEGMA_error("First side ghosts have not been allocated.\n");
  if(dirOr<-1 || dirOr>2*N_DIMS-1)
    PLEGMA_error("Directions should be in [-1,%d] range with -1 all directions",2*N_DIMS-1);

  bool isAll = (dirOr<0) ? true:false;

  std::vector<MsgHandle *> mh_recv;
  std::vector<MsgHandle *> mh_send;

  for(int i=0; i<2*N_DIMS; i++){
    if( HGC_dimBreak[i%N_DIMS] ){
      if(dirOr == i || isAll){
        Float *pointer_receive = h_ext_ghost_r + (HGC_sideGhost[i]-total_length)*field_length*2;
        Float *pointer_send = h_ext_ghost_s + (HGC_sideGhost[i]-total_length)*field_length*2;
        Float *pointer_device = d_elem + HGC_sideGhost[i]*field_length*2;
        int disp[N_DIMS] = {0};
        size_t nbytes = HGC_surface3D[i%N_DIMS]*field_length*2*sizeof(Float);

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
    if( HGC_dimBreak[dirOr%N_DIMS] ){
      Float *host = h_ext_ghost_r + (HGC_sideGhost[dirOr]-total_length)*field_length*2;
      Float *device = d_elem + HGC_sideGhost[dirOr]*field_length*2;
      cudaMemcpy(device, host, HGC_surface3D[dirOr%N_DIMS]*field_length*2*sizeof(Float),
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
    PLEGMA_error("First corner ghosts have not been allocated.\n");
  if(dirOr<-1 || dirOr>2*N_DIMS-1)
    PLEGMA_error("Directions should be in [-1,%d] range with -1 all directions",2*N_DIMS-1);

  bool isAll = (dirOr<0) ? true:false;

  std::vector<MsgHandle *> mh_recv;
  std::vector<MsgHandle *> mh_send;

  for(int i=0; i<2*N_DIMS; i++){
    for(int j=i+1; j<2*N_DIMS; j++){
      if( (i%N_DIMS != j%N_DIMS ) && HGC_dimBreak[i%N_DIMS] && HGC_dimBreak[j%N_DIMS] ){
        if(dirOr == i || dirOr == j || isAll){
          Float *pointer_receive = h_ext_ghost_corner_r + (HGC_cornerGhost[i][j]-total_length-ghost_length)*field_length*2;
          Float *pointer_send = h_ext_ghost_corner_s + (HGC_cornerGhost[i][j]-total_length-ghost_length)*field_length*2;
          Float *pointer_device = d_elem + HGC_cornerGhost[i][j]*field_length*2;
          int disp[N_DIMS] = {0};
          size_t nbytes = HGC_surface2D[i%N_DIMS][j%N_DIMS]*field_length*2*sizeof(Float);

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
        if( (i%N_DIMS != j%N_DIMS ) && HGC_dimBreak[i%N_DIMS] && HGC_dimBreak[j%N_DIMS] ){
          if(dirOr == i || dirOr == j || isAll){
            Float *hostCorner = h_ext_ghost_corner_r + (HGC_cornerGhost[i][j]-total_length-ghost_length)*field_length*2;
            Float *device = d_elem+HGC_cornerGhost[i][j]*field_length*2;
            cudaMemcpy(device, hostCorner, HGC_surface2D[i%N_DIMS][j%N_DIMS]*field_length*2*sizeof(Float),
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
    PLEGMA_error("Asking to communicate ghost but they have not been allocated.\n");
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
      PLEGMA_error("This value of n has not been compiled. Come here to add it");
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
  if(!isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
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
  if(!isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
  if(sign != +1 && sign != -1) PLEGMA_error("Sign should be either +1 or -1\n");
  if(mom.size() != 3 && mom.size() != 4) PLEGMA_error("Momentum size vector should be either 3 or 4\n");
  if(total_length == HGC_localVolume && mom.size() != 4 ) PLEGMA_error("A 4D field needs a 4D momentum vector\n");
  if( (total_length == HGC_localVolume3D) && mom.size() != 3 ) PLEGMA_error("A 3D field needs a 3D momentum vector\n");
  int D3D4 = mom.size();
  int V = D3D4 == 3 ? HGC_localVolume3D : HGC_localVolume;
  Float2<Float> *x;
  cudaMalloc((void**)&x, V*2*sizeof(Float));
  cudaMemset((void*) x,0,V*2*sizeof(Float));
  checkCudaError();
  createMomField(x, mom, D3D4, sign);
  for(int dof = 0; dof < field_length; dof++)
    plegma::elemWiseMul(V,(Float*) x, d_elem + dof*total_length*2);
  cudaFree(x);
}

// y=a*x+y
template<typename Float>
void PLEGMA_Field<Float>::add(PLEGMA_Field<Float> &fieldIn, std::complex<Float> alpha){
  Float a[2]; a[0]=alpha.real(); a[1]=alpha.imag();
  cuBLAS::axpy(total_length*field_length, a, fieldIn.D_elem(), d_elem);
}


template<typename Float>
std::complex<Float> PLEGMA_Field<Float>::dot(PLEGMA_Field<Float> &fieldIn){
  return cuBLAS::dot(total_length*field_length, d_elem, fieldIn.D_elem(), MPI_COMM_WORLD);
}

template<typename Float>
Float PLEGMA_Field<Float>::norm(){
  return cuBLAS::norm(total_length*field_length, d_elem, MPI_COMM_WORLD);
}

template<typename Float>
void PLEGMA_Field<Float>::cscale(std::complex<Float> val){
  if(!isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
  cuBLAS::cscal(field_length*total_length, reinterpret_cast<Float(&)[2]>(val), d_elem );
}

template<typename FloatOut, typename FloatIn>
static void cudaCopyOrCast(PLEGMA_Field<FloatOut> &fieldOut, PLEGMA_Field<FloatIn> &fieldIn){
  if(typeid(FloatIn) != typeid(FloatOut) )
    cudaCast(fieldOut.D_elem(), fieldIn.D_elem(), fieldIn.Bytes_total()/sizeof(FloatIn));
  else
    cudaMemcpy(fieldOut.D_elem(), fieldIn.D_elem(), fieldIn.Bytes_total(), 
	       cudaMemcpyDeviceToDevice);
  checkCudaError();
}

template<typename FloatOut, typename FloatIn>
static void hostCopyOrCast(PLEGMA_Field<FloatOut> &fieldOut, PLEGMA_Field<FloatIn> &fieldIn){
  if(typeid(FloatIn) != typeid(FloatOut) )
    for(size_t i = 0; i<fieldIn.Bytes_total()/sizeof(FloatIn); i++)
      fieldOut.H_elem()[i] = (FloatOut) fieldIn.H_elem()[i];
  else
    memcpy(fieldOut.H_elem(), fieldIn.H_elem(), fieldIn.Bytes_total());
}

template<typename FloatOut>
template<typename FloatIn>
void PLEGMA_Field<FloatOut>::copy(PLEGMA_Field<FloatIn> &f, ALLOCATION_FLAG where){
  if(field_length != f.Field_length()) PLEGMA_error("The d.o.f of the fields does not match\n");
  switch(where){
  case(HOST):
    if(!isAllocHost || !f.IsAllocHost() ) PLEGMA_error("Allocation flags do not match for copying\n");
    hostCopyOrCast(*this, f);
    break;
  case(DEVICE):
    if(!isAllocDevice || !f.IsAllocDevice() ) PLEGMA_error("Allocation flags do not match for copying\n");
    cudaCopyOrCast(*this, f);
    break;
  case(BOTH):
    if(!isAllocHost || !f.IsAllocHost() ) PLEGMA_error("Allocation flags do not match for copying\n");
    hostCopyOrCast(*this, f);
    if(!isAllocDevice || !f.IsAllocDevice() ) PLEGMA_error("Allocation flags do not match for copying\n");
    cudaCopyOrCast(*this, f);
    break;
  }
}

template<typename Float>
void PLEGMA_Field<Float>::applyHpropColoring4D(PLEGMA_Field<Float> &fin,PLEGMA_Hprobing &hprob, int ih, std::vector<int> indDof){
  if(total_length != HGC_localVolume || fin.Total_length() != HGC_localVolume) PLEGMA_error("Probing for now works only for 4D fields");
  if(ih >= hprob.get_NHad()) PLEGMA_error("You have exceeded the size of the Hadamard matrix");
  copy(fin,DEVICE);
  for(int i = 0 ; i < Field_length(); i++){
    std::vector<int>::iterator it = std::find(indDof.begin(), indDof.end(), i);
    if(it != indDof.end()){
      apply_hprob_coloring_4D(D_elem() + i*total_length*2, hprob.D_arrVc(), ih);
    }
  }
}

template class PLEGMA_Field<float>;
template class PLEGMA_Field<double>;
// Forcing initialization of the following cases
template void PLEGMA_Field<float>::copy<float>(PLEGMA_Field<float> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<float>::copy<double>(PLEGMA_Field<double> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<double>::copy<float>(PLEGMA_Field<float> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<double>::copy<double>(PLEGMA_Field<double> &f, ALLOCATION_FLAG where);
