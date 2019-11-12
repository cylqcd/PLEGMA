#include <PLEGMA_Field.h> 
#include <PLEGMA_Thrust.h>
#include <PLEGMA_field_utils.cuh>
#include <PLEGMA_shifts.cuh>
#include <PLEGMA_Random.h>
#include <vector>
#include <algorithm>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_FT.cuh>
#include <utils/PLEGMA_auxiliary.h>
#include <io/PLEGMA_lime.h>
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
initialize(ALLOCATION_FLAG alloc_flag, int field_l, size_t vol_l) {
  if(HGC_init_PLEGMA_flag == false) 
    PLEGMA_error("You must initialize init_PLEGMA first");

  field_length = field_l;
  total_length = vol_l;
  site_shape = {field_l};

  ghost_length = 0;
  ghost_corner_length = 0;
  
  if(ghost_flag>NO_GHOSTS) {
    assert(vol_l==HGC_localVolume || vol_l==HGC_localVolume3D);

    if(vol_l==HGC_localVolume) {
      for(int i = 0 ; i < N_DIMS ; i++){
	if(ghost_flag >= FIRST_SIDE) ghost_length += 2*HGC_surface3D[i];
	for(int j = i+1; j < N_DIMS; j++){
	  if(ghost_flag >= FIRST_CORNER) ghost_corner_length += 4*HGC_surface2D[i][j];
	}
      }
    } else if(vol_l==HGC_localVolume3D) {
      for(int i = 0 ; i < N_DIMS-1 ; i++){
	if(ghost_flag >= FIRST_SIDE) ghost_length += 2*HGC_surface3D[i]/HGC_localL[DIM_T];
	for(int j = i+1; j < N_DIMS-1; j++){
	  if(ghost_flag >= FIRST_CORNER) ghost_corner_length += 4*HGC_surface2D[i][j]/HGC_localL[DIM_T];
	}
      }      
    }
  }
  
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
  else if (alloc_flag == NONE){
  }
  else{
    PLEGMA_error("Error not supported %d\n",alloc_flag);
  }
}

template<typename Float>
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, int site_size, size_t localVol, GHOST_FLAG ghost_flag, bool isPinnedHost, bool checkErr):
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag),isPinnedHost(isPinnedHost), isAllocHost(false), isAllocDevice(false), checkErr(checkErr), field_type(CUSTOM)
{
  initialize(alloc_flag, site_size, localVol);
  field_name = "PLEGMA_CUSTOM";
}

template<typename Float>
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag, bool isPinnedHost):
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag),isPinnedHost(isPinnedHost), isAllocHost(false), isAllocDevice(false), checkErr(true), field_type(classT)
{
  if(HGC_init_PLEGMA_flag == false) 
    PLEGMA_error("You must initialize init_PLEGMA first");
  
  switch(classT){
  case SCALAR:
    initialize(alloc_flag, 1, HGC_localVolume);
    field_name = "PLEGMA_SCALAR";
    setSiteShape({});
    break;
  case SU3FIELD:
    initialize(alloc_flag, N_COLS * N_COLS, HGC_localVolume);
    field_name = "PLEGMA_SU3FIELD";
    setSiteShape({N_COLS, N_COLS});
    break;
  case GAUGE:
    initialize(alloc_flag, N_DIMS * N_COLS * N_COLS, HGC_localVolume);
    field_name = "PLEGMA_GAUGE";
    setSiteShape({N_DIMS, N_COLS, N_COLS});
    break;    
  case GAUGE3D:
    initialize(alloc_flag, N_DIMS * N_COLS * N_COLS, HGC_localVolume3D);
    field_name = "PLEGMA_GAUGE3D";
    setSiteShape({N_DIMS, N_COLS, N_COLS});
    break;
  case VECTOR:
    initialize(alloc_flag, N_SPINS * N_COLS, HGC_localVolume);
    field_name = "PLEGMA_VECTOR";
    setSiteShape({N_SPINS, N_COLS});
    break;
  case VECTOR3D:
    initialize(alloc_flag, N_SPINS * N_COLS, HGC_localVolume3D);
    field_name = "PLEGMA_VECTOR3D";
    setSiteShape({N_SPINS, N_COLS});
    break;
  case PROPAGATOR:
    initialize(alloc_flag, N_SPINS * N_COLS * N_SPINS * N_COLS, HGC_localVolume);
    field_name = "PLEGMA_PROPAGATOR";
    setSiteShape({N_SPINS, N_SPINS, N_COLS, N_COLS});
    break;
  case PROPAGATOR3D:
    initialize(alloc_flag, N_SPINS * N_COLS * N_SPINS * N_COLS, HGC_localVolume3D);
    field_name = "PLEGMA_PROPAGATOR3D";
    setSiteShape({N_SPINS, N_SPINS, N_COLS, N_COLS});
    break;
  case QLOOPS:
    initialize(alloc_flag, N_SPINS * N_SPINS, HGC_localVolume);
    field_name = "PLEGMA_QLOOPS";
    setSiteShape({N_SPINS, N_SPINS});
    break;
  case FMUNU:
    initialize(alloc_flag, ((N_SPINS * (N_SPINS-1))/2) * N_COLS * N_COLS, HGC_localVolume);
    field_name = "PLEGMA_FMUNU";
    setSiteShape({((N_SPINS * (N_SPINS-1))/2), N_COLS, N_COLS});
    break;
  default:
    PLEGMA_error("Unknown field class %d\n", classT);
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
  cudaMemcpy(d_elem, h_elem, Bytes_total(), cudaMemcpyHostToDevice );
  if(checkErr) checkCudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::unload() const{
  if(allocation != BOTH) PLEGMA_error("Load from Host to Device needs BOTH allocation");
  cudaMemcpy(h_elem, d_elem, Bytes_total(), cudaMemcpyDeviceToHost);
  if(checkErr) checkCudaError();
}


template<typename Float>
void PLEGMA_Field<Float>::create_host(){
  if(!isPinnedHost)hostMalloc(h_elem, Bytes_total_plus_ghost());
  else hostMallocPinned(h_elem, Bytes_total_plus_ghost());
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void PLEGMA_Field<Float>::create_device(){
  cudaMalloc((void**)&d_elem,Bytes_total_plus_ghost());
  if(checkErr) checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  // device memory in MB
  HGC_deviceMemory += Bytes_total_plus_ghost()/(1024.*1024.);          
  if(HGC_verbosity>1) PLEGMA_printf("Device memory in use is %f MB A PLEGMA \n",HGC_deviceMemory);
#endif
  zero_device();
  if(ghost_flag >= FIRST_SIDE){
#ifdef HAVE_PINNED_GHOST
    cudaMallocHost((void**)&h_ext_ghost_r, Bytes_ghost());
    cudaMallocHost((void**)&h_ext_ghost_s, Bytes_ghost());
#else
    hostMalloc(h_ext_ghost_r, Bytes_ghost());
    hostMalloc(h_ext_ghost_s, Bytes_ghost());
#endif
  }
  if(ghost_flag == FIRST_CORNER){
#ifdef HAVE_PINNED_GHOST
    cudaMallocHost((void**)&h_ext_ghost_corner_r, Bytes_ghostCorner());
    cudaMallocHost((void**)&h_ext_ghost_corner_s, Bytes_ghostCorner());
#else    
    hostMalloc(h_ext_ghost_corner_r, Bytes_ghostCorner());
    hostMalloc(h_ext_ghost_corner_s, Bytes_ghostCorner());
#endif

  }
  if(checkErr) checkCudaError();
  isAllocDevice = true;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_host(){
  if(!isPinnedHost)hostFree(h_elem, Bytes_total_plus_ghost());
  else hostFreePinned(h_elem, Bytes_total_plus_ghost());
  isAllocHost=false;
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_device(){
  cudaFree(d_elem);
  if(checkErr) checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  HGC_deviceMemory -= Bytes_total_plus_ghost()/(1024.*1024.);
  if(HGC_verbosity>1) PLEGMA_printf("Device memory in use is %f MB D PLEGMA\n",HGC_deviceMemory);
#endif
  if(ghost_flag >= FIRST_SIDE){
#ifdef HAVE_PINNED_GHOST
    cudaFreeHost(h_ext_ghost_r); h_ext_ghost_r=NULL;
    cudaFreeHost(h_ext_ghost_s); h_ext_ghost_s=NULL;
#else
    hostFree(h_ext_ghost_r,Bytes_ghost()); h_ext_ghost_r=NULL;
    hostFree(h_ext_ghost_s,Bytes_ghost()); h_ext_ghost_s=NULL;
#endif
  }
  if(ghost_flag == FIRST_CORNER){
#ifdef HAVE_PINNED_GHOST
    cudaFreeHost(h_ext_ghost_corner_r); h_ext_ghost_corner_r=NULL;
    cudaFreeHost(h_ext_ghost_corner_s); h_ext_ghost_corner_s=NULL;
#else
    hostFree(h_ext_ghost_corner_r,Bytes_ghostCorner()); h_ext_ghost_corner_r=NULL;
    hostFree(h_ext_ghost_corner_s,Bytes_ghostCorner()); h_ext_ghost_corner_s=NULL;
#endif

  }
  if(checkErr) checkCudaError();
  isAllocDevice=false;
}

template<typename Float>
void PLEGMA_Field<Float>::zero_host(){
  if(isAllocHost)memset(h_elem,0,Bytes_total_plus_ghost());
}

template<typename Float>
void PLEGMA_Field<Float>::zero_device(){
  if(isAllocDevice)cudaMemset(d_elem,0,Bytes_total_plus_ghost());
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
  resDesc.res.linear.sizeInBytes = Bytes_total_plus_ghost();

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
  if(checkErr) checkCudaError();
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
      Bytes_total_plus_ghost()/(1024.*1024.));
  PLEGMA_printf("The flag for the host allocation is %d\n",(int) isAllocHost);
  PLEGMA_printf("The flag for the device allocation is %d\n",(int) isAllocDevice);
}

template<typename Float>
void PLEGMA_Field<Float>::communicateSideGhost(short dir, ORIENTATION sign, ACTION action){
  if(comm_size() == 1) return;
  assert(Total_length()==HGC_localVolume || Total_length()==HGC_localVolume3D);
  
  if(ghost_flag < FIRST_SIDE)
    PLEGMA_error("First side ghosts have not been allocated.\n");
  if(dir<-1 || dir>=N_DIMS)
    PLEGMA_error("Directions should be in [-1,%d] range with -1 all directions",N_DIMS);
  if(sign<0 || sign>DIR_BOTH)
    PLEGMA_error("Directions should be an orientation enum");

  bool isAll = (dir<0) ? true:false;
  bool runT = Total_length()==HGC_localVolume;

  if(action==START || action==DO_ALL)
    for(short i=0; i<N_DIMS; i++)
      if( (dir == i || isAll) && HGC_dimBreak[i] && (i < N_DIMS-1 || runT) )
	for(short s = 0; s < DIR_BOTH; s++)
	  if(sign == s || sign==DIR_BOTH){
	    Float *pointer_receive = h_ext_ghost_r+(HGC_sideGhost[i][s]-total_length)*field_length*2;
	    Float *pointer_send = h_ext_ghost_s+(HGC_sideGhost[i][s]-total_length)*field_length*2;
	    Float *pointer_device = d_elem+HGC_sideGhost[i][s]*field_length*2;
	    int disp;
	    size_t nbytes = HGC_surface3D[i]*field_length*2*sizeof(Float);
	    
	    // collecting elements from device
	    copy_side_to_ghost(*this, i, s);

	    // For Field3D we need to run only up to the previous kernel due to tuning.
	    // The rest is useless if not in the timeslice
	    if(not includesActiveTimeSlice()) continue;
	    
	    cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
	    if(checkErr) checkCudaError();
	      
	    disp = (s==DIR_PLUS) ? +1 : -1;
	    messages.push_back(comm_declare_receive_relative(pointer_receive,i,disp,nbytes));
	    comm_start(messages.back());
	    disp *= -1;
	    messages.push_back(comm_declare_send_relative(pointer_send,i,disp,nbytes));
	    comm_start(messages.back());
	  }
  if(not includesActiveTimeSlice()) return;
  if(action==FINISH || action==DO_ALL) {
    // waiting for communications
    while (! messages.empty()) {
      comm_wait(messages.back());
      comm_free(messages.back());
      messages.pop_back();
    }
    //copying to device
    if(isAll && sign==DIR_BOTH) {
      Float *host = h_ext_ghost_r;
      Float *device = d_elem+total_length*field_length*2;
      cudaMemcpy(device, host, Bytes_ghost(),cudaMemcpyHostToDevice);
      if(checkErr) checkCudaError();
    } else {
      for(short i=0; i<N_DIMS; i++)
	if( (dir == i || isAll) && HGC_dimBreak[i] && (i < N_DIMS-1 || runT) )
	  for(short s = 0; s < DIR_BOTH; s++)
	    if(sign == s || sign==DIR_BOTH){
	      Float *host = h_ext_ghost_r + (HGC_sideGhost[dir][s]-total_length)*field_length*2;
	      Float *device = d_elem + HGC_sideGhost[dir][s]*field_length*2;
	      cudaMemcpy(device, host, HGC_surface3D[dir]*field_length*2*sizeof(Float),
			 cudaMemcpyHostToDevice);
	      if(checkErr) checkCudaError();
	    }
    }
  }
}


template<typename Float>
void PLEGMA_Field<Float>::communicateCornerGhost(short dir, ORIENTATION sign, ACTION action){if(comm_size() == 1) return;
  if(comm_size() == 1) return;
  assert(Total_length()==HGC_localVolume || Total_length()==HGC_localVolume3D);
  
  if(ghost_flag < FIRST_CORNER)
    PLEGMA_error("First corner ghosts have not been allocated.\n");
  if(dir<-1 || dir>=N_DIMS)
    PLEGMA_error("Directions should be in [-1,%d] range with -1 all directions",N_DIMS);

  bool isAll = (dir<0) ? true:false;
  bool runT = Total_length()==HGC_localVolume;

  std::vector<MsgHandle*> messages;

  if(action==START || action==DO_ALL)
    for(short i=0; i<N_DIMS; i++)
      for(short j=i+1; j<N_DIMS; j++)
	if( i != j && HGC_dimBreak[i] && HGC_dimBreak[j] && (dir == i || dir == j || isAll) && (j < N_DIMS-1 || runT))
	  if(dir == i || dir == j || isAll)
	    for(short s1 = 0; s1 < DIR_BOTH; s1++)
	      for(short s2 = 0; s2 < DIR_BOTH; s2++)
		if(sign == s1 || sign == s2 || sign==DIR_BOTH) {
		  Float *pointer_receive = h_ext_ghost_corner_r + (HGC_cornerGhost[i][j][s1][s2]-total_length-ghost_length)*field_length*2;
		  Float *pointer_send = h_ext_ghost_corner_s + (HGC_cornerGhost[i][j][s1][s2]-total_length-ghost_length)*field_length*2;
		  Float *pointer_device = d_elem + HGC_cornerGhost[i][j][s1][s2]*field_length*2;
		  int disp[N_DIMS] = {0};
		  size_t nbytes = HGC_surface2D[i][j]*field_length*2*sizeof(Float);

		  // collecting elements from device
		  copy_corner_to_ghost(*this, i, j, s1, s2);
		  // For Field3D we need to run only up to the previous kernel due to tuning.
		  // The rest is useless if not in the timeslice
		  if(not includesActiveTimeSlice()) continue;
		  
		  cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
		  if(checkErr) checkCudaError();
	    
		  // communicating
		  disp[i] = (s1==DIR_PLUS) ? +1 : -1;
		  disp[j] = (s2==DIR_PLUS) ? +1 : -1;
		  messages.push_back(comm_declare_receive_displaced(pointer_receive,disp,nbytes)); 
		  comm_start(messages.back());
		  disp[i] *= -1;
		  disp[j] *= -1;
		  messages.push_back(comm_declare_send_displaced(pointer_send,disp,nbytes));
		  disp[i] = 0; disp[j] = 0;	  
		  comm_start(messages.back());
		}
  if(not includesActiveTimeSlice()) return;
  if(action==FINISH || action==DO_ALL) {
    // waiting for communications
    while (! messages.empty()) {
      comm_wait(messages.back());
      comm_free(messages.back());
      messages.pop_back();
    }
    //copying to device
    if(isAll && sign==DIR_BOTH) {
      Float *hostCorner = h_ext_ghost_corner_r;
      Float *device = d_elem+(total_length+ghost_length)*field_length*2;
      cudaMemcpy(device,hostCorner,Bytes_ghostCorner(),cudaMemcpyHostToDevice);
      if(checkErr) checkCudaError();
    } else {
      for(short i=0; i<N_DIMS; i++)
	for(short j=i+1; j<N_DIMS; j++)
	  if( i != j && HGC_dimBreak[i] && HGC_dimBreak[j] && (dir == i || dir == j || isAll) && (j < N_DIMS-1 || runT))
	    if(dir == i || dir == j || isAll)
	      for(short s1 = 0; s1 < DIR_BOTH; s1++)
		for(short s2 = 0; s2 < DIR_BOTH; s2++)
		  if(sign == s1 || sign == s2 || sign==DIR_BOTH) {
		    Float *hostCorner = h_ext_ghost_corner_r + (HGC_cornerGhost[i][j][s1][s2]-total_length-ghost_length)*field_length*2;
		    Float *device = d_elem+HGC_cornerGhost[i][j][s1][s2]*field_length*2;
		    cudaMemcpy(device, hostCorner, HGC_surface2D[i][j]*field_length*2*sizeof(Float),
			       cudaMemcpyHostToDevice);
		    if(checkErr) checkCudaError();
		  }
    }
  }
}

template<typename Float>
void PLEGMA_Field<Float>::communicateGhost(short dir, ORIENTATION sign, GHOST_FLAG which_ghost, ACTION action){
  if(which_ghost == ALL_GHOSTS) which_ghost = ghost_flag;
  if(ghost_flag < which_ghost) {
    PLEGMA_error("Asking to communicate ghost but they have not been allocated.\n");
  }
  if(which_ghost >= FIRST_SIDE){
    communicateSideGhost(dir, sign, action);
  }
  if(which_ghost >= FIRST_CORNER){
    communicateCornerGhost(dir, sign, action);
  }
}

template<typename Float>
void PLEGMA_Field<Float>::shift(PLEGMA_Field<Float> &Fin, short dirOr){
  // we have to make sure that we have the ghost
  Fin.communicateSideGhost(dirOr%N_DIMS, dirOr<N_DIMS ? DIR_MINUS : DIR_PLUS);
  shiftField(Fin,*this,dirOr);
}

template<typename Float>
void PLEGMA_Field<Float>::randInit(int seed){

  randstate_ptr = new PLEGMA_RNG(seed, total_length);
  if(checkErr) checkCudaError();  
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
  if(checkErr) checkCudaError();
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
  if(checkErr) checkCudaError();
  std::vector<Float> momF(mom.begin(), mom.end());
  createMomField(x, momF, D3D4, sign);
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
  return cuBLAS::dot(total_length*field_length, d_elem, fieldIn.D_elem(), HGC_fullComm);
}

template<typename Float>
Float PLEGMA_Field<Float>::norm(){
  return cuBLAS::norm(total_length*field_length, d_elem, HGC_fullComm);
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

template<typename Float>
void PLEGMA_Field<Float>::writeLIME(std::string filename) const{
  if(total_length != HGC_localVolume) PLEGMA_error("Writing of 3D fields is not supported");
  FILE *fid;
  LimeWriter *limewriter = (LimeWriter*)NULL;
  unload();
  if(comm_rank() == 0){
    fid=fopen(filename.c_str(),"w");
    if(fid==NULL) PLEGMA_error("Error opening file for writing: %s\n", filename.c_str());
    limewriter = limeCreateWriter(fid);
    if(limewriter==(LimeWriter*)NULL) PLEGMA_error("Could not create limeWriter");
    std::string xlf_message = getDateAndTime(); // More xlf-info can be added
    write_lime_header(limewriter,"xlf-info",xlf_message,1,1);
    std::ostringstream oss;
    oss << lime_version_header() << "<field>" << Field_name() << "</field>\n" << "<precision>" <<  Precision()*8 << "</precision>\n";
    oss << "<dof>" << field_length << "</dof>\n";
    std::vector<std::string> xyzt = {"x","y","z","t"};
    for(int i = 0 ; i < N_DIMS; i++) oss << "<l" << xyzt[i] << ">" << HGC_totalL[i] << "</l" << xyzt[i] << ">\n";
    oss << "</ildgFormat>";
    write_lime_header(limewriter,"ildg-format",oss.str(),1,0);
  }
  write_binary_to_lime(filename,fid,limewriter,h_elem,field_length);
  limeDestroyWriter(limewriter);
}

template<typename Float>
void PLEGMA_Field<Float>::readLIME(std::string filename){
  int precRead=0, dofRead=0;

  FILE *fid = NULL;
  LimeReader *limereader = NULL;
  if(comm_rank() == 0){
    fid=fopen(filename.c_str(),"r");
    if(fid==NULL) PLEGMA_error("Error opening file for reading: %s\n", filename.c_str());
    if ((limereader = limeCreateReader(fid))==NULL) PLEGMA_error("Could not create limeReader");
    read_lime_header(limereader,precRead,dofRead);
  }
  comm_broadcast(&precRead,sizeof(int));
  comm_broadcast(&dofRead,sizeof(int));
  if(precRead != Precision()) PLEGMA_error("PLEGMA field precision %d != %d precision read from LIME",Precision(),precRead);
  if(!isAllocHost) PLEGMA_error("Host memory should be allocated to read data from lime");
  if(dofRead > 0 && dofRead != field_length) PLEGMA_error("PLEGMA field dof %d != %d dof read from LIME", field_length, dofRead);
  read_binary_from_lime(filename,fid,limereader,h_elem,field_length);
  if(comm_rank() == 0){
    limeDestroyReader(limereader);
    fclose(fid);
  }
  if(isAllocDevice) load();
}

template<typename Float>
std::string PLEGMA_Field<Float>::
fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start) const{
  std::string descr = "shape: ";

  // Field shape
  if(!this->site_shape.empty()) {
    descr += "/" + field_name;
    for(auto s : this->site_shape) {
      shape.push_back(s);
      lshape.push_back(s);
      start.push_back(0);    
    }
  }
  
  descr += "/t/z/y/x";
  // Volume
  for(int i=N_DIMS-1; i>=0; i--) {
    shape.push_back(HGC_totalL[i]);
    lshape.push_back(HGC_localL[i]);
    start.push_back(HGC_procPosition[i]*HGC_localL[i]);
  }

  //re-im
  descr += "/re-im";
  shape.push_back(2);
  lshape.push_back(2);
  start.push_back(0);

  return descr;
}


template<typename Float>
void PLEGMA_Field<Float>::writeHDF5(std::string filename) const{
  if(total_length != HGC_localVolume) PLEGMA_error("Writing of 3D fields is not supported");
  assert(isAllocHost);
  unload();
  std::vector<hsize_t> shape, lshape, start;
  std::string descr = fill_H5_shapes(shape, lshape, start);

  std::string dataset = "data"; // default name
  
  // checking if dataset name provided in filename
  // NOTE: use '/' at the end of filename to use default name
  size_t ext = filename.rfind(".h5");
  if(ext + 3 < filename.length() && filename[ext+3] == '/') {
    size_t last = filename.rfind("/");
    if(last + 1 < filename.length()) {
      dataset = filename.substr(last+1);
      filename = filename.substr(0, last+1);
    }
  }

  HDF5 writer(filename, HGC_fullComm);

  writer.write_dataset(dataset, h_elem, shape, lshape, start);
  writer.write_attribute(dataset, "description", descr);
}


template<typename Float>
void PLEGMA_Field<Float>::TrFmunuSu3FmunuSu3(PLEGMA_Fmunu<Float> &Fl, std::pair<int,int> munu_l, PLEGMA_Su3field<Float> &Wl,
					PLEGMA_Fmunu<Float> &Fr, std::pair<int,int> munu_r,
					PLEGMA_Su3field<Float> &Wr){
  traceMulFmunuSu3FmunuSu3_k(*this,Fl,munu_l,Wl,Fr,munu_r,Wr);
}

template<typename Float>
void PLEGMA_Field<Float>::trPmunu(PLEGMA_Gauge<Float> &gauge, std::pair<int,int> munu){
  gauge.communicateSideGhost();
  trPmunu_k(*this,gauge,munu);
}

template class PLEGMA_Field<float>;
template class PLEGMA_Field<double>;
// Forcing initialization of the following cases
template void PLEGMA_Field<float>::copy<float>(PLEGMA_Field<float> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<float>::copy<double>(PLEGMA_Field<double> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<double>::copy<float>(PLEGMA_Field<float> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<double>::copy<double>(PLEGMA_Field<double> &f, ALLOCATION_FLAG where);

// field3D <- field4D
template<typename Float>
void PLEGMA_Field3D<Float>::absorb(const PLEGMA_Field<Float> &field, int global_it){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  assert(field.Field_length() == this->Field_length());
  int my_it = global_it - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  this->activeTimeSlice = (my_it >= 0) && ( my_it < HGC_localL[3] );
  size_t V3 = HGC_localVolume3D*2;
  size_t V4 = HGC_localVolume*2;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int i = 0; i < this->Field_length(); i++) {
    pointer_dst = (this->D_elem() + i*V3);
    if(this->activeTimeSlice) {
      pointer_src = (field.D_elem() + i*V4 + my_it*V3);
      cudaMemcpy(pointer_dst, pointer_src, V3 * sizeof(Float), cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(pointer_dst, 0, V3 * sizeof(Float));
    }
  }
  checkCudaError();
}
