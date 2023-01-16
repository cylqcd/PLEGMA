#include <PLEGMA_Field.h> 
#include <PLEGMA_Thrust.h>
#include <PLEGMA_field_utils.cuh>
#include <PLEGMA_shifts.cuh>
#include <PLEGMA_Random.h>
#include <vector>
#include <algorithm>
#include <time.h>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_FT.cuh>
#include <utils/PLEGMA_auxiliary.h>
#include <io/PLEGMA_lime.h>
#include <comm_quda.h>
#include <communicator_quda.h>
#include <malloc_quda.h>
using namespace plegma;

#define DEVICE_MEMORY_REPORT
#define CMPLX_FLOAT std::complex<Float>

//--------------------------//
// class PLEGMA_Field //
//--------------------------//

// This is is a class which allocates memory on either
// the device, or host, or both for the structures:
// Field: one complex number per spacetime point.
// Gauge: one SU(3) link variable per spacetime point X spacetime dimension.
// Vector: 12 complex numbers (3colour x 4spin) per spacetime point.
// Vector3D: as above, but only defined on a single timeslice.
// Propagator: 12x12 complex matrix per sink position 
// (usually a whole spacetime volume.)
// Propagtor3D: as above, but with sinks only at one timeslice.

//: computes the size of fields and alloc mem on both sides
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
  ghost_vertex_length = 0;
  
  if(ghost_flag>NO_GHOSTS) {
    assert(vol_l==HGC_localVolume || vol_l==HGC_localVolume3D);

    if(vol_l==HGC_localVolume) {
      for(int i = 0 ; i < N_DIMS ; i++){
	if(ghost_flag >= FIRST_SIDE) ghost_length += 2*HGC_surface3D[i];
	for(int j = i+1; j < N_DIMS; j++){
	  if(ghost_flag >= FIRST_CORNER) ghost_corner_length += 4*HGC_surface2D[OFF2(i,j)];
	  for(int k = j+1; k < N_DIMS; k++){
	    if(ghost_flag >= FIRST_VERTEX) ghost_vertex_length += 8*HGC_surface1D[OFF3(i,j,k)];
	  }	
	}
      }
      if(ghost_flag >= FIRST_SIDE) assert(ghost_length == HGC_sideGhostVolume);
      if(ghost_flag >= FIRST_CORNER) assert(ghost_corner_length == HGC_cornerGhostVolume);
      if(ghost_flag >= FIRST_VERTEX) assert(ghost_vertex_length == HGC_vertexGhostVolume);
    } else if(vol_l==HGC_localVolume3D) {
      for(int i = 0 ; i < N_DIMS-1 ; i++){
	if(ghost_flag >= FIRST_SIDE) ghost_length += 2*HGC_surface3D[i]/HGC_localL[DIM_T];
	for(int j = i+1; j < N_DIMS-1; j++){
	  if(ghost_flag >= FIRST_CORNER) ghost_corner_length += 4*HGC_surface2D[OFF2(i,j)]/HGC_localL[DIM_T];
	  for(int k = j+1; k < N_DIMS-1; k++){
	    if(ghost_flag >= FIRST_VERTEX) ghost_vertex_length += 8*HGC_surface1D[OFF3(i,j,k)]/HGC_localL[DIM_T];
	  }	
	}
      }
      if(ghost_flag >= FIRST_SIDE) assert(ghost_length == HGC_sideGhostVolume3D);
      if(ghost_flag >= FIRST_CORNER) assert(ghost_corner_length == HGC_cornerGhostVolume3D);
      if(ghost_flag >= FIRST_VERTEX) assert(ghost_vertex_length == HGC_vertexGhostVolume3D);
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
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), h_ext_ghost_vertex_r(NULL), h_ext_ghost_vertex_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag),isPinnedHost(isPinnedHost), isAllocHost(false), isAllocDevice(false), checkErr(checkErr), field_type(CUSTOM)
{
  initialize(alloc_flag, site_size, localVol);
  field_name = "PLEGMA_CUSTOM";
}

template<typename Float>
PLEGMA_Field<Float>::PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag, bool isPinnedHost, bool checkErr):
  h_elem(NULL), d_elem(NULL), h_ext_ghost_r(NULL), h_ext_ghost_s(NULL), h_ext_ghost_corner_r(NULL), h_ext_ghost_corner_s(NULL), h_ext_ghost_vertex_r(NULL), h_ext_ghost_vertex_s(NULL), randstate_ptr(NULL), 
  ghost_flag(ghost_flag), allocation(alloc_flag),isPinnedHost(isPinnedHost), isAllocHost(false), isAllocDevice(false), checkErr(checkErr), field_type(classT)
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
  case U1GAUGE:
    initialize(alloc_flag, N_DIMS, HGC_localVolume);
    field_name = "PLEGMA_U1GAUGE";
    setSiteShape({N_DIMS});
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
  destroy_randstate();
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
  if(checkErr) checkQudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::unload() const{
  if(allocation != BOTH) PLEGMA_error("Unload from Device to Host needs BOTH allocation");
  cudaMemcpy(h_elem, d_elem, Bytes_total(), cudaMemcpyDeviceToHost);
  if(checkErr) checkQudaError();
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
  d_elem=(Float *)device_malloc(Bytes_total_plus_ghost());
  //cudaMalloc((void**)&d_elem,Bytes_total_plus_ghost());
  if(checkErr) checkQudaError();
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
  if(ghost_flag >= FIRST_CORNER){
#ifdef HAVE_PINNED_GHOST
    cudaMallocHost((void**)&h_ext_ghost_corner_r, Bytes_ghostCorner());
    cudaMallocHost((void**)&h_ext_ghost_corner_s, Bytes_ghostCorner());
#else    
    hostMalloc(h_ext_ghost_corner_r, Bytes_ghostCorner());
    hostMalloc(h_ext_ghost_corner_s, Bytes_ghostCorner());
#endif
  }
  if(ghost_flag >= FIRST_VERTEX){
#ifdef HAVE_PINNED_GHOST
    cudaMallocHost((void**)&h_ext_ghost_vertex_r, Bytes_ghostVertex());
    cudaMallocHost((void**)&h_ext_ghost_vertex_s, Bytes_ghostVertex());
#else    
    hostMalloc(h_ext_ghost_vertex_r, Bytes_ghostVertex());
    hostMalloc(h_ext_ghost_vertex_s, Bytes_ghostVertex());
#endif
  }
  if(checkErr) checkQudaError();
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
  if(checkErr) checkQudaError();
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
  if(ghost_flag >= FIRST_CORNER){
#ifdef HAVE_PINNED_GHOST
    cudaFreeHost(h_ext_ghost_corner_r); h_ext_ghost_corner_r=NULL;
    cudaFreeHost(h_ext_ghost_corner_s); h_ext_ghost_corner_s=NULL;
#else
    hostFree(h_ext_ghost_corner_r,Bytes_ghostCorner()); h_ext_ghost_corner_r=NULL;
    hostFree(h_ext_ghost_corner_s,Bytes_ghostCorner()); h_ext_ghost_corner_s=NULL;
#endif
  }
  if(ghost_flag >= FIRST_VERTEX){
#ifdef HAVE_PINNED_GHOST
    cudaFreeHost(h_ext_ghost_vertex_r); h_ext_ghost_vertex_r=NULL;
    cudaFreeHost(h_ext_ghost_vertex_s); h_ext_ghost_vertex_s=NULL;
#else
    hostFree(h_ext_ghost_vertex_r,Bytes_ghostVertex()); h_ext_ghost_vertex_r=NULL;
    hostFree(h_ext_ghost_vertex_s,Bytes_ghostVertex()); h_ext_ghost_vertex_s=NULL;
#endif
  }
  if(checkErr) checkQudaError();
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
  else if(alloc_flag == NONE){
    
  } 
  else{
    PLEGMA_error("Not supported %d\n",alloc_flag);
  }
}

template<typename Float>
cudaTextureObject_t PLEGMA_Field<Float>::createTexObject() const{
#ifdef PLEGMA_TEXTURE
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
  return tex;
#else
  return 0;
#endif
}

template<typename Float>
void PLEGMA_Field<Float>::destroyTexObject(cudaTextureObject_t tex) const{
#ifdef PLEGMA_TEXTURE
  cudaDestroyTextureObject(tex);
#endif
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
  size_t scaleT = runT ? 1 : HGC_localL[DIM_T];

  if(action==START || action==DO_ALL)
    for(short i=0; i<N_DIMS; i++)
      if( (dir == i || isAll) && HGC_dimBreak[i] && (i < N_DIMS-1 || runT) )
	for(short s = 0; s < DIR_BOTH; s++)
	  if(sign == s || sign==DIR_BOTH){
	    // collecting elements from device
	    copy_side_to_ghost(toField2<pFloat2>(*this), i, s);

	    Float *pointer_receive = h_ext_ghost_r+HGC_sideGhost[i][s]/scaleT*field_length*2;
	    Float *pointer_send = h_ext_ghost_s+HGC_sideGhost[i][s]/scaleT*field_length*2;
	    Float *pointer_device = d_elem+(HGC_sideGhost[i][s]/scaleT+total_length)*field_length*2;
	    int disp;
	    size_t nbytes = HGC_surface3D[i]/scaleT*field_length*2*sizeof(Float);
	    
	    cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
	    if(checkErr) checkQudaError();
	      
	    disp = (s==DIR_PLUS) ? +1 : -1;
	    messages.push_back(comm_declare_receive_relative(pointer_receive,i,disp,nbytes));
	    comm_start(messages.back());
	    disp *= -1;
	    messages.push_back(comm_declare_send_relative(pointer_send,i,disp,nbytes));
	    comm_start(messages.back());
	  }
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
      if(checkErr) checkQudaError();
    } else {
      for(short i=0; i<N_DIMS; i++)
	if( (dir == i || isAll) && HGC_dimBreak[i] && (i < N_DIMS-1 || runT) )
	  for(short s = 0; s < DIR_BOTH; s++)
	    if(sign == s || sign==DIR_BOTH){
	      Float *host = h_ext_ghost_r + HGC_sideGhost[dir][s]/scaleT*field_length*2;
	      Float *device = d_elem + (HGC_sideGhost[dir][s]/scaleT+total_length)*field_length*2;
	      cudaMemcpy(device, host, HGC_surface3D[dir]/scaleT*field_length*2*sizeof(Float),
			 cudaMemcpyHostToDevice);
	      if(checkErr) checkQudaError();
	    }
    }
  }
}

template<typename Float>
void PLEGMA_Field<Float>::communicateCornerGhost(short dir, ORIENTATION sign, ACTION action){
  if(comm_size() == 1) return;
  assert(Total_length()==HGC_localVolume || Total_length()==HGC_localVolume3D);
  
  if(ghost_flag < FIRST_CORNER)
    PLEGMA_error("First corner ghosts have not been allocated.\n");
  if(dir<-1 || dir>=N_DIMS)
    PLEGMA_error("Directions should be in [-1,%d] range with -1 all directions",N_DIMS);
  if(sign<0 || sign>DIR_BOTH)
    PLEGMA_error("Directions should be an orientation enum");

  bool isAll = (dir<0) ? true:false;
  bool runT = Total_length()==HGC_localVolume;
  size_t scaleT = runT ? 1 : HGC_localL[DIM_T];

  std::vector<MsgHandle*> messages;

  if(action==START || action==DO_ALL)
    for(short i=0; i<N_DIMS; i++)
      for(short j=i+1; j<N_DIMS; j++)
	if( HGC_dimBreak[i] && HGC_dimBreak[j] && (dir == i || dir == j || isAll) && (j < N_DIMS-1 || runT))
	    for(short s1 = 0; s1 < DIR_BOTH; s1++)
	      for(short s2 = 0; s2 < DIR_BOTH; s2++)
		if(sign == s1 || sign == s2 || sign==DIR_BOTH) {
		  // collecting elements from device
		  copy_corner_to_ghost(toField2<pFloat2>(*this), i, j, s1, s2);
		  
		  Float *pointer_receive = h_ext_ghost_corner_r + HGC_cornerGhost[OFF2SIGN(i,j,s1,s2)]/scaleT*field_length*2;
		  Float *pointer_send = h_ext_ghost_corner_s + HGC_cornerGhost[OFF2SIGN(i,j,s1,s2)]/scaleT*field_length*2;
		  Float *pointer_device = d_elem + (HGC_cornerGhost[OFF2SIGN(i,j,s1,s2)]/scaleT+total_length+ghost_length)*field_length*2;
		  int disp[N_DIMS] = {0};
		  size_t nbytes = HGC_surface2D[OFF2(i,j)]/scaleT*field_length*2*sizeof(Float);

		  cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
		  if(checkErr) checkQudaError();
	    
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
      if(checkErr) checkQudaError();
    } else {
      for(short i=0; i<N_DIMS; i++)
	for(short j=i+1; j<N_DIMS; j++)
	  if( HGC_dimBreak[i] && HGC_dimBreak[j] && (dir == i || dir == j || isAll) && (j < N_DIMS-1 || runT))
	      for(short s1 = 0; s1 < DIR_BOTH; s1++)
		for(short s2 = 0; s2 < DIR_BOTH; s2++)
		  if(sign == s1 || sign == s2 || sign==DIR_BOTH) {
		    Float *hostCorner = h_ext_ghost_corner_r + HGC_cornerGhost[OFF2SIGN(i,j,s1,s2)]/scaleT*field_length*2;
		    Float *device = d_elem+(HGC_cornerGhost[OFF2SIGN(i,j,s1,s2)]/scaleT+total_length+ghost_length)*field_length*2;
		    cudaMemcpy(device, hostCorner, HGC_surface2D[OFF2(i,j)]/scaleT*field_length*2*sizeof(Float),
			       cudaMemcpyHostToDevice);
		    if(checkErr) checkQudaError();
		  }
    }
  }
}

template<typename Float>
void PLEGMA_Field<Float>::communicateVertexGhost(short dir, ORIENTATION sign, ACTION action){
  if(comm_size() == 1) return;
  assert(Total_length()==HGC_localVolume || Total_length()==HGC_localVolume3D);
  
  if(ghost_flag < FIRST_VERTEX)
    PLEGMA_error("First vertex ghosts have not been allocated.\n");
  if(dir<-1 || dir>=N_DIMS)
    PLEGMA_error("Directions should be in [-1,%d] range with -1 all directions",N_DIMS);

  bool isAll = (dir<0) ? true:false;
  bool runT = Total_length()==HGC_localVolume;
  size_t scaleT = runT ? 1 : HGC_localL[DIM_T];

  std::vector<MsgHandle*> messages;

  if(action==START || action==DO_ALL)
    for(short i=0; i<N_DIMS; i++)
      for(short j=i+1; j<N_DIMS; j++)
	for(short k=j+1; k<N_DIMS; k++)
	  if( HGC_dimBreak[i] && HGC_dimBreak[j] && HGC_dimBreak[k] && (dir == i || dir == j || dir == k || isAll) && (k < N_DIMS-1 || runT))
	    for(short s1 = 0; s1 < DIR_BOTH; s1++)
	      for(short s2 = 0; s2 < DIR_BOTH; s2++)
		for(short s3 = 0; s3 < DIR_BOTH; s3++)
		  if(sign == s1 || sign == s2  || sign == s3 || sign==DIR_BOTH) {
		    // collecting elements from device
		    copy_vertex_to_ghost(toField2<pFloat2>(*this), i, j, k, s1, s2, s3);
		  
		    Float *pointer_receive = h_ext_ghost_vertex_r + HGC_vertexGhost[OFF3SIGN(i,j,k,s1,s2,s3)]/scaleT*field_length*2;
		    Float *pointer_send = h_ext_ghost_vertex_s + HGC_vertexGhost[OFF3SIGN(i,j,k,s1,s2,s3)]/scaleT*field_length*2;
		    Float *pointer_device = d_elem + (HGC_vertexGhost[OFF3SIGN(i,j,k,s1,s2,s3)]/scaleT+total_length+ghost_length+ghost_corner_length)*field_length*2;
		    int disp[N_DIMS] = {0};
		    size_t nbytes = HGC_surface1D[OFF3(i,j,k)]/scaleT*field_length*2*sizeof(Float);

		    cudaMemcpy(pointer_send, pointer_device, nbytes, cudaMemcpyDeviceToHost);
		    if(checkErr) checkQudaError();
	    
		    // communicating
		    disp[i] = (s1==DIR_PLUS) ? +1 : -1;
		    disp[j] = (s2==DIR_PLUS) ? +1 : -1;
		    disp[k] = (s3==DIR_PLUS) ? +1 : -1;	
		    messages.push_back(comm_declare_receive_displaced(pointer_receive,disp,nbytes)); 
		    comm_start(messages.back());
		    disp[i] *= -1;
		    disp[j] *= -1;
		    disp[k] *= -1;
		    messages.push_back(comm_declare_send_displaced(pointer_send,disp,nbytes));
		    disp[i] = 0; disp[j] = 0; disp[k] = 0;  
		    comm_start(messages.back());
		  }
  if(action==FINISH || action==DO_ALL) {
    // waiting for communications
    while (! messages.empty()) {
      comm_wait(messages.back());
      comm_free(messages.back());
      messages.pop_back();
    }
    //copying to device
    if(isAll && sign==DIR_BOTH) {
      Float *hostVertex = h_ext_ghost_vertex_r;
      Float *device = d_elem+(total_length+ghost_length)*field_length*2;
      cudaMemcpy(device,hostVertex,Bytes_ghostVertex(),cudaMemcpyHostToDevice);
      if(checkErr) checkQudaError();
    } else {
      for(short i=0; i<N_DIMS; i++)
	for(short j=i+1; j<N_DIMS; j++)
	  for(short k=j+1; k<N_DIMS; k++)
	    if( HGC_dimBreak[i] && HGC_dimBreak[j] && HGC_dimBreak[k] && (dir == i || dir == j || dir == k || isAll) && (k < N_DIMS-1 || runT))
	      for(short s1 = 0; s1 < DIR_BOTH; s1++)
		for(short s2 = 0; s2 < DIR_BOTH; s2++)
		  for(short s3 = 0; s3 < DIR_BOTH; s3++)
		    if(sign == s1 || sign == s2 || sign == s3 || sign==DIR_BOTH) {
		      Float *hostVertex = h_ext_ghost_vertex_r + HGC_vertexGhost[OFF3SIGN(i,j,k,s1,s2,s3)]/scaleT*field_length*2;
		      Float *device = d_elem+(HGC_vertexGhost[OFF3SIGN(i,j,k,s1,s2,s3)]/scaleT+total_length+ghost_length+ghost_corner_length)*field_length*2;
		      cudaMemcpy(device, hostVertex, HGC_surface1D[OFF3(i,j,k)]/scaleT*field_length*2*sizeof(Float),
				 cudaMemcpyHostToDevice);
		      if(checkErr) checkQudaError();
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
  if(which_ghost >= FIRST_VERTEX){
    communicateVertexGhost(dir, sign, action);
  }
}

template<typename Float>
void PLEGMA_Field<Float>::conjugate(){
  // we have to make sure that we have the ghost
  conjugate_k(*this);
}


template<typename Float>
void PLEGMA_Field<Float>::shift(PLEGMA_Field<Float> &Fin, short dirOr){
  // we have to make sure that we have the ghost
  Fin.communicateSideGhost(dirOr%N_DIMS, dirOr<N_DIMS ? DIR_MINUS : DIR_PLUS);
  shiftField(Fin,*this,dirOr);
}

template<typename Float>
void PLEGMA_Field<Float>::shift(PLEGMA_Field<Float> &Fin, short dirOr1, short dirOr2){
  // we have to make sure that we have the ghost
  assert(dirOr1!=dirOr2);
  Fin.communicateGhost(-1, DIR_BOTH, FIRST_CORNER);
  shiftField(Fin,*this,dirOr1,dirOr2);
}

template<typename Float>
void PLEGMA_Field<Float>::shift(PLEGMA_Field<Float> &Fin, short dirOr1, short dirOr2, short dirOr3){
  // we have to make sure that we have the ghost
  assert(dirOr1!=dirOr2 && dirOr2!=dirOr3 && dirOr1!=dirOr3);
  Fin.communicateGhost(-1, DIR_BOTH, FIRST_VERTEX);
  shiftField(Fin,*this,dirOr1,dirOr2,dirOr3);
}

template<typename Float>
void PLEGMA_Field<Float>::randInit(int seed){
  randstate_ptr = new RNG(this, seed, total_length);
  if(checkErr) checkQudaError();  
}

template<typename Float>
void PLEGMA_Field<Float>::destroy_randstate(){
  if(randstate_ptr != NULL) delete randstate_ptr;
}

template<typename Float>
void PLEGMA_Field<Float>::stochastic_Z(int n){
  this->zero_device();
  if(randstate_ptr == NULL) PLEGMA_error("Random number generator state not initialized");
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
  if(checkErr) checkQudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::random(DIST sampling){
  this->zero_device();
  int rng_size = this->total_length;
  if(randstate_ptr==NULL)
    randInit(time(NULL));
  set_random<Float>( *randstate_ptr, *this, this->field_length, rng_size, sampling);    
}


template<typename Float>
void PLEGMA_Field<Float>::setUnit(std::vector<int> indDiag){
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
template<typename FloatMom>
void PLEGMA_Field<Float>::mulMomentumPhases(std::vector<FloatMom> mom, int sign){
  if(!isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
  if(sign != +1 && sign != -1) PLEGMA_error("Sign should be either +1 or -1\n");
  if(mom.size() != 3 && mom.size() != 4) PLEGMA_error("Momentum size vector should be either 3 or 4\n");
  if(total_length == HGC_localVolume && mom.size() != 4 ) PLEGMA_error("A 4D field needs a 4D momentum vector\n");
  if( (total_length == HGC_localVolume3D) && mom.size() != 3 ) PLEGMA_error("A 3D field needs a 3D momentum vector\n");
  int D3D4 = mom.size();
  int V = D3D4 == 3 ? HGC_localVolume3D : HGC_localVolume;
  Float2<Float> *x;
  x=(Float2<Float> *)device_malloc(V*2*sizeof(Float));
  //cudaMalloc((void**)&x, V*2*sizeof(Float));
  cudaMemset((void*) x,0,V*2*sizeof(Float));
  if(checkErr) checkQudaError();
  std::vector<Float> momF(mom.begin(), mom.end());
  createMomField(x, momF, D3D4, sign);
  for(int dof = 0; dof < field_length; dof++)
    plegma::elemWiseMul(V,(Float*) x, d_elem + dof*total_length*2);
  cudaFree(x);
}

template<typename Float>
void PLEGMA_Field<Float>::mulThetaPhase(Float theta, bool dagger){
  int sign = dagger?+1:-1;
  std::vector<Float> mom = {0.,0.,0.,static_cast<Float>(theta*0.5)};
  mulMomentumPhases(mom,sign);
}

// y=a*x+y
template<typename Float>
void PLEGMA_Field<Float>::add(PLEGMA_Field<Float> &fieldIn, std::complex<Float> alpha){
  if(field_length != fieldIn.Field_length()) PLEGMA_error("The d.o.f of the fields do not match\n");
  if(total_length != fieldIn.Total_length()) PLEGMA_error("The lattice points of the fields do not match\n");
  Float a[2]; a[0]=alpha.real(); a[1]=alpha.imag();
  cuBLAS::axpy(total_length*field_length, a, fieldIn.D_elem(), d_elem);
}


template<typename Float>
std::complex<Float> PLEGMA_Field<Float>::dot(PLEGMA_Field<Float> &fieldIn){
  if(field_length != fieldIn.Field_length()) PLEGMA_error("The d.o.f of the fields do not match\n");
  if(total_length != fieldIn.Total_length()) PLEGMA_error("The lattice points of the fields do not match\n");
  return cuBLAS::dot(total_length*field_length, d_elem, fieldIn.D_elem(), HGC_fullComm);
}

template<typename Float>
Float PLEGMA_Field<Float>::norm(){
  return cuBLAS::norm(total_length*field_length, d_elem, HGC_fullComm);
}

template<typename Float>
void PLEGMA_Field<Float>::scale(Float val){
  if(!isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
  cuBLAS::scal(field_length*total_length, val, d_elem );
}

template<typename Float>
void PLEGMA_Field<Float>::cscale(std::complex<Float> val){
  if(!isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
  cuBLAS::cscal(field_length*total_length, reinterpret_cast<Float(&)[2]>(val), d_elem );
}

template<typename FloatOut, typename FloatIn>
static void cudaCopyOrCast(PLEGMA_Field<FloatOut> &fieldOut, PLEGMA_Field<FloatIn> &fieldIn){
  assert(fieldOut.checkVolume(fieldIn));
  if(typeid(FloatIn) != typeid(FloatOut) )
    cudaCast(toField2<pFloat2>(fieldOut), toField2<pFloat2>(fieldIn));
  else
    cudaMemcpy(fieldOut.D_elem(), fieldIn.D_elem(), fieldIn.Bytes_total(), 
	       cudaMemcpyDeviceToDevice);
  checkQudaError();
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
  assert(this->checkVolume(f));
  if(field_length != f.Field_length()) PLEGMA_error("The d.o.f of the fields does not match\n");
  if(total_length != f.Total_length()) PLEGMA_error("The lattice points of the fields do not match\n");
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

// field4D <- field3D
template<typename Float>
void PLEGMA_Field<Float>::absorb(const PLEGMA_Field3D<Float> &field, int global_it, bool forcetozero){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  assert(field.Field_length() == this->Field_length());
  if (forcetozero == true){
    this->zero_where(allocation);
  }
  
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  if(not is_myIt) return;
  
  size_t V4 = HGC_localVolume*2;
  size_t V3 = HGC_localVolume3D*2;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int i = 0; i < this->Field_length(); i++) {
    pointer_src = (field.D_elem() + i*V3);
    pointer_dst = (this->D_elem() + i*V4 + my_it*V3);
    cudaMemcpy(pointer_dst, pointer_src, V3 * sizeof(Float), cudaMemcpyDeviceToDevice);
  }
  checkQudaError();
}

template<typename Float>
void PLEGMA_Field<Float>::writeLIME(std::string filename, bool unloadFromDev) const{
  if(total_length != HGC_localVolume) PLEGMA_error("Writing of 3D fields is not supported");
  FILE *fid;
  LimeWriter *limewriter = (LimeWriter*)NULL;
  if(unloadFromDev) unload();
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
void PLEGMA_Field<Float>::readLIME(std::string filename, bool loadToDev){
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
  if(loadToDev) if(isAllocDevice) load();
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
  
  if(total_length == HGC_localVolume3D) {
    descr += "/z/y/x";
    // Volume
    for(int i=N_DIMS-2; i>=0; i--) {
      shape.push_back(HGC_totalL[i]);
      lshape.push_back(HGC_localL[i]);
      start.push_back(HGC_procPosition[i]*HGC_localL[i]);
    }
    if(not includesActiveTimeSlice()) lshape[0]=0;
  } else {
    descr += "/t/z/y/x";
    // Volume
    for(int i=N_DIMS-1; i>=0; i--) {
      shape.push_back(HGC_totalL[i]);
      lshape.push_back(HGC_localL[i]);
      start.push_back(HGC_procPosition[i]*HGC_localL[i]);
    }
  }

  //re-im
  descr += "/re-im";
  shape.push_back(2);
  lshape.push_back(2);
  start.push_back(0);

  return descr;
}


template<typename Float>
void PLEGMA_Field<Float>::writeHDF5(std::string filename, bool unloadFromDev) const{
  if(total_length != HGC_localVolume && total_length != HGC_localVolume3D)
    PLEGMA_error("Writing of 3D fields is not supported");
  assert(isAllocHost);
  if(unloadFromDev) unload();
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
void PLEGMA_Field<Float>::absorbTimeslice(PLEGMA_Field<Float> &srcfield, int global_it, bool forcetozero){
  if(!this->isAllocDevice) PLEGMA_error("This function needs allocation on the device to work\n");
  if(!srcfield.IsAllocDevice()) PLEGMA_error("This function needs allocation of input field on the device to work\n");
  
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  if( this->field_name.compare(srcfield.Field_name()) != 0) PLEGMA_error("Fields types does not match\n");

  //check dimensions
  
  int my_it = global_it - comm_coord(3) * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;


  for(int i = 0 ; i < this->field_length; i++){
    if( forcetozero )
      cudaMemset( this->d_elem + i*V4*2, 0, V4*2*sizeof(Float));
    if(is_myIt){
      pointer_dst = (this->d_elem + i*V4*2 + my_it*V3*2);
      pointer_src = (srcfield.D_elem() + i*V4*2 + my_it*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
    }
  }
  comm_barrier();
  checkQudaError();
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

template<typename Float>
void PLEGMA_Field<Float>::SU3Trace(PLEGMA_Su3field<Float> &su3field){
  SU3Trace_k(*this,su3field);
}
template class PLEGMA_Field<float>;
template class PLEGMA_Field<double>;
// Forcing initialization of the following cases
template void PLEGMA_Field<float>::copy<float>(PLEGMA_Field<float> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<float>::copy<double>(PLEGMA_Field<double> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<double>::copy<float>(PLEGMA_Field<float> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<double>::copy<double>(PLEGMA_Field<double> &f, ALLOCATION_FLAG where);
template void PLEGMA_Field<float>::mulMomentumPhases<int>(std::vector<int> mom, int sign);
template void PLEGMA_Field<float>::mulMomentumPhases<float>(std::vector<float> mom, int sign);
template void PLEGMA_Field<float>::mulMomentumPhases<double>(std::vector<double> mom, int sign);
template void PLEGMA_Field<double>::mulMomentumPhases<int>(std::vector<int> mom, int sign);
template void PLEGMA_Field<double>::mulMomentumPhases<float>(std::vector<float> mom, int sign);
template void PLEGMA_Field<double>::mulMomentumPhases<double>(std::vector<double> mom, int sign);

// field3D <- field4D
template<typename Float>
void PLEGMA_Field3D<Float>::absorb(const PLEGMA_Field<Float> &field, int global_it, bool broadcast){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  assert(field.Field_length() == this->Field_length());
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
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
    }
    if (broadcast == true){
      int time_rank=global_it/HGC_localL[3];
      Float *temp=(Float *)malloc(sizeof(Float)*V3);
      cudaMemcpy(temp, pointer_dst, V3* sizeof(Float), cudaMemcpyDeviceToHost);
      MPI_Bcast(temp, V3 , MPI_Type<Float>(), time_rank, HGC_timeComm);
      cudaMemcpy(pointer_dst, temp, V3* sizeof(Float), cudaMemcpyHostToDevice);
      free(temp);
    }
    if (broadcast == false && !(this->activeTimeSlice)){
      cudaMemset(pointer_dst, 0, V3 * sizeof(Float));
    }
  }
  checkQudaError();
}


template<typename Float>
std::complex<Float> PLEGMA_Field3D<Float>::dot(PLEGMA_Field3D<Float> &fieldIn){
  // TODO: need to think about appropriate communicator
  if (HGC_localVolume != HGC_totalVolume)
    PLEGMA_warning("3D Vector dot might not work with multiple MPI ranks\n");
  return cuBLAS::dot(this->total_length*this->field_length, this->d_elem, fieldIn.D_elem(), HGC_fullComm);
}

	template class PLEGMA_Field3D<float>;
template class PLEGMA_Field3D<double>;
