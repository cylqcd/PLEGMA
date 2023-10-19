#include <PLEGMA_Field.h>
#include <PLEGMA_FT.h>
#include <PLEGMA_Thrust.h>
#include <vector>
#include <algorithm>
#include <PLEGMA_BLAS.h>
#include <kernels/PLEGMA_FT.cuh>
#include <complex>
#include <cmath>
#include <malloc_quda.h>
using namespace plegma;



template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(int Q2_max, int D3D4, bool accum, int dimT):
  Q2_max(Q2_max), dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions %d\n",dims);
  if(Q2_max < 0) PLEGMA_error("The maximum number of Q2 cannot be negative\n");
  if(dimT<0 || dimT>HGC.localL[DIM_T]) PLEGMA_error("The time dimension cannot be negative or larger than local size\n");
  createMom();
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(std::vector<int> mom, int D3D4, bool accum, int dimT):
  dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  if(mom.size() != dims) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT %zu %d\n",mom.size(), dims);
  VFloat momF(mom.begin(),mom.end());
  momList.push_back(momF);
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT( std::vector<std::vector<int>> &moms, int D3D4, bool accum, int dimT ):
  dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  for(auto &mom : moms){
    if(mom.size()%dims != 0) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT %zu %d",mom.size(),dims);
    VFloat momF(mom.begin(),mom.end());
    momList.push_back(momF);
  }
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(std::vector<float> mom, int D3D4, bool accum, int dimT):
  dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  if(mom.size() != dims) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT %zu %d\n",mom.size(), dims);
  VFloat momF(mom.begin(),mom.end());
  momList.push_back(momF);
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT( std::vector<std::vector<float>> &moms, int D3D4, bool accum, int dimT ):
  dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  for(auto &mom : moms){
    if(mom.size()%dims != 0) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT %zu %d",mom.size(),dims);
    VFloat momF(mom.begin(),mom.end());
    momList.push_back(momF);
  }
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(std::vector<double> mom, int D3D4, bool accum, int dimT):
  dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  if(mom.size() != dims) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT %zu %d\n",mom.size(), dims);
  VFloat momF(mom.begin(),mom.end());
  momList.push_back(momF);
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT( std::vector<std::vector<double>> &moms, int D3D4, bool accum, int dimT ):
  dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(D3D4==3?dimT:1), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  for(auto &mom : moms){
    if(mom.size()%dims != 0) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT %zu %d",mom.size(),dims);
    VFloat momF(mom.begin(),mom.end());
    momList.push_back(momF);
  }
}


template<typename Float>
void PLEGMA_FT<Float>::zero(){
  if(h_elem) memset(h_elem.get(), 0, Nmoms()*dimT*dof*2*sizeof(Float));
}

template<typename Float>
void PLEGMA_FT<Float>::createMom(){
  VFloat v3 = {0,0,0};
  VFloat v4 = {0,0,0,0};
  for(int iQ = 0 ; iQ <= Q2_max ; iQ++)
    for(int nx = iQ ; nx >= -iQ ; nx--)
      for(int ny = iQ ; ny >= -iQ ; ny--)
        for(int nz = iQ ; nz >= -iQ ; nz--){
  	  if(dims == 3){
  	    if( nx*nx + ny*ny + nz*nz == iQ ){
	      v3[0]=nx; v3[1]=ny; v3[2]=nz;
  	      momList.push_back(v3);
	    }
  	  }
  	  else{
  	    for(int nt = iQ; nt >= -iQ; nt--)
  	      if( nx*nx + ny*ny + nz*nz + nt*nt == iQ ){
		v4[0]=nx; v4[1]=ny; v4[2]=nz; v4[3]=nt;
  		momList.push_back(v4);
	      }
  	  }
  	}
}

template<typename Float>
void PLEGMA_FT<Float>::checkAllocation(int newDof){
  dof = newDof;
  sizeN=Nmoms()*dimT*dof*2;
  h_elem.reset(new Float[sizeN]);
  zero();
}

template<typename Float>
std::shared_ptr<tex_mom_list> PLEGMA_FT<Float>::getTexMomList() {
#ifdef __NVCC__
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  desc.f = cudaChannelFormatKindSigned;
#elif defined (__HIP__)
  hipChannelFormatDesc desc;
  memset(&desc, 0, sizeof(hipChannelFormatDesc));
  desc.f = hipChannelFormatKindSigned;
#endif
  desc.x = 8*4;
  desc.y = 8*4;
  desc.z = 8*4;
  desc.w = 8*4;
 
#ifdef __NVCC__ 
  cudaResourceDesc resDesc;
#elif defined (__HIP__)
  hipResourceDesc resDesc;
#endif
  memset(&resDesc, 0, sizeof(resDesc));
#ifdef __NVCC__
  resDesc.resType = cudaResourceTypeLinear;
#elif defined (__HIP__)
  resDesc.resType = hipResourceTypeLinear;
#endif
  resDesc.res.linear.desc = desc;

  void * devPtr;
  int hostPtr[Nmoms()*N_DIMS];
  memset(hostPtr, 0, sizeof(hostPtr));
  devPtr=device_malloc(sizeof(hostPtr));
//  cudaMalloc(&devPtr, sizeof(hostPtr));
  Float intp;
  for(int i=0; i<Nmoms(); i++) {
    for(int j=0; j<dims; j++) {
      if(abs(std::modf(momList[i][j],&intp)) > std::numeric_limits<Float>::epsilon()) PLEGMA_warning("Function getTexMomList expects integers momenta but non integers are given");
      hostPtr[i*N_DIMS+j]=(int) std::lround(momList[i][j]);
    }
  }
  qudaMemcpy(devPtr, hostPtr, sizeof(hostPtr), qudaMemcpyHostToDevice );
  resDesc.res.linear.devPtr = devPtr;
  resDesc.res.linear.sizeInBytes = sizeof(hostPtr);

#if defined __NVCC__
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
  
  return std::shared_ptr<tex_mom_list>(new tex_mom_list(Nmoms(), tex, devPtr), [](tex_mom_list* moms) { cudaDestroyTextureObject(moms->tex); device_free(moms->devPtr);});
#elif defined (__HIP__)
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;

  hipTextureObject_t tex;
  hipCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  return std::shared_ptr<tex_mom_list>(new tex_mom_list(Nmoms(), tex, devPtr), [](tex_mom_list* moms) { hipDestroyTextureObject(moms->tex); device_free(moms->devPtr);});
#endif
}

template<typename Float>
void PLEGMA_FT<Float>::applyNaive(const PLEGMA_Field<Float> &f, int sign){
  if(dims == 4) PLEGMA_error("This FT implementation is implemented for a 3D transformation only");
  if(f.Total_length() != HGC.localVolume && dims == 4) PLEGMA_error("Cannot do a 4D FT on a 3D field\n");
  if(f.Total_length() != HGC.localVolume) dimT=1; // if the field is 3D
  checkAllocation(f.Field_length());
  field_name = f.Field_name();
  site_shape = f.getSiteShape();
  auto moms = this->getTexMomList();
  if(!accum) zero();
  for(int it =0 ; it < dimT; it++)
    fourier_transform_3D_k(*this,f,*moms,it,sign);
}

template<typename Float>
void PLEGMA_FT<Float>::applyFFT(const PLEGMA_Field<Float> &f, int sign){
  checkAllocation(f.Field_length());
  field_name = f.Field_name();
  site_shape = f.getSiteShape();
  PLEGMA_error("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::applyGEMV(const PLEGMA_Field<Float> &f, int sign){
  if(f.Total_length() != HGC.localVolume && dims == 4) PLEGMA_error("Cannot do a 4D FT on a 3D field\n");
  if(f.Total_length() != HGC.localVolume) dimT=1; // if the field is 3D
  checkAllocation(f.Field_length());
  field_name = f.Field_name();
  site_shape = f.getSiteShape();
  if(!accum) zero();
  FT_gemv<Float>(*this,f,momList,sign);
}

template<typename Float>
void PLEGMA_FT<Float>::apply(const PLEGMA_Field<Float> &f, FT_TYPE type, int sign){
  switch(type){
  case FT_NAIVE: applyNaive(f,sign); break;
  case FT_GEMV: applyGEMV(f,sign); break;
  case FT_FFT: applyFFT(f,sign); break;
  }
}

template<typename Float>
void PLEGMA_FT<Float>::mulConstMomentumPhases(Vint src, int sign){
  if(dims == 3 && src.size() != 3) PLEGMA_error("src size is incompatible with the dimensionality of the FT");
  if(dims == 4 && src.size() != 4) PLEGMA_error("src size is incompatible with the dimensionality of the FT");
  if(sign != +1 && sign != -1) PLEGMA_error("Sign should be either +1 or -1\n");
  Float phase;
  std::complex<Float> expPhase;
  if(!h_elem) PLEGMA_error("Apply first FT and then the const phases");
  std::complex<Float> *h2 = (std::complex<Float> *) h_elem.get();
  for(int imom = 0; imom < Nmoms(); imom++){
    phase=0.;
    for(int i = 0; i < dims; i++) phase += ((Float) src[i] * momList[imom][i])/((Float) HGC.totalL[i]);
    phase *= 2. * PI;
    expPhase = (std::complex<Float>) { cos(phase), sign*sin(phase) };
    for(int idf = 0 ; idf < dof; idf++)
      for(int it = 0 ; it < dimT; it++)
	h2[it*dof*Nmoms()+idf*Nmoms()+imom] *= expPhase;
  }
}

template<typename Float>
void PLEGMA_FT<Float>::scale(Float a){
  if(!h_elem) PLEGMA_error("Apply first FT and then you can scale it");
  cBLAS::scal(sizeN/2, a, h_elem.get());
}

template<typename Float>
void PLEGMA_FT<Float>::store3DFTs(std::complex<Float> *Ts, int timeshift, bool append) const{
  if(dims == 4 && timeshift > 0) PLEGMA_error("The temporal dimension has been reduced therefore cannot shift it\n");
  if(!h_elem) PLEGMA_error("Memory not allocated cannot write data");
  if(dims == 3 && dimT != HGC.localL[DIM_T]) PLEGMA_error("Custom time dimension is not supported in storing (TODO)\n");

  std::shared_ptr<Float> helem_global = h_elem;
  if(dimT != 1 && HGC.nProc[3] != 1 && HGC.spaceRank == 0){
    helem_global.reset(new Float[HGC.nProc[DIM_T]*sizeN]);
    if(HGC.timeComm == MPI_COMM_NULL) PLEGMA_error("Try to use a NULL communicator for MPI Gather which will give an error");
    int error = MPI_Gather(h_elem.get(), sizeN, MPI_Type<Float>(), helem_global.get(), sizeN, MPI_Type<Float>(),0,HGC.timeComm);
    if(error != MPI_SUCCESS) PLEGMA_error("MPI_Gather with %d\n",error);
  }

  if(comm_rank() == 0){
    int T = (dimT != 1)?HGC.totalL[DIM_T]:1;
    for(int idf = 0 ; idf < dof; idf++)
      for(int it = 0 ; it < T; it++){
        int its = (it + timeshift)%HGC.totalL[DIM_T];
        for(int imom = 0; imom < Nmoms(); imom++) 
	  Ts[its*dof*Nmoms()+idf*Nmoms()+imom] = std::complex<Float>(helem_global.get()[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+0],
								     helem_global.get()[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+1]);
      }
  }
  comm_barrier();
}

template<typename Float>
void PLEGMA_FT<Float>::writeASCII(std::string filename, int timeshift, bool append) const{
  if(dims == 4 && timeshift > 0) PLEGMA_error("The temporal dimension has been reduced therefore cannot shift it\n");
  if(!h_elem) PLEGMA_error("Memory not allocated cannot write data");
  if(dims == 3 && dimT != HGC.localL[DIM_T]) PLEGMA_error("Custom time dimension is not supported in writing (TODO)\n");

  std::shared_ptr<Float> helem_global = h_elem;
  if(dimT != 1 && HGC.nProc[3] != 1 && HGC.spaceRank == 0){
    helem_global.reset(new Float[HGC.nProc[DIM_T]*sizeN]);
    if(HGC.timeComm == MPI_COMM_NULL) PLEGMA_error("Try to use a NULL communicator for MPI Gather which will give an error");
    int error = MPI_Gather(h_elem.get(), sizeN, MPI_Type<Float>(), helem_global.get(), sizeN, MPI_Type<Float>(),0,HGC.timeComm);
    if(error != MPI_SUCCESS) PLEGMA_error("MPI_Gather with %d\n",error);
  }
  
  if(comm_rank() == 0){
    // If append==true, the file needs to be initialized before calling this method instad of initializing it using append==false
    //  Othereise, the order will be messed, and some iterms will be missing.
    FILE *ptr = append?fopen(filename.c_str(), "a"):fopen(filename.c_str(), "w");
    if(ptr == NULL) PLEGMA_error("Cannot open file:%s for writting\n",filename.c_str());
    int T = (dimT != 1)?HGC.totalL[DIM_T]:1;
    for(int idf = 0 ; idf < dof; idf++)
      for(int it = 0 ; it < T; it++){
	int its = (it + timeshift)%HGC.totalL[DIM_T];
	for(int imom = 0; imom < Nmoms(); imom++)
	  fprintf(ptr, "%d %d  %+d %+d %+d \t %+16.15e %+15.15e\n", idf,it,(int) round(momList[imom][0]),(int) round(momList[imom][1]),(int) round(momList[imom][2]),
		  helem_global.get()[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+0], helem_global.get()[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+1] );
      }
    fclose(ptr);
  }
  comm_barrier();
}


template<typename Float>
std::string PLEGMA_FT<Float>::
fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start, int timeshift) const{
  std::string descr;
  
  // Time
  if(dims==3 && dimT == HGC.localL[DIM_T]) {
    descr += "/time";
    shape.push_back(HGC.totalL[DIM_T]);
    lshape.push_back(HGC.localL[DIM_T]);
    start.push_back((HGC.procPosition[DIM_T]*HGC.localL[DIM_T] + HGC.totalL[DIM_T] - timeshift) % HGC.totalL[DIM_T]);
  } else {
    assert(dimT==1);
  }

  // Field shape
  if(!this->site_shape.empty()) {
    descr += "/" + field_name;
    for(auto s : this->site_shape) {
      shape.push_back(s);
      lshape.push_back(s);
      start.push_back(0);    
    }
  }

  // Moms
  descr += "/moms";
  shape.push_back(Nmoms());
  lshape.push_back(Nmoms());
  start.push_back(0);

  //re-im
  descr += "/re-im";
  shape.push_back(2);
  lshape.push_back(2);
  start.push_back(0);

  return descr;
}

static size_t
use_multiple_writers(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start, int &nWriters, int id) {
  if(nWriters==1) return 0;
  int usedWriters = 1;
  size_t shift = 0;
  for(int i=0; i<lshape.size(); i++) {
    int iSize = (lshape[i] + nWriters-1)/nWriters;
    int iWriters = (lshape[i] + iSize-1)/iSize;
    nWriters /= iWriters;
    usedWriters *= iWriters;
    int iId = id % iWriters;
    id /= iWriters;
    int iShift = iSize*iId;
    shift = shift*lshape[i] + iShift;
    if(iShift+iSize > lshape[i])
      lshape[i] -= iShift;
    else
      lshape[i] = iSize;
    start[i] = (start[i] + iShift) % shape[i];
  }
  nWriters = usedWriters;
  return shift;
}

template<typename T>
static std::string str(T begin, T end) {
  std::stringstream ss;
  bool first = true;
  for (; begin != end; begin++) {
    if (!first) ss << ", ";
    ss << *begin;
    first = false;
  }
  return ss.str();
}

template<typename Float>
void PLEGMA_FT<Float>::
writeHDF5(std::string filename, int timeshift, bool append) const{
  if(dims == 3 && dimT != HGC.localL[DIM_T]) PLEGMA_error("Custom time dimension is not supported in writing (TODO)\n");
  std::vector<hsize_t> shape, lshape, start;
  std::string descr = fill_H5_shapes(shape, lshape, start, timeshift);
  
  hsize_t writeSize = 1;
  for(auto l: lshape) writeSize*=l;
  assert(sizeN==writeSize);

  size_t shift = 0;
  if(dims==3 && dimT != HGC.localL[DIM_T]) { // then it was a 3D Field. Using timeshift to determine the origin
    int my_it = timeshift - HGC.procPosition[DIM_T] * HGC.localL[DIM_T];
    bool is_myIt = (my_it >= 0) && ( my_it < HGC.localL[DIM_T] );
    if(!is_myIt) lshape[0]=0; // not writing
  } else {
    int nWriters = (dims==4) ? HGC.fullSize : HGC.spaceSize;
    int id = (dims==4) ? HGC.fullRank : HGC.spaceRank;
    shift = use_multiple_writers(shape, lshape, start, nWriters, id);
    if(id >= nWriters) lshape[0] = 0; // not writing
    if(HGC.verbosity > 3) {
      std::string out = "rank: "+std::to_string(id)+
	", shape: ("+str(shape.begin(), shape.end())+
	"), lshape: ("+str(lshape.begin(), lshape.end())+
	"), start: ("+str(start.begin(), start.end())+
	"), shift: "+std::to_string(shift)+"\n";
      printf(out.c_str());
    }
  }
  
  std::string dataset = "FT_data"; // default name
  
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

  HDF5 writer(filename, HGC.fullComm);

  writer.write_dataset(dataset, h_elem.get()+shift, shape, lshape, start);
  writer.write_attribute(dataset, "description", descr);

  std::vector<hsize_t> momShape = { (hsize_t) dims };
  std::vector<int> mvec;
  for(auto mv: MomList()) for(auto m: mv) mvec.push_back(m);
  writer.write_dataset("mvec", mvec, momShape);
}

template class plegma::PLEGMA_FT<float>;
template class plegma::PLEGMA_FT<double>;
#if 0
template plegma::PLEGMA_FT<float>::PLEGMA_FT<int>(std::vector<int>,int,bool,int);
template plegma::PLEGMA_FT<double>::PLEGMA_FT<int>(std::vector<int>,int,bool,int);
template plegma::PLEGMA_FT<float>::PLEGMA_FT<float>(std::vector<float>,int,bool,int);
template plegma::PLEGMA_FT<double>::PLEGMA_FT<float>(std::vector<float>,int,bool,int);
template plegma::PLEGMA_FT<float>::PLEGMA_FT<double>(std::vector<double>,int,bool,int);
template plegma::PLEGMA_FT<double>::PLEGMA_FT<double>(std::vector<double>,int,bool,int);
template plegma::PLEGMA_FT<float>::PLEGMA_FT<int>( std::vector<std::vector<int>> &,int,bool,int);
template plegma::PLEGMA_FT<double>::PLEGMA_FT<int>( std::vector<std::vector<int>> &,int,bool,int);
#endif
