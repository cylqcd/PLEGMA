#include <PLEGMA_Field.h>
#include <PLEGMA_FT.h>
#include <PLEGMA_Thrust.h>
#include <vector>
#include <algorithm>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_FT.cuh>
#include <complex>
#include <cmath>
using namespace plegma;

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(int Q2_max, int D3D4, bool accum):
Q2_max(Q2_max), isAllocated(false), dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(0), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  dimT = (dims == 3) ? HGC_localL[3] : 1; // when apply, if a 3D field set dimT=1 even if dims=3
  if(Q2_max < 0) PLEGMA_error("The maximum number of Q2 cannot be negative\n");
  createMom();
  
}

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(std::vector<int> mom, int D3D4, bool accum):
  isAllocated(false), dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(0), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  dimT = (dims == 3) ? HGC_localL[3] : 1; // when apply, if a 3D field set dimT=1 even if dims=3
  if(mom.size() != dims) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT");
  momList.push_back(mom);
  
}


template<typename Float>
PLEGMA_FT<Float>::~PLEGMA_FT(){
  if(isAllocated) hostFree(h_elem, sizeN*sizeof(Float));
}

template<typename Float>
void PLEGMA_FT<Float>::zero(){
  if(isAllocated) memset(h_elem, 0, Nmoms()*dimT*dof*2*sizeof(Float));
}

template<typename Float>
void PLEGMA_FT<Float>::createMom(){
  std::vector<int> v3 = {0,0,0};
  std::vector<int> v4 = {0,0,0,0};
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
  if(isAllocated && dof == newDof) return;
  try{
    if(!isAllocated){dof = newDof; sizeN = Nmoms()*dimT*dof*2; hostMalloc(h_elem, sizeN*sizeof(Float));}
    else{
      if(dof != newDof){
	dof = newDof;
	hostFree(h_elem, sizeN*sizeof(Float));
	sizeN = Nmoms()*dimT*dof*2;
	hostMalloc(h_elem, sizeN*sizeof(Float));
      }
    }
  }
  catch (std::bad_alloc& err){
    PLEGMA_error(err.what());
  }
  isAllocated=true;
  zero();
}

template<typename Float>
tex_mom_list PLEGMA_FT<Float>::getTexMomList() {
  tex_mom_list tex_mom;
  tex_mom.Nmoms=Nmoms();
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  desc.f = cudaChannelFormatKindSigned;
  desc.x = 8*4;
  desc.y = 8*4;
  desc.z = 8*4;
  desc.w = 8*4;

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.desc = desc;

  size_t bytes = tex_mom.Nmoms*4*sizeof(int);
  void * devPtr;
  int * hostPtr;
  hostMalloc(hostPtr, bytes);
  memset(hostPtr, 0, sizeof(bytes));
  cudaMalloc(&devPtr, bytes);
  for(int i=0; i<tex_mom.Nmoms; i++) {
    for(int j=0; j<dims; j++) {
      hostPtr[i*4+j]=momList[i][j];
    }
  }
  cudaMemcpy(devPtr, hostPtr, bytes, cudaMemcpyHostToDevice );
  hostFree(hostPtr, bytes);
  resDesc.res.linear.devPtr = devPtr;
  resDesc.res.linear.sizeInBytes = bytes;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&tex_mom.tex, &resDesc, &texDesc, NULL);
  checkCudaError();
  return tex_mom;
}
    

template<typename Float>
void PLEGMA_FT<Float>::applyNaive(const PLEGMA_Field<Float> &f, int sign){
  PLEGMA_error("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::applyFFT(const PLEGMA_Field<Float> &f, int sign){
  PLEGMA_error("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::apply(const PLEGMA_Field<Float> &f, int sign){
  if(f.Total_length() != HGC_localVolume && dims == 4) PLEGMA_error("Cannot do a 4D FT on a 3D field\n");
  if(f.Total_length() != HGC_localVolume) dimT=1; // if the field is 3D
  checkAllocation(f.Field_length());
  if(!accum) zero();
  FT<Float>(*this,f,momList,sign);
}

template<typename Float>
void PLEGMA_FT<Float>::mulConstMomentumPhases(Vint src, int sign){
  if(dims == 3 && src.size() != 3) PLEGMA_error("Src size is incompatible with the dimensionality of the FT");
  if(dims == 4 && src.size() != 4) PLEGMA_error("Src size is incompatible with the dimensionality of the FT");
  if(sign != +1 && sign != -1) PLEGMA_error("Sign should be either +1 or -1\n");
  Float phase;
  std::complex<Float> expPhase;
  if(!isAllocated) PLEGMA_error("Apply first FT and then the const phases");
  std::complex<Float> *h2 = (std::complex<Float> *) h_elem;
  for(int imom = 0; imom < Nmoms(); imom++){
    phase=0.;
    for(int i = 0; i < dims; i++) phase += ((Float) src[i] * momList[imom][i])/((Float) HGC_totalL[i]);
    phase *= 2. * PI;
    expPhase = (std::complex<Float>) { cos(phase), sign*sin(phase) };
    for(int idf = 0 ; idf < dof; idf++)
      for(int it = 0 ; it < dimT; it++)
	h2[it*dof*Nmoms()+idf*Nmoms()+imom] *= expPhase;
  }
}

template<typename Float>
void PLEGMA_FT<Float>::scale(Float a){
  if(!isAllocated) PLEGMA_error("Apply first FT and then you can scale it");
  cBLAS::scal(sizeN/2, a, h_elem);
}


template<typename Float>
void PLEGMA_FT<Float>::writeToFile(std::string filename, FILE_WRITE_FORMAT outputFormat, int timeshift){
  if(dims == 4 && timeshift > 0) PLEGMA_error("The temporal dimension has been reduced therefore cannot shift it\n");
  if(!isAllocated) PLEGMA_error("Memory not allocated cannot write data");
  if(outputFormat == ASCII_FORM){
    Float *helem_global=NULL;
    bool gAlloc=false;
    if(dimT != 1 && HGC_nProc[3] != 1 && HGC_spaceRank == 0){
      hostMalloc(helem_global, HGC_nProc[3]*sizeN*sizeof(Float));
      gAlloc=true;
      if(HGC_timeComm == MPI_COMM_NULL) PLEGMA_error("Try to use a NULL communicator for MPI Gather which will give an error");
      int error = MPI_Gather(h_elem, sizeN, MPI_Type(h_elem), helem_global, sizeN, MPI_Type(h_elem),0,HGC_timeComm);
      if(error != MPI_SUCCESS) PLEGMA_error("MPI_Gather with %d\n",error);
    }
    else
      helem_global = h_elem;
    if(comm_rank() == 0){
      FILE *ptr = fopen(filename.c_str(), "w");
      if(ptr == NULL) PLEGMA_error("Cannot open file:%s for writting\n",filename.c_str());
      int T = (dimT != 1)?HGC_totalL[3]:1;
      for(int idf = 0 ; idf < dof; idf++)
	for(int it = 0 ; it < T; it++){
	  int its = (it + timeshift)%HGC_totalL[3];
	  for(int imom = 0; imom < Nmoms(); imom++)
	    fprintf(ptr, "%d %d  %+d %+d %+d \t %+e %+e\n", idf,it, momList[imom][0], momList[imom][1], momList[imom][2],
		    helem_global[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+0], helem_global[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+1] );
	}
      fclose(ptr);
    }
    if(gAlloc) hostFree(helem_global, HGC_nProc[3]*sizeN*sizeof(Float));
    comm_barrier();
  }
  else if(outputFormat == HDF5_FORM){
    PLEGMA_error("Not implemented yet");
  }
  else
    PLEGMA_error("The output file format is unknown");
}

template class PLEGMA_FT<float>;
template class PLEGMA_FT<double>;
