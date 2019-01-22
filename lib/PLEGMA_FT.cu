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
  Q2_max(Q2_max), isAllocated(false), dof(0), h_elem(nullptr), dims(D3D4), dimT(0), accum(accum){
  if(dims!= 3 && dims !=4) errorQuda("This class transforms only 3 and 4 dimensions\n");
  dimT = (dims == 3) ? GK_localL[3] : 1; // when apply, if a 3D field set dimT=1 even if dims=3
  if(Q2_max < 0) errorQuda("The maximum number of Q2 cannot be negative\n");
  createMom();
}

template<typename Float>
PLEGMA_FT<Float>::~PLEGMA_FT(){
  if(isAllocated) delete[] h_elem;
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
    if(!isAllocated){dof = newDof; h_elem = new Float[Nmoms()*dimT*dof*2];}
    else{
      if(dof != newDof){
	dof = newDof;
	delete[] h_elem;
	h_elem = new Float[Nmoms()*dimT*dof*2];
      }
    }
  }
  catch (std::bad_alloc& err){
    errorQuda(err.what());
  }
  isAllocated=true;
  zero();
}

template<typename Float>
void PLEGMA_FT<Float>::applyNaive(const PLEGMA_Field<Float> &f, int sign){
  errorQuda("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::applyFFT(const PLEGMA_Field<Float> &f, int sign){
  errorQuda("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::apply(const PLEGMA_Field<Float> &f, int sign){
  if(f.Total_length() != GK_localVolume && dims == 4) errorQuda("Cannot do a 4D FT on a 3D field\n");
  if(f.Total_length() != GK_localVolume) dimT=1; // if the field is 3D
  checkAllocation(f.Field_length());
  if(!accum) zero();
  FT<Float>(*this,f,momList,sign);
}

template<typename Float>
void PLEGMA_FT<Float>::mulConstMomentumPhases(Vint src, int sign){
  if(dims == 3 && src.size() != 3) errorQuda("Src size is incompatible with the dimensionality of the FT");
  if(dims == 4 && src.size() != 4) errorQuda("Src size is incompatible with the dimensionality of the FT");
  if(sign != +1 && sign != -1) errorQuda("Sign should be either +1 or -1\n");
  Float phase;
  std::complex<Float> expPhase;
  if(!isAllocated) errorQuda("Apply first FT and then the const phases");
  std::complex<Float> *h2 = (std::complex<Float> *) h_elem;
  for(int imom = 0; imom < Nmoms(); imom++){
    phase=0.;
    for(int i = 0; i < dims; i++) phase += ((Float) src[i] * momList[imom][i])/((Float) GK_totalL[i]);
    phase *= 2. * PI;
    expPhase = (std::complex<Float>) { cos(phase), sign*sin(phase) };
    for(int idf = 0 ; idf < dof; idf++)
      for(int it = 0 ; it < dimT; it++)
	h2[it*dof*Nmoms()+idf*Nmoms()+imom] *= expPhase;
  }
}

template<typename Float>
void PLEGMA_FT<Float>::writeToFile(std::string filename, FILE_WRITE_FORMAT outputFormat, int timeshift){
  if(dims == 4 && timeshift > 0) errorQuda("The temporal dimension has been reduced therefore cannot shift it\n");
  if(!isAllocated) errorQuda("Memory not allocated cannot write data");
  if(outputFormat == ASCII_FORM){
    Float *helem_global=NULL;
    if(dimT != 1){
      int sizeN= Nmoms()*GK_localL[3]*dof*2;
      helem_global = (Float*) malloc(GK_nProc[3]*sizeN*sizeof(Float));
      if(helem_global == NULL) errorQuda("Allocation failed\n");
      MPI_Gather(h_elem, sizeN, MPI_Type(h_elem), helem_global, sizeN, MPI_Type(h_elem),0,GK_timeComm);
    }
    else
      helem_global = h_elem;
    if(comm_rank() == 0){
      FILE *ptr = fopen(filename.c_str(), "w");
      if(ptr == NULL) errorQuda("Cannot open file:%s for writting\n",filename.c_str());
      int T = (dimT != 1)?GK_totalL[3]:1;
      for(int idf = 0 ; idf < dof; idf++)
	for(int it = 0 ; it < T; it++){
	  int its = (it + timeshift)%GK_totalL[3];
	  for(int imom = 0; imom < Nmoms(); imom++)
	    fprintf(ptr, "%d %d  %+d %+d %+d \t %+e %+e\n", idf,it, momList[imom][0], momList[imom][1], momList[imom][2],
		    helem_global[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+0], helem_global[its*dof*Nmoms()*2+idf*Nmoms()*2+imom*2+1] );
	}
      fclose(ptr);
    }
    if(dimT != 1)free(helem_global);
  }
  else if(outputFormat == HDF5_FORM){
    errorQuda("Not implemented yet");
  }
  else
    errorQuda("The output file format is unknown");
}

template class PLEGMA_FT<float>;
template class PLEGMA_FT<double>;
