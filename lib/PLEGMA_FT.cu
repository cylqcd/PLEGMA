#include <PLEGMA_FT.h>
using namespace plegma;

template<typename Float>
PLEGMA_FT<Float>::PLEGMA_FT(int Q2_max, int D3D4):
  Q2_max(Q2_max), isAllocated(false), dof(0), h_elem(nullptr), dims(D3D4), dimT(0){
  if(dims!= 3 && dims !=4) errorQuda("This class transforms only 3 and 4 dimensions\n");
  dimT = (dims == 3) ? GK_localL[3] : 1;
  if(Q2_max < 0) errorQuda("The maximum number of Q2 cannot be negative\n");
  createMom();
}

template<typename Float>
PLEGMA_FT<Float>::~PLEGMA_FT(){
  if(isAllocated) delete[] h_elem;
}

template<typename Float>
void PLEGMA_FT<Float>::createMom(){
  std::vector<int> v3,v4;
  v3.reserve(3); v4.reserve(4);
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
  try{
    if(!isAllocated){dof = newDof; h_elem = new Float[Nmoms()*dimT*dof*2];}
    else{
      if(dof != newDof){
	dof = newDof;
	if(isAllocated) delete[] h_elem;
	h_elem = new Float[Nmoms()*dimT*dof*2];
      }
    }
  }
  catch (std::bad_alloc& err){
    errorQuda(err.what());
  }
  isAllocated=true;
}

template<typename Float>
void PLEGMA_FT<Float>::applyNaive(const PLEGMA_Field<Float> &f){
  errorQuda("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::applyFFT(const PLEGMA_Field<Float> &f){
  errorQuda("Not implemented yet");
}

template<typename Float>
void PLEGMA_FT<Float>::apply(const PLEGMA_Field<Float> &f){
  if(f.Total_length() != GK_localVolume && dims == 4) errorQuda("Cannot do a 4D FT on a 3D field\n");
  checkAllocation(f.Field_length());
}
