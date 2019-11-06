#include <PLEGMA_Field.h>
#include <PLEGMA_Tensor.h>

using namespace plegma;

template<typename Float> 
PLEGMA_FT<Float>::PLEGMA_FT(int Q2_max, int D3D4, bool accum):
Q2_max(Q2_max), isAllocated(false), dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(0), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  dimT = (dims == 3) ? HGC_localL[3] : 1; // when apply, if a 3D field set dimT=1 even if dims=3
  if(Q2_max < 0) PLEGMA_error("The maximum number of Q2 cannot be negative\n");
  createMom();
  texMomList.Nmoms=0;
}

template<typename Float>
template<typename T>
PLEGMA_FT<Float>::PLEGMA_FT(std::vector<T> mom, int D3D4, bool accum):
  isAllocated(false), dof(0), h_elem(nullptr), sizeN(0), dims(D3D4), dimT(0), accum(accum){
  if(dims!= 3 && dims !=4) PLEGMA_error("This class transforms only 3 and 4 dimensions\n");
  dimT = (dims == 3) ? HGC_localL[3] : 1; // when apply, if a 3D field set dimT=1 even if dims=3
  if(mom.size() != dims) PLEGMA_error("The size of the momentum vector does not match the dimensionality of FT");
  VFloat momF(mom.begin(),mom.end());
  momList.push_back(momF);
  texMomList.Nmoms=0;
}


template<typename Float>
PLEGMA_FT<Float>::~PLEGMA_FT(){
  if(isAllocated) hostFree(h_elem, sizeN*sizeof(Float));
  if(texMomList.Nmoms>0)
    texMomList.free();
}

template<typename Float>
void PLEGMA_FT<Float>::zero(){
  if(isAllocated) memset(h_elem, 0, Nmoms()*dimT*dof*2*sizeof(Float));
}
