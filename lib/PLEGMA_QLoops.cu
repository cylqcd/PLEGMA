#include <PLEGMA_QLoops.h>
#include <PLEGMA_BLAS.h>
using namespace plegma;

//!!!!!!!!! TODO: Allow for all classes the option to do pagelocking for host memory
//!!!!!!!!! TODO: host ext ghost is a good idea to be pagelocked
void contract(const quda::cudaColorSpinorField x, const quda::cudaColorSpinorField y, void *ctrn, const QudaContractType cType);
  
template<typename Float>
PLEGMA_QLoops<Float>::PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag, quda::GaugeCovDev *cov):
  PLEGMA_Field<Float>(alloc_flag, QLOOPS),isOneD(false),cov(cov){
  h_loc = (Float*) malloc(this->Bytes_total());
  if(h_loc == NULL) errorQuda("Error with allocation host memory");
  memset(h_loc,0,this->Bytes_total());
  for(int idir = 0 ; idir < N_DIMS ; idir++){
    h_oneD[idir] = NULL;
    h_oneDC[idir] = NULL;
  }
  if(cov != NULL){
    isOneD = true;
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      h_oneD[idir] = (Float*) malloc(this->Bytes_total());
      h_oneDC[idir] = (Float*) malloc(this->Bytes_total());
      if(h_oneD[idir] == NULL || h_oneDC[idir] == NULL) errorQuda("Error with allocation host memory");
      memset(h_oneD[idir],0,this->Bytes_total());
      memset(h_oneDC[idir],0,this->Bytes_total());
    }
  }
}

template<typename Float>
PLEGMA_QLoops<Float>::~PLEGMA_QLoops(){
  free(h_loc);
  if(isOneD)
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      free(h_oneDC);
      free(h_oneD);
    }
}


template<typename Float>
void PLEGMA_QLoops<Float>::oneEnd_trick(quda::cudaColorSpinorField &x_l, quda::cudaColorSpinorField &x_r,
				   quda::cudaColorSpinorField &tmp, Float val , bool accum ){
  if(x_l.Precision() != this->Precision()) errorQuda("Precision between QUDA and PLEGMA fields do not much");
  
  int NN = (this->Field_length()) * (this->Total_length());
  Float valsP[] = {val,0.};
  Float valsM[] = {-val,0.};
  // local contraction
  contract(x_l, x_r, this->D_elem(), QUDA_CONTRACT_GAMMA5);
  this->unload();
  if(accum) axpy<Float>(NN,valsP, this->H_elem(), h_loc);
  else memcpy(h_loc, this->H_elem(), this->Bytes_total());

  if(isOneD)
    for(int mu=0; mu<4; mu++)
      {
	cov->MCD(tmp,x_r,mu);
	contract(x_l, tmp, this->D_elem(), QUDA_CONTRACT_GAMMA5);      // Term 0
	cov->MCD(tmp,x_l,mu+4);
	contract(tmp, x_r, this->D_elem(), QUDA_CONTRACT_GAMMA5_PLUS); //Term 0 + Term 3
	this->unload();
	if(accum) axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneD[mu]);
	else memcpy(h_oneD[mu], this->H_elem(), this->Bytes_total());
	memcpy(h_oneDC[mu], h_oneD[mu], this->Bytes_total());

	cov->MCD(tmp,x_l,mu);
	contract(tmp,x_r, this->D_elem(), QUDA_CONTRACT_GAMMA5); // Term 2
	cov->MCD(tmp,x_r,mu+4);
	contract(x_l,tmp, this->D_elem(), QUDA_CONTRACT_GAMMA5_PLUS); // Term2 + Term1
	this->unload();
	axpy(NN,valsM, (Float*) this->H_elem(), (Float*) h_oneD[mu]); // (0+3-(1+2))
	axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneDC[mu]); // (0+3+(1+2))
      }
}

template class PLEGMA_QLoops<float>;
template class PLEGMA_QLoops<double>;
