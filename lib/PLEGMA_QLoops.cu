#include <PLEGMA_QLoops.h>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_contractG5_bilinear.cuh>
using namespace plegma;
  
template<typename Float>
PLEGMA_QLoops<Float>::PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag, bool isOneD):
  PLEGMA_Field<Float>(alloc_flag, QLOOPS),isOneD(isOneD){
  cudaMallocHost((void**)&h_loc, this->Bytes_total());
  checkCudaError();
  memset(h_loc,0,this->Bytes_total());
  for(int idir = 0 ; idir < N_DIMS ; idir++){
    h_oneD[idir] = NULL;
    h_oneDC[idir] = NULL;
  }
  if(isOneD){
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      cudaMallocHost((void**)&h_oneD[idir], this->Bytes_total());
      cudaMallocHost((void**)&h_oneDC[idir], this->Bytes_total());
      checkCudaError();
      memset(h_oneD[idir],0,this->Bytes_total());
      memset(h_oneDC[idir],0,this->Bytes_total());
    }
  }
}

template<typename Float>
PLEGMA_QLoops<Float>::~PLEGMA_QLoops(){
  cudaFreeHost(h_loc);
  if(isOneD)
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      cudaFreeHost(h_oneDC[idir]);
      cudaFreeHost(h_oneD[idir]);
    }
}

template<typename Float>
void PLEGMA_QLoops<Float>::contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r, ACCUM_TYPE acc_type){
  Float accum_sign;
  switch (acc_type){
  case(ACC_ZERO):
    this->zero_device();
    accum_sign = +1;
    break;
  case(ACC_PLUS):
    accum_sign = +1;
    break;
  case(ACC_MINUS):
    accum_sign = -1;
    break;
  }
  vectorTex<Float> vtex_l, vtex_r;
  vtex_l.tex = x_l.createTexObject();
  vtex_r.tex = x_r.createTexObject();
  contractG5_bilinear(*this, vtex_l, vtex_r, accum_sign);
  x_l.destroyTexObject(vtex_l.tex);
  x_r.destroyTexObject(vtex_r.tex);
  checkCudaError();
}

template<typename Float>
void PLEGMA_QLoops<Float>::contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r){
  contractG5(x_l,x_r,ACC_ZERO);
}

template<typename Float>
void PLEGMA_QLoops<Float>::oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
					Float val , bool accum ){
  if(isOneD) errorQuda("This function cannot do the oneD");
  int NN = (this->Field_length()) * (this->Total_length());
  Float valsP[] = {val,0.};
  //local contraction
  contractG5(x_l, x_r); 
  this->unload();
  if(accum) cBLAS::axpy<Float>(NN,valsP, this->H_elem(), h_loc);
  else{
    memcpy(h_loc, this->H_elem(), this->Bytes_total());
    cBLAS::scal<Float>(NN,valsP[0],h_loc);
  }
}

template<typename Float>
void PLEGMA_QLoops<Float>::oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
					PLEGMA_Vector<Float> &tmp, PLEGMA_Gauge<Float> &gauge, Float val , bool accum ){
  if(!isOneD) errorQuda("This function is called to do also the oneD");
  int NN = (this->Field_length()) * (this->Total_length());
  Float valsP[] = {val,0.};
  Float valsM[] = {-val,0.};
  // local contraction
  contractG5(x_l, x_r);
  this->unload();
  if(accum) cBLAS::axpy<Float>(NN,valsP, this->H_elem(), h_loc);
  else{
    memcpy(h_loc, this->H_elem(), this->Bytes_total());
    cBLAS::scal<Float>(NN,valsP[0],h_loc);
  }
  
  x_l.communicateGhost();
  x_r.communicateGhost();
  for(int mu=0; mu<4; mu++)
    {
      tmp.covD(x_r,gauge,mu);
      contractG5(x_l, tmp);      // Term 0
      tmp.covD(x_l,gauge,mu+4);
      contractG5(tmp, x_r, ACC_PLUS); //Term 0 + Term 3
      this->unload();
      if(accum) cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneD[mu]);
      else{
	memcpy(h_oneD[mu], this->H_elem(), this->Bytes_total());
	cBLAS::scal<Float>(NN,valsP[0],h_oneD[mu]);
      }
      memcpy(h_oneDC[mu], h_oneD[mu], this->Bytes_total());
      
      tmp.covD(x_l,gauge,mu);
      contractG5(tmp,x_r); // Term 2
      tmp.covD(x_r,gauge,mu+4);
      contractG5(x_l,tmp, ACC_PLUS); // Term2 + Term1
      this->unload();
      cBLAS::axpy(NN,valsM, (Float*) this->H_elem(), (Float*) h_oneD[mu]); // (0+3-(1+2))
      cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneDC[mu]); // (0+3+(1+2))
    }
}

template<typename Float>
void PLEGMA_QLoops<Float>::write_ASCII(std::string filename_local){
  FILE *ptr_out = NULL;
  ptr_out = fopen(filename_local.c_str(),"w");
  if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
  for(int i =0 ; i < this->Field_length() * this->Total_length(); i++)
    fprintf(ptr_out,"%+e %+e\n", h_loc[i*2], h_loc[i*2+1] );
  fclose(ptr_out);
}

template<typename Float>
void PLEGMA_QLoops<Float>::write_ASCII(std::string filename_local, std::string filename_oneD, std::string filename_oneDC){
  // just for crosschecking improve in the future
  write_ASCII(filename_local);
  if(!isOneD) errorQuda("Cannot write oneD files because oneD is not enabled");

  FILE *ptr_out_oneD = NULL;
  ptr_out_oneD = fopen(filename_oneD.c_str(),"w");
  if(ptr_out_oneD == NULL) errorQuda("Error opening file for writing\n");
  for(int mu = 0; mu < N_DIMS; mu++)
    for(int i =0 ; i < this->Field_length() * this->Total_length(); i++)
      fprintf(ptr_out_oneD,"%+e %+e\n", h_oneD[mu][i*2], h_oneD[mu][i*2+1] );
  fclose(ptr_out_oneD);

  FILE *ptr_out_oneDC = NULL;
  ptr_out_oneDC = fopen(filename_oneDC.c_str(),"w");
  if(ptr_out_oneDC == NULL) errorQuda("Error opening file for writing\n");
  for(int mu = 0; mu < N_DIMS; mu++)
    for(int i =0 ; i < this->Field_length() * this->Total_length(); i++)
      fprintf(ptr_out_oneDC,"%+e %+e\n", h_oneDC[mu][i*2], h_oneDC[mu][i*2+1] );
  fclose(ptr_out_oneDC);

}

template<typename Float>
void PLEGMA_QLoops<Float>::load(Float* h_ptr){
  cudaMemcpy(this->D_elem(), h_ptr, this->Bytes_total(), cudaMemcpyHostToDevice );
}


template class PLEGMA_QLoops<float>;
template class PLEGMA_QLoops<double>;
