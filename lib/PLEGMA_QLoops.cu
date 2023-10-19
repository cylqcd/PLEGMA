#include <PLEGMA_QLoops.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_FT.h>
#include <PLEGMA_BLAS.h>
#include <kernels/PLEGMA_contractG5_bilinear.cuh>
#include <functional>
using namespace plegma;
  
template<typename Float>
PLEGMA_QLoops<Float>::PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag, bool isPinnedHost, bool isOneD, bool isTwoD):
  PLEGMA_Field<Float>(alloc_flag, QLOOPS,ghost_flag,isPinnedHost),isOneD(isOneD),isTwoD(isTwoD){

  std::function<void(Float* (&),size_t)> fHmalloc = isPinnedHost? hostMallocPinned<Float*>  : static_cast<void(*)(Float* (&),size_t)>(hostMalloc<Float*>);  
  fHmalloc(h_loc,this->Bytes_total());
  memset(h_loc,0,this->Bytes_total());
  for(int idir = 0 ; idir < N_DIMS ; idir++){
    h_oneD[idir] = NULL;
    h_oneDC[idir] = NULL;
  }
  if(!isOneD && isTwoD) PLEGMA_error("Asking for two derivatives when one derivative is disabled is not allowed");
  if(isOneD){
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      fHmalloc(h_oneD[idir],this->Bytes_total());
      fHmalloc(h_oneDC[idir],this->Bytes_total());
      memset(h_oneD[idir],0,this->Bytes_total());
      memset(h_oneDC[idir],0,this->Bytes_total());
    }
  }
  if(isTwoD){
    for(int idirs = 0; idirs < N_DIMS*(N_DIMS-1); idirs++){
      fHmalloc(h_twoD[idirs], this->Bytes_total());
      memset(h_twoD[idirs],0,this->Bytes_total());
    }
    for(int i = 0; i < N_DIMS-1; i++)
      for(int j = i+1; j < N_DIMS; j++){
	twoD_index.push_back(std::make_pair(i,j));
	twoD_index.push_back(std::make_pair(j,i));
      }
  }
}

template<typename Float>
PLEGMA_QLoops<Float>::~PLEGMA_QLoops(){
  std::function<void(Float* (&),size_t)> fHfree = this->isPinnedHost ? static_cast<void(*)(Float* (&),size_t)>(hostFreePinned<Float*>)
    : static_cast<void(*)(Float* (&),size_t)>(hostFree<Float*>);
  fHfree(h_loc,this->Bytes_total());
  if(isOneD)
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      fHfree(h_oneDC[idir],this->Bytes_total());
      fHfree(h_oneD[idir],this->Bytes_total());
    }
  if(isTwoD)
    for(int idirs = 0; idirs < N_DIMS*(N_DIMS-1); idirs++){
      fHfree(h_twoD[idirs],this->Bytes_total());
    }
}

template<typename Float>
void PLEGMA_QLoops<Float>::clearAccumBuffs(){
  memset(h_loc,0,this->Bytes_total());
  if(isOneD)
    for(int idir = 0 ; idir < N_DIMS ; idir++){
      memset(h_oneD[idir],0,this->Bytes_total());
      memset(h_oneDC[idir],0,this->Bytes_total());
    }
  if(isTwoD)
    for(int idirs = 0 ; idirs < N_DIMS*(N_DIMS-1); idirs++)
      memset(h_twoD[idirs],0,this->Bytes_total());
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
  assert(this->checkVolume(x_l,x_r));
  auto vtex_l = toTexture<vectorTex>(x_l);
  auto vtex_r = toTexture<vectorTex>(x_r);
  contractG5_bilinear(toField2<generic2>(*this), *vtex_l, *vtex_r, accum_sign);
  checkQudaError();
}

template<typename Float>
void PLEGMA_QLoops<Float>::contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r){
  contractG5(x_l,x_r,ACC_ZERO);
}

template<typename Float>
void PLEGMA_QLoops<Float>::oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
					Float val , bool accum ){
  if(isOneD) PLEGMA_error("This function cannot do the oneD");
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
					PLEGMA_Vector<Float> *tmp[16], PLEGMA_QLoops<Float> *qLtmp,
					PLEGMA_Gauge<Float> &gauge, Float val , bool accum ){
  if(!isOneD && !isTwoD) PLEGMA_error("This function is capable to do the oneD and twoD, if you do not need derivative call the other function");
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

  //Derivatives part
  if(isOneD){
    x_l.communicateGhost();
    x_r.communicateGhost();
    //oneD part
    if(tmp[0] == nullptr) PLEGMA_error("Need to allocate memory for tmp[0] to use in the oneD");
    for(int mu=0; mu<4; mu++)
      {
	tmp[0]->covD(x_r,gauge,mu);
	contractG5(x_l, *(tmp[0]));      // Term 0
	tmp[0]->covD(x_l,gauge,mu+4);
	contractG5(*(tmp[0]), x_r, ACC_PLUS); //Term 0 + Term 3
	this->unload();
	if(accum){
	  cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneD[mu]);
	  cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneDC[mu]);
	}
	else{
	  memcpy(h_oneD[mu], this->H_elem(), this->Bytes_total());
	  cBLAS::scal<Float>(NN,valsP[0],h_oneD[mu]);
	  memcpy(h_oneDC[mu], h_oneD[mu], this->Bytes_total());
	}

      
	tmp[0]->covD(x_l,gauge,mu);
	contractG5(*(tmp[0]),x_r); // Term 2
	tmp[0]->covD(x_r,gauge,mu+4);
	contractG5(x_l,*(tmp[0]), ACC_PLUS); // Term2 + Term1
	this->unload();
	cBLAS::axpy(NN,valsM, (Float*) this->H_elem(), (Float*) h_oneD[mu]); // (0+3-(1+2))
	cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_oneDC[mu]); // (0+3+(1+2))
	// factor of 1/4 should be added later
      }
  }

  if(isTwoD){
    //twoD part
    for(int i = 1 ; i < 16; i++) if(tmp[i] == nullptr) PLEGMA_error("Need to allocate memory for tmp[%d] to use in the twoD",i);
    if(qLtmp == nullptr) PLEGMA_error("Need to allocate memory for qLtmp to use in the twoD");
    int count=0;
    for(int mu = 0 ; mu < 4 ; mu++){
      tmp[mu*2+0]->covD(x_r,gauge,mu);
      tmp[mu*2+1]->covD(x_r,gauge,mu+4);
      tmp[8+mu*2+0]->covD(x_l,gauge,mu);
      tmp[8+mu*2+1]->covD(x_l,gauge,mu+4);
    }
    for(auto munu : twoD_index){
      int mu=std::get<0>(munu), nu=std::get<1>(munu);
      if(mu != 3 && nu != 3){
	contractG5(*(tmp[8+mu*2+1]), *(tmp[nu*2+0]));
	contractG5(*(tmp[8+mu*2+0]), *(tmp[nu*2+1]), ACC_PLUS);
	contractG5(*(tmp[8+mu*2+1]), *(tmp[nu*2+1]), ACC_MINUS);
	contractG5(*(tmp[8+mu*2+0]), *(tmp[nu*2+0]), ACC_MINUS);
	this->unload();
	cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_twoD[count]);
      }
      else if(mu == 3 || nu == 3){
	if(mu == 3){
	  qLtmp->contractG5(*(tmp[8+mu*2+1]), *(tmp[nu*2+0]));
	  qLtmp->contractG5(*(tmp[8+mu*2+1]), *(tmp[nu*2+1]), ACC_MINUS);
	}
	else if(nu == 3){
	  qLtmp->contractG5(*(tmp[8+mu*2+0]), *(tmp[nu*2+1]));
	  qLtmp->contractG5(*(tmp[8+mu*2+1]), *(tmp[nu*2+1]), ACC_MINUS);
	}
	  
	this->shift(*qLtmp,mu==3?mu+4:nu+4);
	this->add(*qLtmp,(std::complex<Float>) {1.,0.});
	this->unload();
	cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_twoD[count]);
	if(mu == 3){
	  qLtmp->contractG5(*(tmp[8+mu*2+0]), *(tmp[nu*2+1]));
	  qLtmp->contractG5(*(tmp[8+mu*2+0]), *(tmp[nu*2+0]), ACC_MINUS);
	}
	else if(nu == 3){
	  qLtmp->contractG5(*(tmp[8+mu*2+1]), *(tmp[nu*2+0]));
	  qLtmp->contractG5(*(tmp[8+mu*2+0]), *(tmp[nu*2+0]), ACC_MINUS);	  
	}
	this->shift(*qLtmp,mu==3?mu:nu);
	this->add(*qLtmp,(std::complex<Float>) {1.,0.});
	this->unload();
	cBLAS::axpy(NN,valsP, (Float*) this->H_elem(), (Float*) h_twoD[count]);
      }
      else{
	PLEGMA_error("Something Fishy is going on here");
      }
      count++;
      // factor of (1/16)*4 or (1/16)*2 should be added later, multiply by 4 since terms from neighboring x are repeated for mu,nu!=3
      // multiply by mu=3 or nu=3 we need to multiply by 2
      // !!!!!!!! note for now this will work only for zero momentum otherwise terms above should be kept separately
      // and fourier transform should be modified to consider the dislocalization terms
    }
  }
}


template<typename Float>
void PLEGMA_QLoops<Float>::oneEnd_trick_wilsonLine(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r, Float val , PLEGMA_Gauge<Float> &gauge,
						   PLEGMA_FT<Float> **FTs){
  if(&x_l == &x_r) PLEGMA_error("The function with Wilson line needs different left from right locations");
  if(isOneD || isTwoD) PLEGMA_error("oneD or twoD cannot be computed with this function");
  if(!x_r.IsAllocHost())PLEGMA_error("You need to allocate also host memory for the x_r");
  x_r.unload();
  PLEGMA_Su3field<Float> su3;
  PLEGMA_Su3field<Float> WL;
  PLEGMA_Su3field<Float> tmp;
  PLEGMA_Vector<Float> *vecExchange = nullptr;
  PLEGMA_Vector<Float> *vecIn = new PLEGMA_Vector<Float>(DEVICE);
  PLEGMA_Vector<Float> *vec_ptr = nullptr;
  PLEGMA_Vector<Float> vecTmp(DEVICE);

  std::complex<Float> cr;
  cr.real(val);
  cr.imag(0.);
  if(!(HGC.totalL[0] == HGC.totalL[1] && HGC.totalL[1] == HGC.totalL[2])) PLEGMA_error("Spatial total volume should be symmetric for this to work");
  int L=HGC.totalL[0];
  if(L%2 != 0) PLEGMA_error("If spatial extent is not multiple of 2 then it will not work");
  int Lo2 = L/2;
  for(int wilsDir = 0 ; wilsDir < 3; wilsDir++){
    su3.absorbDir_device(gauge,wilsDir);
    WL.setUnit((std::vector<int>) {0,4,8});
    vec_ptr = &x_r;
    for(int i = 0 ; i < Lo2;i++){
      vecTmp.mulGV(*vec_ptr,WL);
      contractG5(x_l,vecTmp);
      this->cscale(cr);
      if(!FTs[wilsDir*L+i]->IsAccum()) PLEGMA_error("We need accumulation on here");
      FTs[wilsDir*L+i]->apply(*this,FT_GEMV);
      vecExchange=vecIn; vecIn=vec_ptr; vec_ptr = vecExchange; 
      WL.wilsonLineUpdate(su3,tmp,4+wilsDir);
      vec_ptr->shift(*vecIn,4+wilsDir);
    }

    vec_ptr->load();
    su3.absorbDir_device(gauge,wilsDir);
    WL.setUnit((std::vector<int>) {0,4,8});
    for(int i = 0 ; i < Lo2;i++){
      vecTmp.mulGV(*vec_ptr,WL);
      contractG5(x_l,vecTmp);
      this->cscale(cr);
      if(!FTs[wilsDir*L+i+Lo2]->IsAccum()) PLEGMA_error("We need accumulation on here");
      FTs[wilsDir*L+i+Lo2]->apply(*this,FT_GEMV);
      vecExchange=vecIn; vecIn=vec_ptr; vec_ptr = vecExchange;
      WL.wilsonLineUpdate(su3,tmp,wilsDir);
      vec_ptr->shift(*vecIn,wilsDir);
    }
    vec_ptr->load();
  }
  delete vecIn;
}


template<typename Float>
void PLEGMA_QLoops<Float>::write_ASCII(std::string filename_local){
  FILE *ptr_out = NULL;
  ptr_out = fopen(filename_local.c_str(),"w");
  if(ptr_out == NULL) PLEGMA_error("Error opening file for writing\n");
  for(int i =0 ; i < this->Field_length() * this->Total_length(); i++)
    fprintf(ptr_out,"%+e %+e\n", h_loc[i*2], h_loc[i*2+1] );
  fclose(ptr_out);
}

template<typename Float>
void PLEGMA_QLoops<Float>::write_ASCII(std::string filename_local, std::string filename_oneD, std::string filename_oneDC){
  // just for crosschecking improve in the future
  write_ASCII(filename_local);
  if(!isOneD) PLEGMA_error("Cannot write oneD files because oneD is not enabled");

  FILE *ptr_out_oneD = NULL;
  ptr_out_oneD = fopen(filename_oneD.c_str(),"w");
  if(ptr_out_oneD == NULL) PLEGMA_error("Error opening file for writing\n");
  for(int mu = 0; mu < N_DIMS; mu++)
    for(int i =0 ; i < this->Field_length() * this->Total_length(); i++)
      fprintf(ptr_out_oneD,"%+e %+e\n", h_oneD[mu][i*2], h_oneD[mu][i*2+1] );
  fclose(ptr_out_oneD);

  FILE *ptr_out_oneDC = NULL;
  ptr_out_oneDC = fopen(filename_oneDC.c_str(),"w");
  if(ptr_out_oneDC == NULL) PLEGMA_error("Error opening file for writing\n");
  for(int mu = 0; mu < N_DIMS; mu++)
    for(int i =0 ; i < this->Field_length() * this->Total_length(); i++)
      fprintf(ptr_out_oneDC,"%+e %+e\n", h_oneDC[mu][i*2], h_oneDC[mu][i*2+1] );
  fclose(ptr_out_oneDC);

}

template<typename Float>
void PLEGMA_QLoops<Float>::load(Float* h_ptr){
  qudaMemcpy(this->D_elem(), h_ptr, this->Bytes_total(), qudaMemcpyHostToDevice );
}


template class plegma::PLEGMA_QLoops<float>;
template class plegma::PLEGMA_QLoops<double>;
