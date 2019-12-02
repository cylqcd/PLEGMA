#include <PLEGMA_Propagator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_propagator_utils.cuh> 
using namespace plegma;

//-------------------------------//
// class PLEGMA_Propagator //
//-------------------------------//

template<typename Float>
PLEGMA_Propagator<Float>::PLEGMA_Propagator(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag): 
  PLEGMA_Field<Float>(alloc_flag, PROPAGATOR, ghost_flag){;}

template <typename Float>
void PLEGMA_Propagator<Float>::
absorbVectorToHost(PLEGMA_Vector<Float> &vec, int nu, int c2){
  Float *pointProp_host;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointProp_host = (this->h_elem + 
			mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume*2 + 
			nu*N_COLS*N_COLS*HGC_localVolume*2 + 
			c1*N_COLS*HGC_localVolume*2 + 
			c2*HGC_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*N_COLS*HGC_localVolume*2 + c1*HGC_localVolume*2;
      cudaMemcpy(pointProp_host,pointVec_dev,HGC_localVolume*2*sizeof(Float),cudaMemcpyDeviceToHost); 
    }
  checkCudaError();
}

// Prop4D <- Vec4D
template <typename Float>
void PLEGMA_Propagator<Float>::absorb(PLEGMA_Vector<Float> &vec, int nu, int c2){
  Float *pointProp_dev;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointProp_dev = (this->d_elem + 
		       mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume*2 + 
		       nu*N_COLS*N_COLS*HGC_localVolume*2 + 
		       c1*N_COLS*HGC_localVolume*2 + 
		       c2*HGC_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*N_COLS*HGC_localVolume*2 + c1*HGC_localVolume*2;
      cudaMemcpy(pointProp_dev,pointVec_dev,HGC_localVolume*2*sizeof(Float),
		 cudaMemcpyDeviceToDevice); 
    }
  checkCudaError();
}

// Prop4D <- Vec4D (it)
template<typename Float> 
void PLEGMA_Propagator<Float>::absorb(PLEGMA_Vector<Float> &vec, int global_it, int nu, int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      cudaMemset(this->d_elem + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2, 0, V4*2*sizeof(Float));
      if(is_myIt){
	pointer_dst = (this->d_elem + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2 + my_it*V3*2);
	pointer_src = (vec.D_elem() + mu*N_COLS*V4*2 + c1*V4*2 + my_it*V3*2);
	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
    }
  comm_barrier();
  checkCudaError();
}

//Prop4D <- Vec3D
template<typename Float> 
void PLEGMA_Propagator<Float>::absorb(PLEGMA_Vector3D<Float> &vec, int global_it, int nu, int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      cudaMemset(this->d_elem + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2, 0, V4*2*sizeof(Float));
      if(is_myIt){
	pointer_dst = (this->d_elem + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2 + my_it*V3*2);
	pointer_src = (vec.D_elem() + mu*N_COLS*V3*2 + c1*V3*2);
	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
    }
  comm_barrier();
  checkCudaError();
}

template<typename Float>
void PLEGMA_Propagator<Float>::applyBoundaries_device(int t0){
  apply_boundaries(this->d_elem, t0);
}

template<typename Float>
void PLEGMA_Propagator<Float>::rotateToPhysicalBase_device(int sign){
  if( (sign != +1) && (sign != -1) ) PLEGMA_error("The sign can be only +-1\n");
  rotateToPhysicalBase(this->d_elem, sign);
}

//PLEGMA: DMH Rewrote some parts of this function to conform to new
// QUDA standards. Eg, assigning vaules to complex variable:
// var.real() = 1.0; is changed to var.real(1.0);
template <typename Float>
void PLEGMA_Propagator<Float>::rotateToPhysicalBase_host(int sign_int){
  if( (sign_int != +1) && (sign_int != -1) ) 
    PLEGMA_error("The sign can be only +-1\n");
  
  std::complex<Float> sign;
  sign.real(1.0*sign_int);
  sign.imag(0.0);

  std::complex<Float> coeff;
  coeff.real(0.5);
  coeff.imag(0.0);
  
  std::complex<Float> P[4][4];
  std::complex<Float> PT[4][4];
  std::complex<Float> imag_unit;
  //imag_unit.real() = 0.;
  //imag_unit.imag() = 1.;
  imag_unit.real(0.0);
  imag_unit.imag(1.0);

  for(int iv = 0 ; iv < HGC_localVolume ; iv++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	      
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    //P[mu][nu].real() = this->h_elem[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + nu*N_COLS*N_COLS*HGC_localVolume + c1*N_COLS*HGC_localVolume + c2*HGC_localVolume + iv)*2 + 0];
	    //P[mu][nu].imag() = this->h_elem[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + nu*N_COLS*N_COLS*HGC_localVolume + c1*N_COLS*HGC_localVolume + c2*HGC_localVolume + iv)*2 + 1]
	    P[mu][nu].real(this->h_elem[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + 
				       nu*N_COLS*N_COLS*HGC_localVolume + 
				       c1*N_COLS*HGC_localVolume + 
				       c2*HGC_localVolume + iv)*2 + 0]);
	    P[mu][nu].imag(this->h_elem[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + 
				       nu*N_COLS*N_COLS*HGC_localVolume + 
				       c1*N_COLS*HGC_localVolume + 
				       c2*HGC_localVolume + iv)*2 + 1]);
	  }
	
	PT[0][0] = coeff * (P[0][0] + sign * ( imag_unit * P[0][2] ) + sign * ( imag_unit * P[2][0] ) - P[2][2]);
	PT[0][1] = coeff * (P[0][1] + sign * ( imag_unit * P[0][3] ) + sign * ( imag_unit * P[2][1] ) - P[2][3]);
	PT[0][2] = coeff * (sign * ( imag_unit * P[0][0] ) + P[0][2] - P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[0][3] = coeff * (sign * ( imag_unit * P[0][1] ) + P[0][3] - P[2][1] + sign * ( imag_unit * P[2][3] ));
	
	PT[1][0] = coeff * (P[1][0] + sign * ( imag_unit * P[1][2] ) + sign * ( imag_unit * P[3][0] ) - P[3][2]);
	PT[1][1] = coeff * (P[1][1] + sign * ( imag_unit * P[1][3] ) + sign * ( imag_unit * P[3][1] ) - P[3][3]);
	PT[1][2] = coeff * (sign * ( imag_unit * P[1][0] ) + P[1][2] - P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[1][3] = coeff * (sign * ( imag_unit * P[1][1] ) + P[1][3] - P[3][1] + sign * ( imag_unit * P[3][3] ));
	
	PT[2][0] = coeff * (sign * ( imag_unit * P[0][0] ) - P[0][2] + P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[2][1] = coeff * (sign * ( imag_unit * P[0][1] ) - P[0][3] + P[2][1] + sign * ( imag_unit * P[2][3] ));
	PT[2][2] = coeff * (sign * ( imag_unit * P[0][2] ) - P[0][0] + sign * ( imag_unit * P[2][0] ) + P[2][2]);
	PT[2][3] = coeff * (sign * ( imag_unit * P[0][3] ) - P[0][1] + sign * ( imag_unit * P[2][1] ) + P[2][3]);

	PT[3][0] = coeff * (sign * ( imag_unit * P[1][0] ) - P[1][2] + P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[3][1] = coeff * (sign * ( imag_unit * P[1][1] ) - P[1][3] + P[3][1] + sign * ( imag_unit * P[3][3] ));
	PT[3][2] = coeff * (sign * ( imag_unit * P[1][2] ) - P[1][0] + sign * ( imag_unit * P[3][0] ) + P[3][2]);
	PT[3][3] = coeff * (sign * ( imag_unit * P[1][3] ) - P[1][1] + sign * ( imag_unit * P[3][1] ) + P[3][3]);

	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    this->h_elem[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + 
			nu*N_COLS*N_COLS*HGC_localVolume + 
			c1*N_COLS*HGC_localVolume + 
			c2*HGC_localVolume + iv)*2 + 0] = PT[mu][nu].real();
	    this->h_elem[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + 
			nu*N_COLS*N_COLS*HGC_localVolume + 
			c1*N_COLS*HGC_localVolume + 
			c2*HGC_localVolume + iv)*2 + 1] = PT[mu][nu].imag();
	  }
      }
}

template<typename Float>
void  PLEGMA_Propagator<Float>::apply_gamma(GAMMAS gMat,LEFTRIGHT LR){
  apply_gamma_prop(LR,*this,gMat);
}

template<typename Float>
void  PLEGMA_Propagator<Float>::apply_gamma5(){
  apply_gamma5_propagator(*this);
}

//----------------------------------//
// class PLEGMA_ Propagator3D //
//----------------------------------//

template<typename Float>
void PLEGMA_Propagator3D<Float>::
absorbTimeSliceFromHost(PLEGMA_Propagator<Float> &prop, 
			int timeslice){
  int V3 = HGC_localVolume/HGC_localL[3];
  
  for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
  for(int iv3 = 0 ; iv3 < V3 ; iv3++)
  for(int ipart = 0 ; ipart < 2 ; ipart++)
    this->h_elem[ (mu*N_SPINS*N_COLS*N_COLS*V3 + 
		 nu*N_COLS*N_COLS*V3 + 
		 c1*N_COLS*V3 + 
		 c2*V3 + iv3)*2 + ipart] = 
      prop.H_elem()[(mu*N_SPINS*N_COLS*N_COLS*HGC_localVolume + 
		     nu*N_COLS*N_COLS*HGC_localVolume + 
		     c1*N_COLS*HGC_localVolume + 
		     c2*HGC_localVolume + 
		     timeslice*V3 + iv3)*2 + ipart];
  
  cudaMemcpy(this->d_elem,this->h_elem,
	     N_SPINS*N_SPINS*N_COLS*N_COLS*V3*2*sizeof(Float),
	     cudaMemcpyHostToDevice);
  checkCudaError();
}

//Prop3D <- Vec4D
template<typename Float> 
void PLEGMA_Propagator3D<Float>::absorb(PLEGMA_Vector<Float> &vec, int global_it, int nu, int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  this->activeTimeSlice = is_myIt;
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointer_dst = (this->d_elem + mu*N_SPINS*N_COLS*N_COLS*V3*2 + nu*N_COLS*N_COLS*V3*2 + c1*N_COLS*V3*2 + c2*V3*2);
      if(is_myIt){
	pointer_src = (vec.D_elem() + mu*N_COLS*V4*2 + c1*V4*2 + my_it*V3*2);
	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
      else
	cudaMemset(pointer_dst, 0, V3*2 * sizeof(Float));
    }
  checkCudaError();
}

//Prop3D <- Vec3D
template<typename Float> 
void PLEGMA_Propagator3D<Float>::absorb(PLEGMA_Vector3D<Float> &vec, int nu, int c2){
  this->activeTimeSlice = vec.includesActiveTimeSlice();
  int V3 = HGC_localVolume/HGC_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointer_dst = (this->d_elem + mu*N_SPINS*N_COLS*N_COLS*V3*2 + nu*N_COLS*N_COLS*V3*2 + c1*N_COLS*V3*2 + c2*V3*2);
      pointer_src = (vec.D_elem() + mu*N_COLS*V3*2 + c1*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
    }
  checkCudaError();
}

template  class PLEGMA_Propagator<double>;
template  class PLEGMA_Propagator3D<double>;
template  class PLEGMA_Propagator<float>;
template  class PLEGMA_Propagator3D<float>;
