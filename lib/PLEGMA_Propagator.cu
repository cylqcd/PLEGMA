#include <PLEGMA_Propagator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_propagator_utils.cuh> 
using namespace plegma;

//-------------------------------//
// class PLEGMA_Propagator //
//-------------------------------//

template<typename Float>
PLEGMA_Propagator<Float>::
PLEGMA_Propagator(ALLOCATION_FLAG alloc_flag): 
  PLEGMA_Field<Float>(alloc_flag, PROPAGATOR){;}

template <typename Float>
void PLEGMA_Propagator<Float>::
absorbVectorToHost(PLEGMA_Vector<Float> &vec, int nu, int c2){
  Float *pointProp_host;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointProp_host = (PLEGMA_Field<Float>::h_elem + 
			mu*N_SPINS*N_COLS*N_COLS*GK_localVolume*2 + 
			nu*N_COLS*N_COLS*GK_localVolume*2 + 
			c1*N_COLS*GK_localVolume*2 + 
			c2*GK_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*N_COLS*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_host,pointVec_dev,GK_localVolume*2*sizeof(Float),cudaMemcpyDeviceToHost); 
    }
  checkCudaError();
}
 
template <typename Float>
void PLEGMA_Propagator<Float>::absorbVectorToDevice(PLEGMA_Vector<Float> &vec, int nu, int c2){
  Float *pointProp_dev;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointProp_dev = (PLEGMA_Field<Float>::d_elem + 
		       mu*N_SPINS*N_COLS*N_COLS*GK_localVolume*2 + 
		       nu*N_COLS*N_COLS*GK_localVolume*2 + 
		       c1*N_COLS*GK_localVolume*2 + 
		       c2*GK_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*N_COLS*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_dev,pointVec_dev,GK_localVolume*2*sizeof(Float),
		 cudaMemcpyDeviceToDevice); 
    }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Propagator<Float>::applyBoundaries_device(int t0){
  apply_boundaries(PLEGMA_Field<Float>::d_elem, t0);
}

template<typename Float>
void PLEGMA_Propagator<Float>::rotateToPhysicalBase_device(int sign){
  if( (sign != +1) && (sign != -1) ) errorQuda("The sign can be only +-1\n");
  rotateToPhysicalBase(PLEGMA_Field<Float>::d_elem, sign);
}

//PLEGMA: DMH Rewrote some parts of this function to conform to new
// QUDA standards. Eg, assigning vaules to complex variable:
// var.real() = 1.0; is changed to var.real(1.0);
template <typename Float>
void PLEGMA_Propagator<Float>::rotateToPhysicalBase_host(int sign_int){
  if( (sign_int != +1) && (sign_int != -1) ) 
    errorQuda("The sign can be only +-1\n");
  
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

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	      
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    //P[mu][nu].real() = PLEGMA_Field<Float>::h_elem[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + nu*N_COLS*N_COLS*GK_localVolume + c1*N_COLS*GK_localVolume + c2*GK_localVolume + iv)*2 + 0];
	    //P[mu][nu].imag() = PLEGMA_Field<Float>::h_elem[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + nu*N_COLS*N_COLS*GK_localVolume + c1*N_COLS*GK_localVolume + c2*GK_localVolume + iv)*2 + 1]
	    P[mu][nu].real(PLEGMA_Field<Float>::h_elem[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + 
				       nu*N_COLS*N_COLS*GK_localVolume + 
				       c1*N_COLS*GK_localVolume + 
				       c2*GK_localVolume + iv)*2 + 0]);
	    P[mu][nu].imag(PLEGMA_Field<Float>::h_elem[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + 
				       nu*N_COLS*N_COLS*GK_localVolume + 
				       c1*N_COLS*GK_localVolume + 
				       c2*GK_localVolume + iv)*2 + 1]);
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
	    PLEGMA_Field<Float>::h_elem[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + 
			nu*N_COLS*N_COLS*GK_localVolume + 
			c1*N_COLS*GK_localVolume + 
			c2*GK_localVolume + iv)*2 + 0] = PT[mu][nu].real();
	    PLEGMA_Field<Float>::h_elem[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + 
			nu*N_COLS*N_COLS*GK_localVolume + 
			c1*N_COLS*GK_localVolume + 
			c2*GK_localVolume + iv)*2 + 1] = PT[mu][nu].imag();
	  }
      }
}


template<typename Float>
void  PLEGMA_Propagator<Float>::conjugate(){
  conjugate_propagator(PLEGMA_Field<Float>::d_elem);
}

template<typename Float>
void  PLEGMA_Propagator<Float>::apply_gamma5(){
  apply_gamma5_propagator(PLEGMA_Field<Float>::d_elem);
}

//----------------------------------//
// class PLEGMA_ Propagator3D //
//----------------------------------//

template<typename Float>
PLEGMA_Propagator3D<Float>::
PLEGMA_Propagator3D(ALLOCATION_FLAG alloc_flag): 
  PLEGMA_Field<Float>(alloc_flag, PROPAGATOR3D){
  if(alloc_flag != BOTH)
    errorQuda("Propagator3D class is only implemented to allocate memory for both\n");
}

template<typename Float>
void PLEGMA_Propagator3D<Float>::
absorbTimeSliceFromHost(PLEGMA_Propagator<Float> &prop, 
			int timeslice){
  int V3 = GK_localVolume/GK_localL[3];
  
  for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
  for(int iv3 = 0 ; iv3 < V3 ; iv3++)
  for(int ipart = 0 ; ipart < 2 ; ipart++)
    PLEGMA_Field<Float>::h_elem[ (mu*N_SPINS*N_COLS*N_COLS*V3 + 
		 nu*N_COLS*N_COLS*V3 + 
		 c1*N_COLS*V3 + 
		 c2*V3 + iv3)*2 + ipart] = 
      prop.H_elem()[(mu*N_SPINS*N_COLS*N_COLS*GK_localVolume + 
		     nu*N_COLS*N_COLS*GK_localVolume + 
		     c1*N_COLS*GK_localVolume + 
		     c2*GK_localVolume + 
		     timeslice*V3 + iv3)*2 + ipart];
  
  cudaMemcpy(PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::h_elem,
	     N_SPINS*N_SPINS*N_COLS*N_COLS*V3*2*sizeof(Float),
	     cudaMemcpyHostToDevice);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Propagator3D<Float>::
absorbTimeSlice(PLEGMA_Propagator<Float> &prop, int timeslice){
  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;

  for(int mu=0; mu<4; mu++)
    for(int nu=0; nu<4; nu++)
      for(int c1=0; c1<3; c1++)
	for(int c2=0; c2<3; c2++){
	  pointer_dst = (PLEGMA_Field<Float>::d_elem + mu*4*3*3*V3*2 + nu*3*3*V3*2 + 
			 c1*3*V3*2 + c2*V3*2);
	  pointer_src = (prop.D_elem() + mu*4*3*3*GK_localVolume*2 + 
			 nu*3*3*GK_localVolume*2 + c1*3*GK_localVolume*2 + 
			 c2*GK_localVolume*2 + timeslice*V3*2);
	  cudaMemcpy(pointer_dst, pointer_src, V3*2*sizeof(Float), 
		     cudaMemcpyDeviceToDevice);
	}
  checkCudaError();
  pointer_src = NULL;
  pointer_dst = NULL;
}

template<typename Float>
void PLEGMA_Propagator3D<Float>::
absorbVectorTimeSlice(PLEGMA_Vector<Float> &vec, 
		      int timeslice, int nu, int c2){
  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (PLEGMA_Field<Float>::d_elem + mu*4*3*3*V3*2 + nu*3*3*V3*2 + 
		     c1*3*V3*2 + c2*V3*2);
      pointer_src = (vec.D_elem() + mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + timeslice*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), 
		 cudaMemcpyDeviceToDevice);
    }
}

template<typename Float>
void PLEGMA_Propagator3D<Float>::broadcast(int tsink){
  cudaMemcpy(PLEGMA_Field<Float>::h_elem , PLEGMA_Field<Float>::d_elem , PLEGMA_Field<Float>::bytes_total_length , 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
  comm_barrier();
  int bcastRank = tsink/GK_localL[3];
  int V3 = GK_localVolume/GK_localL[3];
  if( typeid(Float) == typeid(float) ){
    int error = MPI_Bcast(PLEGMA_Field<Float>::h_elem , 4*4*3*3*V3*2 , MPI_FLOAT , 
			  bcastRank , GK_timeComm );
    if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");
  }
  else if( typeid(Float) == typeid(double) ){
    int error = MPI_Bcast(PLEGMA_Field<Float>::h_elem , 4*4*3*3*V3*2 , MPI_DOUBLE , 
			  bcastRank , GK_timeComm );
    if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");    
  }
  cudaMemcpy(PLEGMA_Field<Float>::d_elem , PLEGMA_Field<Float>::h_elem , PLEGMA_Field<Float>::bytes_total_length, 
	     cudaMemcpyHostToDevice);
  checkCudaError();
}

template  class PLEGMA_Propagator<double>;
template  class PLEGMA_Propagator3D<double>;
template  class PLEGMA_Propagator<float>;
template  class PLEGMA_Propagator3D<float>;
