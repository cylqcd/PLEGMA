#include <PLEGMA_Vector.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_vector_utils.cuh> 
#include <PLEGMA_gaussian_smearing.cuh> 
#include <PLEGMA_covD.cuh>
using namespace plegma;
using namespace quda;
//---------------------------//
// class PLEGMA_Vector //
//---------------------------//

template<typename Float>
PLEGMA_Vector<Float>::PLEGMA_Vector(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag): 
  PLEGMA_Field<Float>(alloc_flag, VECTOR, ghost_flag){ ; }

template<typename Float>
void PLEGMA_Vector<Float>::gaussianSmearing(PLEGMA_Vector<Float> &vecIn,
					    PLEGMA_Gauge<Float> &gauge,
					    int nsmearGauss, Float alphaGauss, int timeSlice){
  if(vecIn.IsAllocHost()) {
    vecIn.unload(); // backing up the vecIn
  } else {
    PLEGMA_warning("VecIn is not allocated on BOTH; gaussianSmearing will overwrite the device memory.\n");
  }
  
  gaugeTex<Float> texGauge;
  vectorTex<Float> texVecIn, texVecOut;
  texVecOut.tex = this->createTexObject();
  texVecIn.tex = vecIn.createTexObject();
  texGauge.tex = gauge.createTexObject();

  for(int i = 0 ; i < nsmearGauss ; i++){
    if( (i%2) == 0){
      for(int dir=0; dir<N_DIMS-1; dir++) {
	if(i==0) gauge.communicateSideGhost(dir, START);
	vecIn.communicateSideGhost(dir, START);
      }
      gaussian_smearing_no_ghost(this->D_elem(),texVecIn,texGauge, alphaGauss);
      for(int dir=0; dir<N_DIMS-1; dir++) {
	if(i==0) gauge.communicateSideGhost(dir, FINISH);
	vecIn.communicateSideGhost(dir, FINISH);
      }
      gaussian_smearing_only_ghost(this->D_elem(),texVecIn,texGauge, alphaGauss);
    }
    else{
      for(int dir=0; dir<N_DIMS-1; dir++) {
	this->communicateSideGhost(dir, START);
      }
      gaussian_smearing_no_ghost(vecIn.D_elem(), texVecOut, texGauge, alphaGauss);
      for(int dir=0; dir<N_DIMS-1; dir++) {
	this->communicateSideGhost(dir, FINISH);
      }
      gaussian_smearing_only_ghost(vecIn.D_elem(), texVecOut, texGauge, alphaGauss);
    }
  }
  if( (nsmearGauss%2) == 0)
    cudaMemcpy(this->D_elem(),vecIn.D_elem(),
	       this->bytes_total_length,cudaMemcpyDeviceToDevice);
  
  this->destroyTexObject(texVecOut.tex);
  vecIn.destroyTexObject(texVecIn.tex);
  gauge.destroyTexObject(texGauge.tex);
  checkCudaError();

  if(vecIn.IsAllocHost()) {
    vecIn.load(); // restoring vecIn
  }
}


template<typename Float>
void PLEGMA_Vector<Float>::copyToQUDA(ColorSpinorField *qudaVector, bool isEv){
  copy_to_QUDA(this->d_elem, *qudaVector, isEv);
}

template<typename Float>
void PLEGMA_Vector<Float>::copyFromQUDA(ColorSpinorField *qudaVector, bool isEv){
  copy_from_QUDA(this->d_elem, *qudaVector, isEv);
}

template<typename Float>
void  PLEGMA_Vector<Float>::scaleVector(Float a){
  scale_vector(a,this->d_elem);
}

template<typename Float>
void  PLEGMA_Vector<Float>::conjugate(){
  conjugate_vector(this->d_elem);
}

template<typename Float>
void  PLEGMA_Vector<Float>::apply_gamma5(){
  apply_gamma5_vector(this->d_elem);
}


template<typename Float> 
void  PLEGMA_Vector<Float>::apply_gamma(GAMMAS gMat,LEFTRIGHT LR){
  apply_gamma_vector(LR,this->d_elem,gMat);
}

template<typename Float>
void PLEGMA_Vector<Float>::norm2Host(){
  Float res = 0.;
  Float globalRes;

  for(int i = 0 ; i < N_SPINS*N_COLS*HGC_localVolume ; i++){
    res += this->h_elem[i*2 + 0]*this->h_elem[i*2 + 0] + this->h_elem[i*2 + 1]*this->h_elem[i*2 + 1];
  }

  int rc = MPI_Allreduce(&res, &globalRes , 1, sizeof(Float)==4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, HGC_fullComm);
  if( rc != MPI_SUCCESS ) PLEGMA_error("Error in MPI reduction for plaquette");
  PLEGMA_printf("Vector norm2 is %e\n",globalRes);
}

// vec4D <- Prop3D
template<typename Float>
void PLEGMA_Vector<Float>::absorb(PLEGMA_Propagator3D<Float> &prop, int global_it, int nu , int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      cudaMemset(this->d_elem + mu*N_COLS*V4*2 + c1*V4*2, 0, V4*2*sizeof(Float));
      if(is_myIt){
	pointer_dst = (this->d_elem + mu*N_COLS*V4*2 + c1*V4*2 + my_it*V3*2);
	pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V3*2 + nu*N_COLS*N_COLS*V3*2 + c1*N_COLS*V3*2 + c2*V3*2);
	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
    }
  comm_barrier();
  checkCudaError();
}

// vec4D <- prop4D (it)
template<typename Float>
void PLEGMA_Vector<Float>::absorb(PLEGMA_Propagator<Float> &prop, int global_it, int nu , int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      cudaMemset(this->d_elem + mu*N_COLS*V4*2 + c1*V4*2, 0, V4*2*sizeof(Float));
      if(is_myIt){
	pointer_dst = (this->d_elem + mu*N_COLS*V4*2 +  c1*V4*2 + my_it*V3*2);
       	pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2 + my_it*V3*2);
       	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
    }
  comm_barrier();
  checkCudaError();
}

// vec4D <- prop4D
template<typename Float>
void PLEGMA_Vector<Float>::absorb(PLEGMA_Propagator<Float> &prop, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  int V4 = HGC_localVolume;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointer_dst = (this->d_elem + mu*N_COLS*V4*2 +  c1*V4*2);
      pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2);
      cudaMemcpy(pointer_dst, pointer_src, V4*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
    }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Vector<Float>::dilutespin(PLEGMA_Vector<Float> &vecIn, int spin){
  Float *pointer_src = NULL;
  if(spin >= N_SPINS) PLEGMA_error("The spin index you provided exceed the total spin content\n");
  this->zero_device();
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      if(mu == spin){
        pointer_src = (vecIn.D_elem() + (c1 + mu*N_COLS)*HGC_localVolume*2);
        cudaMemcpy((this->d_elem + ((c1 + mu*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), cudaMemcpyDeviceToDevice); 
      } 
    }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Vector<Float>::dilutecolor(PLEGMA_Vector<Float> &vecIn, int color){
  Float *pointer_src = NULL;
  if(color >= N_COLS) PLEGMA_error("The color index you provided exceed the total color content\n");
  this->zero_device();
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      if(c1 == color){
        pointer_src = (vecIn.D_elem() + (c1 + mu*N_COLS)*HGC_localVolume*2);
        cudaMemcpy((this->d_elem + ((c1 + mu*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), cudaMemcpyDeviceToDevice); 
      } 
    }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Vector<Float>::dilutespincolor(PLEGMA_Vector<Float> &vecIn, int spin, int color){
  Float *pointer_src = NULL;
  if(color >= N_COLS) PLEGMA_error("The color index you provided exceed the total color content\n");
  if(spin >= N_SPINS) PLEGMA_error("The spin index you provided exceed the total spin content\n");
  this->zero_device();
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      if(c1 == color && mu == spin){
        pointer_src = (vecIn.D_elem() + (c1 + mu*N_COLS)*HGC_localVolume*2);
        cudaMemcpy((this->d_elem + ((c1 + mu*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), cudaMemcpyDeviceToDevice); 
      } 
    }
  checkCudaError();
}


template<typename Float>
void PLEGMA_Vector<Float>::pointSource(const site& sourceposition, int spin, int color, ALLOCATION_FLAG where){
  for(int i = 0; i < N_DIMS; i++)
    if(sourceposition[i] >= HGC_totalL[i]) PLEGMA_error("Source position component in dir=%d, is %d >= %d the lattice extent", i, sourceposition[i],HGC_totalL[i]);
  
  this->zero_where(where);
  int my_src[N_DIMS];
  size_t id=0;
  Float temp[1];
  temp[0] = 1.0;

  for(int i = N_DIMS-1; i >= 0; i--) {
    my_src[i] = (sourceposition[i] - comm_coords(HGC_default_topo)[i] * HGC_localL[i]);

    // if out of the local lattice we break
    if((my_src[i]<0) || (my_src[i]>=HGC_localL[i]))
      return;

    id = id * HGC_localL[i] + my_src[i];
  }

  if( where == BOTH ){
    this->h_elem[((spin*N_COLS+color)*HGC_localVolume + id)*2] = 1.0; 
    cudaMemcpy((this->d_elem + ((spin*N_COLS+color)*HGC_localVolume + id)*2), temp,sizeof(Float),
                cudaMemcpyHostToDevice ); 
  }
  else if (where == HOST){
    this->h_elem[((spin*N_COLS+color)*HGC_localVolume + id)*2] = 1.0; 
  }
  else if (where == DEVICE){
    cudaMemcpy((this->d_elem + ((spin*N_COLS+color)*HGC_localVolume + id)*2), temp,sizeof(Float),
                cudaMemcpyHostToDevice ); 
  }
  else{
    PLEGMA_error("Not supported %d\n",where);
  }
}

template<typename Float>
void PLEGMA_Vector<Float>::pointSource(const site& sourceposition, int spin, int color){
  pointSource(sourceposition,spin,color,this->allocation);
}


template<typename Float>
void PLEGMA_Vector<Float>::covD(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gauge, int dirOr){
  // to increase efficiency the communication of the ghost for the the vector should happen before calling this function
  if(dirOr < 0 || dirOr > 7) PLEGMA_error("Wrong direction is given");
  vectorTex<Float> texVecIn;
  texVecIn.tex= vecIn.createTexObject();
  gaugeTex<Float> texGaugeIn;
  texGaugeIn.tex = gauge.createTexObject();
  covD_k<Float,Float,Float>(this->D_elem(), texVecIn, texGaugeIn, dirOr);
  vecIn.destroyTexObject(texVecIn.tex);
  gauge.destroyTexObject(texGaugeIn.tex);
}

template<typename FloatC, typename FloatA>
void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
template<typename Float>
void PLEGMA_Vector<Float>::seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop, WHICHPROJECTOR proj, WHICHPARTICLE particle, int global_it, int c_nu, int c_c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  this->zero_device();
  if(is_myIt){
    genericTex<Float> texProp;
    texProp.tex = prop.createTexObject();
    contractNucleonSeqSource<Float,Float>(*this, texProp, proj, particle, my_it, c_nu, c_c2);
    prop.destroyTexObject(texProp.tex);
  }
  comm_barrier();
}

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
template<typename Float>
void PLEGMA_Vector<Float>::seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop1, PLEGMA_Propagator3D<Float> &prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int global_it, int c_nu, int c_c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  this->zero_device();
  if(is_myIt){
    genericTex<Float> texProp1;
    texProp1.tex = prop1.createTexObject();
    genericTex<Float> texProp2;
    texProp2.tex = prop2.createTexObject();
    contractNucleonSeqSource<Float,Float,Float>(*this, texProp1,texProp2, proj, particle, my_it, c_nu, c_c2);
    prop1.destroyTexObject(texProp1.tex);
    prop2.destroyTexObject(texProp2.tex);
  }
  comm_barrier();
}

template<typename Float>
std::vector<Float> PLEGMA_Vector<Float>::rms(std::vector<int> listR2, int *sourceposition){
  if(listR2.size() <= 0) PLEGMA_error("Provided list of r2 is empty");
  for(int i = 0; i < N_DIMS; i++)
    if(sourceposition[i] >= HGC_totalL[i]) PLEGMA_error("Source position component in dir=%d, is %d >= %d the lattice extent", i, sourceposition[i],HGC_totalL[i]);
  int my_it = sourceposition[3] - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int coords[4];
  for(int i = 0 ; i < N_DIMS; i++) coords[i] = sourceposition[i] / HGC_localL[i];
  std::vector<Float> absPsi_loc(listR2.size(),0.0);
  std::vector<Float> absPsi(listR2.size(),0.0);
  if(is_myIt) compute_rms(*this,listR2,absPsi_loc,my_it,sourceposition);
  comm_barrier();
  int mpiErr = MPI_Allreduce(absPsi_loc.data(), absPsi.data(), listR2.size(), MPI_Type<Float>(), MPI_SUM, HGC_spaceComm);
  if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
  int rankHas = comm_rank_from_coords(HGC_default_topo, coords);
  mpiErr = MPI_Bcast(absPsi.data(), listR2.size(), MPI_Type<Float>(), rankHas, HGC_fullComm);
  if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);
  return absPsi;
}

template<typename Float>
void PLEGMA_Vector<Float>::mulGV(PLEGMA_Vector<Float> &vecIn, PLEGMA_Su3field<Float> &u){
  mulGV_k(*this, u, vecIn);
}


template class PLEGMA_Vector<float>;
template class PLEGMA_Vector<double>;

namespace plegma{
  
  template<typename Float>
  void copyToQUDA(ColorSpinorField *qudaVector, Float* delem, bool isEv){
    copy_to_QUDA(delem, *qudaVector, isEv);
  }
  template void copyToQUDA<float>(ColorSpinorField *qudaVector, float* delem, bool isEv);
  template void copyToQUDA<double>(ColorSpinorField *qudaVector, double* delem, bool isEv);

  template<typename Float>
  void copyFromQUDA(Float* delem, ColorSpinorField *qudaVector, bool isEv){
    copy_from_QUDA(delem, *qudaVector, isEv);
  }

  template void copyFromQUDA<float>(float* delem, ColorSpinorField *qudaVector, bool isEv);
  template void copyFromQUDA<double>(double* delem, ColorSpinorField *qudaVector, bool isEv);

  //----------------------------------//
  // class PLEGMA_Vector3D //
  //----------------------------------//
  template<typename Float>
  PLEGMA_Vector3D<Float>::
  PLEGMA_Vector3D(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag):
    PLEGMA_Field3D<Float>(alloc_flag, VECTOR3D, ghost_flag){
  }

  // vec3D <- Prop3D
  template<typename Float>
  void PLEGMA_Vector3D<Float>::absorb(PLEGMA_Propagator3D<Float> &prop, int nu , int c2){
    this->activeTimeSlice = prop.includesActiveTimeSlice();
    int V3 = HGC_localVolume/HGC_localL[3];
    Float *pointer_src = NULL;
    Float *pointer_dst = NULL;
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++){
	pointer_dst = (this->d_elem + mu*N_COLS*V3*2 + c1*V3*2);
	pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V3*2 + nu*N_COLS*N_COLS*V3*2 + c1*N_COLS*V3*2 + c2*V3*2);
	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
    checkCudaError();
  }

  // vec3D <- Prop4D
  template<typename Float>
  void PLEGMA_Vector3D<Float>::absorb(PLEGMA_Propagator<Float> &prop, int global_it, int nu , int c2){
    if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
    int my_it = global_it - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
    bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
    this->activeTimeSlice = is_myIt;
    int V3 = HGC_localVolume/HGC_localL[3];
    int V4 = HGC_localVolume;
    Float *pointer_src = NULL;
    Float *pointer_dst = NULL;
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++){
	pointer_dst = (this->d_elem + mu*N_COLS*V3*2 + c1*V3*2);
	if(is_myIt){
	  pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2 + my_it*V3*2);
	  cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
	}
	else
	  cudaMemset(pointer_dst, 0, V3*2 * sizeof(Float));
      }
    checkCudaError();
  }

  template class PLEGMA_Vector3D<float>;
  template class PLEGMA_Vector3D<double>;

}
