#include <PLEGMA_Vector.h>
#include <PLEGMA_Su3field.h>
#include <cuda_fp16.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_vector_utils.cuh> 
#include <PLEGMA_gaussian_smearing.cuh> 
#include <PLEGMA_seqSourceNucleon.cuh> 
#include <PLEGMA_covD.cuh>
#include <split_grid.h>  // single-arg comm_rank_from_coords(const int*)
#ifdef PLEGMA_SCATTERING_CONTRACTIONS
#include <PLEGMA_gammas.h>
#include <kernels/PLEGMA_gammas_scatt.cuh>
#endif
#include <communicator_quda.h>
#include <quda_api.h>
#include <device.h>
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
					    int nsmearGauss, Float alphaGauss){
	
  if(vecIn.IsAllocHost()) {
    vecIn.unload(); // backing up the vecIn
  } else {
    PLEGMA_warning("VecIn is not allocated on BOTH; gaussianSmearing will overwrite the device memory.\n");
  }

  assert(this->checkVolume(vecIn, gauge));
  auto texGauge = toTexture<gaugeTex>(gauge);
  auto texVecIn = toTexture<vectorTex>(vecIn);
  auto texVecOut = toTexture<vectorTex>(*this);

  for(int i = 0 ; i < nsmearGauss ; i++){
    if( (i%2) == 0){
      for(int dir=0; dir<N_DIMS-1; dir++) {
	if(i==0) {
	  gauge.communicateSideGhost(dir, DIR_BOTH, START);
	}
	vecIn.communicateSideGhost(dir, DIR_BOTH, START);
      }
      gaussian_smearing_no_ghost(*texVecOut,*texVecIn,*texGauge, alphaGauss);
      for(int dir=0; dir<N_DIMS-1; dir++) {
	if(i==0) {
	  gauge.communicateSideGhost(dir, DIR_BOTH, FINISH);
	}
	vecIn.communicateSideGhost(dir, DIR_BOTH, FINISH);
      }
      gaussian_smearing_only_ghost(*texVecOut,*texVecIn,*texGauge, alphaGauss);
    }
    else{
      for(int dir=0; dir<N_DIMS-1; dir++) {
	this->communicateSideGhost(dir, DIR_BOTH, START);
      }
      gaussian_smearing_no_ghost(*texVecIn, *texVecOut, *texGauge, alphaGauss);
      for(int dir=0; dir<N_DIMS-1; dir++) {
	this->communicateSideGhost(dir, DIR_BOTH, FINISH);
      }
      gaussian_smearing_only_ghost(*texVecIn, *texVecOut, *texGauge, alphaGauss);
    }
  }
  if( (nsmearGauss%2) == 0)
    qudaMemcpy(this->D_elem(),vecIn.D_elem(),
	       this->Bytes_total(),qudaMemcpyDeviceToDevice);
  
  checkQudaError();

  if(vecIn.IsAllocHost()) {
    vecIn.load(); // restoring vecIn
  }
}


template<typename Float>
void PLEGMA_Vector<Float>::copyToQUDA(std::vector<ColorSpinorField> &qudaVector, bool isEv){
  copy_to_QUDA(this->d_elem, (qudaVector), 0, isEv);
}

template<typename Float>
void PLEGMA_Vector<Float>::copyFromQUDA( std::vector<ColorSpinorField> &qudaVector, bool isEv){
  copy_from_QUDA(this->d_elem, (qudaVector), 0, isEv);
}

template<typename Float>
void  PLEGMA_Vector<Float>::apply_gamma5(){
  apply_gamma5_vector(toField2<vector2>(*this));
}

template<typename Float>
void PLEGMA_Vector<Float>::rotateToPhysicalBasis(PLEGMA_Vector<Float> &vecIn, int sgn){
  PLEGMA_Vector<Float> temporary;
  temporary.copy(vecIn);
  temporary.apply_gamma5();
  temporary.cscale((std::complex<Float>) {0.,(Float)sgn});
  temporary.add(vecIn);
  temporary.scale(1./sqrt(2.));
  this->copy(temporary);
   
}

template<typename Float> 
void  PLEGMA_Vector<Float>::apply_gamma(GAMMAS gMat,LEFTRIGHT LR){
  apply_gamma_vector(LR,toField2<vector2>(*this),gMat);
}
#ifdef PLEGMA_SCATTERING_CONTRACTIONS
template<typename Float>
void  PLEGMA_Vector<Float>::apply_gamma_scatt(GAMMAS_SCATT gMat,LEFTRIGHT LR){
  apply_gamma_scatt_vector(LR,toField2<vector2>(*this),gMat);
}
#endif
template<typename Float>
void PLEGMA_Vector<Float>::rotate_uk_ch_g5g4(){
  rotate_uk_ch_g5g4_k(toField2<vector2>(*this));
}

template<typename Float>
void PLEGMA_Vector<Float>::rotate_uk_ch_etmc(){
  rotate_uk_ch_etmc_k(toField2<vector2>(*this));
}


// vec4D <- Prop3D
template<typename Float>
void PLEGMA_Vector<Float>::absorb(PLEGMA_Propagator3D<Float> &prop, int global_it, int nu , int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  static bool init_absorb_vec4D_prop3D = false;

  if (!init_absorb_vec4D_prop3D) {

    Float2<Float>* tempquda=(Float2<Float>*)device_malloc(V4*2*sizeof(Float));
    PLEGMA_memset(tempquda,0, V4*2*sizeof(Float));
    PLEGMA_memcpy(tempquda, tempquda,V3*2*sizeof(Float),qudaMemcpyDeviceToDevice);
    device_free(tempquda);
    init_absorb_vec4D_prop3D=true;

  }
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      PLEGMA_memset(this->d_elem + mu*N_COLS*V4*2 + c1*V4*2, 0, V4*2*sizeof(Float));
      if(is_myIt){
	pointer_dst = (this->d_elem + mu*N_COLS*V4*2 + c1*V4*2 + my_it*V3*2);
	pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V3*2 + nu*N_COLS*N_COLS*V3*2 + c1*N_COLS*V3*2 + c2*V3*2);
	PLEGMA_memcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
      }
    }
  comm_barrier();
  checkQudaError();
}

// vec4D <- prop4D (it)
template<typename Float>
void PLEGMA_Vector<Float>::absorb(PLEGMA_Propagator<Float> &prop, int global_it, int nu , int c2){
  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
  int V3 = HGC_localVolume/HGC_localL[3];
  int V4 = HGC_localVolume;
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      qudaMemset(this->d_elem + mu*N_COLS*V4*2 + c1*V4*2, 0, V4*2*sizeof(Float));
      if(is_myIt){
	pointer_dst = (this->d_elem + mu*N_COLS*V4*2 +  c1*V4*2 + my_it*V3*2);
       	pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2 + my_it*V3*2);
       	qudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
      }
    }
  comm_barrier();
  checkQudaError();
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
      qudaMemcpy(pointer_dst, pointer_src, V4*2 * sizeof(Float), qudaMemcpyDeviceToDevice);   
    }
  checkQudaError();
}
template<typename Float>
void PLEGMA_Vector<Float>::absorb(PLEGMA_Propagator<Float> *prop, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  int V4 = HGC_localVolume;
  for(int mu = 0 ; mu < N_SPINS ; mu++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      pointer_dst = (this->d_elem + mu*N_COLS*V4*2 +  c1*V4*2);
      pointer_src = (prop->D_elem() + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2);
      qudaMemcpy(pointer_dst, pointer_src, V4*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
    }
  checkQudaError();
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
        qudaMemcpy((this->d_elem + ((c1 + mu*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), qudaMemcpyDeviceToDevice); 
      } 
    }
  checkQudaError();
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
        qudaMemcpy((this->d_elem + ((c1 + mu*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), qudaMemcpyDeviceToDevice); 
      } 
    }
  checkQudaError();
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
        qudaMemcpy((this->d_elem + ((c1 + mu*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), qudaMemcpyDeviceToDevice); 
      } 
    }
  checkQudaError();
}

template<typename Float>
void PLEGMA_Vector<Float>::diluteSpinDisplace(PLEGMA_Vector<Float> &vecIn, int spin1, int spin2){
  Float *pointer_src = NULL;
  if( (spin1 >= N_SPINS) || (spin2>=N_SPINS) ) PLEGMA_error("The spin index you provided exceed the total spin content\n");
  this->zero_device();
  for(int c1 = 0 ; c1 < N_COLS ; c1++){
    pointer_src = (vecIn.D_elem() + (c1 + spin2*N_COLS)*HGC_localVolume*2);
    qudaMemcpy((this->d_elem + ((c1 + spin1*N_COLS)*HGC_localVolume)*2), pointer_src, HGC_localVolume*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
      
  }
  checkQudaError();
}


template<typename Float>
void PLEGMA_Vector<Float>::pack_fermion_to_sink(std::vector<PLEGMA_Vector<Float>*> &stochastic_vector, int sinktime){
  PLEGMA_Vector<Float> stmp;

  stmp.zero_where(DEVICE);
  stmp.zero_where(HOST);


  PLEGMA_Vector<Float> temporary4D;
  PLEGMA_Vector3D<Float> temporary3D;
  for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
    temporary4D.copy(*stochastic_vector[timeidx], HOST);
    temporary4D.load();
    temporary3D.absorb(temporary4D, sinktime );
    temporary4D.absorb(temporary3D, timeidx );
    stmp.absorbTimeslice(temporary4D, timeidx, false);
  }

  this->copy(stmp);

}

template<typename Float>
void PLEGMA_Vector<Float>::pack_propagator(PLEGMA_Vector<Float> &in_ppa, PLEGMA_Vector<Float> &in_pma, int t0, int deltat, bool initialize){
  PLEGMA_Vector<Float> stmp;
  if (initialize==true){
    stmp.zero_where(DEVICE);
    stmp.zero_where(HOST);
  }
  else{
      stmp.copy(*this);
  }

  for ( int dt_tmp=-deltat+1; dt_tmp<deltat; ++dt_tmp){
    int t1=t0+dt_tmp;
    if ((t1>=0) && (t1<HGC_totalL[DIM_T])){
      int t_tmp=t1;
      PLEGMA_printf("# [pack_propagator] packing v1 timslice t = %3d for t0 = %3d and dt = %3d\n", t_tmp, t0, dt_tmp );
      stmp.absorbTimeslice(in_ppa, t_tmp,false);
    }
    else{
      int t_tmp=(t1+HGC_totalL[DIM_T])%HGC_totalL[DIM_T];
      PLEGMA_printf("# [pack_propagator] packing v2 timslice t = %3d for t0 = %3d and dt = %3d\n", t_tmp, t0, dt_tmp );
      stmp.absorbTimeslice(in_pma, t_tmp,false);
    }
  }
  this->copy(stmp);
}
template<typename Float>
void PLEGMA_Vector<Float>::pack_propagator_as_sink(PLEGMA_Vector<Float> &in, int sinktimeslice, int source_sink_separation, bool initialize){

  PLEGMA_Vector<Float> stmp;
  PLEGMA_Vector3D<Float> vector1;

  if (initialize==true){
    stmp.zero_where(DEVICE);
    stmp.zero_where(HOST);
  }
  else{
      stmp.copy(*this);
  }

  vector1.absorb(in, sinktimeslice, true);

  for (int dt=0; dt<source_sink_separation; ++dt){
    int actualtimeslice= ((sinktimeslice-source_sink_separation+dt)+  HGC_totalL[DIM_T])%HGC_totalL[DIM_T];
    stmp.absorb(vector1, actualtimeslice, false);
  }

  this->copy(stmp);
}


template<typename Float>
void PLEGMA_Vector<Float>::pack_propagator_from_source_to_sink(PLEGMA_Vector<Float> &in, int sinktimeslice, int source_sink_separation, bool initialize){

  PLEGMA_Vector<Float> stmp;
  PLEGMA_Vector3D<Float> vector1;

  if (initialize==true){
    stmp.zero_where(DEVICE);
    stmp.zero_where(HOST);
  }
  else{
    stmp.copy(*this);
  }


  for (int dt=0; dt<source_sink_separation; ++dt){
    int actualtimeslice= ((sinktimeslice-source_sink_separation+dt)+  HGC_totalL[DIM_T])%HGC_totalL[DIM_T];
    vector1.absorb(in, actualtimeslice );
    stmp.absorb(vector1, actualtimeslice, false);
  }

  this->copy(stmp);
}




template<typename Float>
void PLEGMA_Vector<Float>::pointSource(const site& sourceposition, int spin, int color, ALLOCATION_FLAG where){
  if(where == EVERY) where = this->allocation;
  for(int i = 0; i < N_DIMS; i++)
    if(sourceposition[i] >= HGC_totalL[i]) PLEGMA_error("Source position component in dir=%d, is %d >= %d the lattice extent", i, sourceposition[i],HGC_totalL[i]);
  
  this->zero_where(where);
  int my_src[N_DIMS];
  size_t id=0;
  static bool init_PointSource = false;

  if (!init_PointSource) {
    Float * temphost=(Float*)malloc(sizeof(Float));
    Float2<Float> *tempquda=(Float2<Float> *)device_malloc(sizeof(Float));
    PLEGMA_memcpy(tempquda, temphost,sizeof(Float),
                qudaMemcpyHostToDevice );
    free(temphost);
    device_free(tempquda);
    init_PointSource=true;
  }

  for(int i = N_DIMS-1; i >= 0; i--) {
    my_src[i] = (sourceposition[i] - HGC_procPosition[i] * HGC_localL[i]);

    // if out of the local lattice we break
    if((my_src[i]<0) || (my_src[i]>=HGC_localL[i])) return;

    id = id * HGC_localL[i] + my_src[i];
  }
  // This make it work also for vector3D
  id = id % this->Total_length();

  Float temp[1];
  temp[0] = 1.0;
  if( where == BOTH ){
    this->h_elem[((spin*N_COLS+color)*HGC_localVolume + id)*2] = 1.0; 
    PLEGMA_memcpy((this->d_elem + ((spin*N_COLS+color)*this->Total_length() + id)*2), temp,sizeof(Float),
                qudaMemcpyHostToDevice ); 
  }
  else if (where == HOST){
    this->h_elem[((spin*N_COLS+color)*this->Total_length() + id)*2] = 1.0; 
  }
  else if (where == DEVICE){
    PLEGMA_memcpy((this->d_elem + ((spin*N_COLS+color)*this->Total_length() + id)*2), temp,sizeof(Float),
                qudaMemcpyHostToDevice ); 
  }
  else{
    PLEGMA_error("Not supported %d\n",where);
  }
}
template<typename Float>
std::shared_ptr<Float> PLEGMA_Vector<Float>::getPointSource( const site& sourceposition, ALLOCATION_FLAG where){
  if (where == HOST){
    std::shared_ptr<Float> ptr((Float *)malloc(sizeof(Float)*N_SPINS*N_COLS*2), free);

    for(int i = 0; i < N_DIMS; i++)
      if(sourceposition[i] >= HGC_totalL[i]) PLEGMA_error("Source position component in dir=%d, is %d >= %d the lattice extent", i, sourceposition[i],HGC_totalL[i]);

  
    int my_src[N_DIMS];
  
    size_t id=0;
    for(int i = N_DIMS-1; i >= 0; i--) {

      my_src[i] = (sourceposition[i] - HGC_procPosition[i] * HGC_localL[i]);
       
      id = id * HGC_localL[i] + my_src[i];
    
    }
  
    // This make it work also for vector3D
    id = id % this->Total_length();

    int coords[4];
    for(int i = 0 ; i < N_DIMS; i++) coords[i] = sourceposition[i] / HGC_localL[i];
    int rankHas = comm_rank_from_coords(coords);

    if (comm_rank()==rankHas){
      for (int spin=0; spin<N_SPINS; ++spin){
        for (int color=0; color<N_COLS; ++color){
    
	  ptr.get()[2*(spin*N_COLS+color)+0]=this->h_elem[((spin*N_COLS+color)*HGC_localVolume + id)*2] ;
          ptr.get()[2*(spin*N_COLS+color)+1]=this->h_elem[((spin*N_COLS+color)*HGC_localVolume + id)*2+1] ;
        }
      }
    }

    MPI_Barrier(HGC_fullComm);

    int mpiErr = MPI_Bcast(ptr.get(), 2*N_SPINS*N_COLS, MPI_Type<Float>(), rankHas, HGC_fullComm);

    MPI_Barrier(HGC_fullComm);

    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);

    return ptr;

  }
  else{
    PLEGMA_error("Not supported %d\n",where);
  }

}



template<typename Float>
void PLEGMA_Vector<Float>::covD(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gauge, int dirOr){
  // to increase efficiency the communication of the ghost for the the vector should happen before calling this function
  if(dirOr < 0 || dirOr > 7) PLEGMA_error("Wrong direction is given");
  assert(this->checkVolume(vecIn, gauge));
  auto texVecIn = toTexture<vectorTex>(vecIn);
  auto texGaugeIn = toTexture<gaugeTex>(gauge);
  covD_k<Float,Float,Float>(toField2<vector2>(*this), *texVecIn, *texGaugeIn, dirOr);
}

template<typename Float>
void PLEGMA_Vector<Float>::mulGV(PLEGMA_Vector<Float> &vecIn, PLEGMA_Su3field<Float> &u){
  assert(this->checkVolume(vecIn, u));
  mulGV_k(toField2<vector2>(*this), toField2<su3_2>(u), toField2<vector2>(vecIn));
}


template class PLEGMA_Vector<float>;
template class PLEGMA_Vector<double>;

namespace plegma{
  
  template<typename Float>
  void copyToQUDA(std::vector<ColorSpinorField>& qudaVector, Float* delem,  bool isEv){
    copy_to_QUDA(delem, qudaVector, 0, isEv);
  }
  template void copyToQUDA<float>(std::vector<ColorSpinorField>&qudaVector, float* delem, bool isEv);
  template void copyToQUDA<double>(std::vector<ColorSpinorField>&qudaVector, double* delem, bool isEv);

  template<typename Float>
  void copyFromQUDA(Float* delem, std::vector<ColorSpinorField>&qudaVector,  bool isEv){
    copy_from_QUDA(delem, qudaVector, 0, isEv);
  }

  template void copyFromQUDA<float>(float* delem, std::vector<ColorSpinorField>& qudaVector,  bool isEv);
  template void copyFromQUDA<double>(double* delem, std::vector<ColorSpinorField>& qudaVector, bool isEv);

  //----------------------------------//
  // class PLEGMA_Vector3D //
  //----------------------------------//
  
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
	qudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
      }
    checkQudaError();
  }

  // vec3D <- prop4D
  template<typename Float>
  void PLEGMA_Vector3D<Float>::absorb(PLEGMA_Propagator<Float> &prop, int global_it, int nu , int c2, bool broadcast){
    if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
    int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
    bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
    this->activeTimeSlice = is_myIt;
    int V3 = HGC_localVolume/HGC_localL[3];
    int V4 = HGC_localVolume;
    Float *pointer_src = NULL;
    Float *pointer_dst = NULL;
    static bool init_vector3D_prop4D = false;

    if (!init_vector3D_prop4D) {

      Float2<Float>* tempquda=(Float2<Float>*)device_malloc(V3*2 * sizeof(Float));
      PLEGMA_memcpy(tempquda, tempquda, V3*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
      PLEGMA_memset(tempquda, 0, V3*2 * sizeof(Float));
      device_free(tempquda);
      init_vector3D_prop4D=true;

    }

    for(int mu = 0 ; mu < N_SPINS ; mu++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++){
	pointer_dst = (this->d_elem + mu*N_COLS*V3*2 + c1*V3*2);
	if(is_myIt){
	  pointer_src = (prop.D_elem() + mu*N_SPINS*N_COLS*N_COLS*V4*2 + nu*N_COLS*N_COLS*V4*2 + c1*N_COLS*V4*2 + c2*V4*2 + my_it*V3*2);
	  PLEGMA_memcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
	}
	
	if (broadcast == true){
         int time_rank=global_it/HGC_localL[3];
//         printf("Time rank %d\n",time_rank);
//         fflush(stdout);
         Float *temp=(Float *)malloc(sizeof(Float)*V3*2);
         PLEGMA_memcpy(temp, pointer_dst, V3*2 * sizeof(Float), qudaMemcpyDeviceToHost);
         MPI_Bcast(temp, V3*2 , MPI_Type<Float>(), time_rank, HGC_timeComm);
//         printf("Temp 0 %e\n",temp[0]);
//         fflush(stdout);
         PLEGMA_memcpy(pointer_dst, temp, V3*2 * sizeof(Float), qudaMemcpyHostToDevice);
         free(temp);
       }
       if (broadcast == false && is_myIt ==false){
         PLEGMA_memset(pointer_dst, 0, V3*2 * sizeof(Float));
       }
      
    }
    checkQudaError();
    
  }

  // vec3D <- vec4D
  template<typename Float>
  void PLEGMA_Vector3D<Float>::absorb(PLEGMA_Vector<Float> &prop, int global_it, bool broadcast){
    if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");
    int my_it = global_it - HGC_procPosition[3] * HGC_localL[3];
    bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
    this->activeTimeSlice = is_myIt;
    int V3 = HGC_localVolume/HGC_localL[3];
    int V4 = HGC_localVolume;
    Float *pointer_src = NULL;
    Float *pointer_dst = NULL;
    static bool init_vector3D_vector4D = false;

    if (!init_vector3D_vector4D) {
     Float2<Float> *tmpdevice=(Float2<Float>*)device_malloc(V3*2 * sizeof(Float));
     PLEGMA_memset(tmpdevice, 0, V3*2 * sizeof(Float));
     PLEGMA_memcpy(tmpdevice, tmpdevice,V3*2 * sizeof(Float),qudaMemcpyDeviceToDevice);
     device_free(tmpdevice);
     init_vector3D_vector4D=true;
    }
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
      {
        pointer_dst = (this->d_elem + mu*N_COLS*V3*2 + c1*V3*2);
        if(is_myIt){
          pointer_src = (prop.D_elem() + mu*N_COLS*V4*2 + c1*V4*2 + my_it*V3*2);
          PLEGMA_memcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), qudaMemcpyDeviceToDevice);
        }
	if (broadcast == true){
         int time_rank=global_it/HGC_localL[3];
//         printf("Time rank %d\n",time_rank);
//         fflush(stdout);
         Float *temp=(Float *)malloc(sizeof(Float)*V3*2);
         PLEGMA_memcpy(temp, pointer_dst, V3*2 * sizeof(Float), qudaMemcpyDeviceToHost);
         MPI_Bcast(temp, V3*2 , MPI_Type<Float>(), time_rank, HGC_timeComm);
//         printf("Temp 0 %e\n",temp[0]);
//         fflush(stdout);
         PLEGMA_memcpy(pointer_dst, temp, V3*2 * sizeof(Float), qudaMemcpyHostToDevice);
         free(temp);
       }
       if (broadcast == false && is_myIt ==false){
         PLEGMA_memset(pointer_dst, 0, V3*2 * sizeof(Float));
       }

      }

    checkQudaError();

  }



  template<typename Float>
  std::vector<Float> PLEGMA_Vector3D<Float>::rms(std::vector<int> listR2, const site& sourceposition) const{
    if(listR2.size() <= 0) PLEGMA_error("Provided list of r2 is empty");
    for(int i = 0; i < N_DIMS; i++)
      if(sourceposition[i] >= HGC_totalL[i]) PLEGMA_error("Source position component in dir=%d, is %d >= %d the lattice extent", i, sourceposition[i],HGC_totalL[i]);
    std::vector<Float> absPsi_loc(listR2.size(),0.0);
    std::vector<Float> absPsi(listR2.size(),0.0);
    if(this->includesActiveTimeSlice()) {
      compute_rms(*this,listR2,absPsi_loc,sourceposition);
      int mpiErr = MPI_Allreduce(absPsi_loc.data(), absPsi.data(), listR2.size(), MPI_Type<Float>(), MPI_SUM, HGC_spaceComm);
      if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
    }
    
    int coords[4];
    for(int i = 0 ; i < N_DIMS; i++) coords[i] = sourceposition[i] / HGC_localL[i];
    int rankHas = comm_rank_from_coords(coords);
    int mpiErr = MPI_Bcast(absPsi.data(), listR2.size(), MPI_Type<Float>(), rankHas, HGC_fullComm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);
    return absPsi;
  }

  template<typename Float>
  void PLEGMA_Vector3D<Float>::seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2){
    this->activeTimeSlice = prop.activeTimeSlice;
    this->zero_device();

    auto texProp = toTexture<propTex>(prop);
    contractNucleonSeqSource<Float,Float>(toField2<vector2>(*this), *texProp, proj, particle, c_nu, c_c2);
  }
  
  template<typename Float>
  void PLEGMA_Vector3D<Float>::seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop1, PLEGMA_Propagator3D<Float> &prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2){
    this->activeTimeSlice = prop1.activeTimeSlice;
    this->zero_device();

    auto texProp1 = toTexture<propTex>(prop1);
    auto texProp2 = toTexture<propTex>(prop2);
    contractNucleonSeqSource<Float,Float,Float>(toField2<vector2>(*this), *texProp1, *texProp2, proj, particle, c_nu, c_c2);
  }
  
  
  template class PLEGMA_Vector3D<float>;
  template class PLEGMA_Vector3D<double>;
}
