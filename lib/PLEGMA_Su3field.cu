#include <PLEGMA_Su3field.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_su3field.cuh>
#include <PLEGMA_field_utils.cuh>
#include <PLEGMA_SU3_projection.cuh>

using namespace plegma;

//--------------------------//
// class PLEGMA_Su3field //
//--------------------------//

template<typename Float>
PLEGMA_Su3field<Float>::PLEGMA_Su3field(ALLOCATION_FLAG alloc_flag): 
  PLEGMA_Field<Float>(alloc_flag, SU3FIELD){ ; }

template<typename Float>
void PLEGMA_Su3field<Float>::absorbDir_device(PLEGMA_Gauge<Float> &u,int dir){
  cudaMemcpy(this->d_elem, u.D_elem()+dir*(this->field_length)*(this->total_length)*2,
  	     this->bytes_total_length, cudaMemcpyDeviceToDevice);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Su3field<Float>::absorbDir_host(PLEGMA_Gauge<Float> &u,int dir){
  memcpy(this->h_elem, u.H_elem()+dir*(this->field_length)*(this->total_length)*2,
	 this->bytes_total_length);
}

template<typename Float>
void PLEGMA_Su3field<Float>::Udag(PLEGMA_Su3field<Float> &B){
  Udag_k(*this,B);
}

template<typename Float>
void PLEGMA_Su3field<Float>::UxU(PLEGMA_Su3field<Float> &B, PLEGMA_Su3field<Float> &C){
  UxU_k(*this,B,C);
}

template<typename Float>
void PLEGMA_Su3field<Float>::UxUdag(PLEGMA_Su3field<Float> &B, PLEGMA_Su3field<Float> &C){
  UxUdag_k(*this,B,C);
}

template<typename Float>
void PLEGMA_Su3field<Float>::su3Projection(){
  su3Projection_k(*this);
}

template<typename Float>
void PLEGMA_Su3field<Float>::traceHerExpMap(PLEGMA_Su3field<Float> &A){
  traceHerExpMap_kernel(*this,A);
}


template<typename Float>
static void pathX(int *dir, int *sign, int length,PLEGMA_Su3field<Float> **u_s,
		 PLEGMA_Su3field<Float>& s1, PLEGMA_Su3field<Float>& s2){
  // do first step
  if(sign[0] > 0) s1.shift( *(u_s[dir[0]]), dir[0] );
  else s1.Udag( *(u_s[dir[0]]) );
  // do the next steps
  for(int j=1 ; j < length ; j++){
    if(sign[j] > 0){
      s2.UxU(s1, *(u_s[dir[j]]) );
      s1.shift(s2, dir[j]);
    }
    else{
      s2.shift(s1,4+dir[j]);
      s1.UxUdag(s2, *(u_s[dir[j]]) );
    }
  }
}

static void dirsOrien(std::vector<int> &steps, int len, int *dir, int *sign){
  for(int i = 0 ; i < len ; i++)
    if( !((steps[i] >= 0) && (steps[i] <= 7)) ) errorQuda("Error you provided a direction which is not supported");
  for(int i=0; i<len; ++i)
    {
      dir[i] = (steps[i]>3)?steps[i]-4:steps[i];
      sign[i] = (steps[i]>3)?-1:1;
    }
}

template<typename Float>
void PLEGMA_Su3field<Float>::path(std::vector<int> &steps, PLEGMA_Su3field<Float> **u, PLEGMA_Su3field<Float> &tmp){
  int len = steps.size();
  int dir[len], sign[len];
  dirsOrien(steps,len,dir,sign);
  pathX(dir,sign,len,u,*this,tmp);
}

template<typename Float>
void PLEGMA_Su3field<Float>::path(std::vector<int> &steps, PLEGMA_Su3field<Float> **u){
  PLEGMA_Su3field<Float> tmp(BOTH);
  path(steps,u,tmp);
}

template<typename Float>
void PLEGMA_Su3field<Float>::staples(PLEGMA_Su3field<Float> **u, int dir, PLEGMA_Su3field<Float> &tmp1,
				     PLEGMA_Su3field<Float> &tmp2, Float rho, int D3D4){
  if(D3D4 != 3 && D3D4 !=4) errorQuda("Only 3D and 4D sum of staples is allowed");
  this->zero_device();
  for(int i = 0 ; i < D3D4 ; i++)
    if (i != dir){
      int spath1[] = {i,dir,4+i};
      std::vector<int> vspath1(spath1,spath1+3);
      tmp2.path(vspath1,u,tmp1);
      xpby(*this,*this,tmp2,rho);
    }
  for(int i = 0 ; i < D3D4 ; i++)
    if (i != dir){
      int spath2[] = {4+i,dir,i};
      std::vector<int> vspath2(spath2,spath2+3);
      tmp2.path(vspath2,u,tmp1);
      xpby(*this,*this,tmp2,rho);
    }
  tmp1.shift(*this,4+dir);
  cudaMemcpy(this->D_elem(), tmp1.D_elem(), this->Bytes_total(), cudaMemcpyDeviceToDevice);
  checkCudaError();
}


template<typename Float>
void PLEGMA_Su3field<Float>::wilsonLineUpdate(PLEGMA_Su3field<Float> &inOut, PLEGMA_Su3field<Float> &tmp,  int dirOr){
  /* This function updates the wilson line with the provided input field
   * The input field is also output in the sense that is shifted in order to
   * continue building the Wilson line. The option dirOr is the direction and
   * orientation we shift the fields to do the line. For example if you want
   * to build a Wilson line in +x direction you should provide dirOr=4+0.
   */
  if( !((dirOr >= 0) && (dirOr <= 7)) ) errorQuda("Error you provided a direction which is not supported");
  cudaMemcpy(tmp.D_elem(), inOut.D_elem(), tmp.Bytes_total(), cudaMemcpyDeviceToDevice);
  checkCudaError();
  if(dirOr > 3){
    UxU(*this, inOut);
    inOut.shift(tmp,dirOr);
  }
  else{
    inOut.shift(tmp,dirOr);
    UxUdag(*this, inOut);
  }
}

template class PLEGMA_Su3field<float>;
template class PLEGMA_Su3field<double>;
