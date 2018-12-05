#if defined(HAVE_MKL) && defined(HAVE_OPENBLAS)
#error Cannot define both mkl and openBLAS
#endif

#if defined(HAVE_MKL)
#include <mkl.h>
#elif defined(HAVE_OPENBLAS)
#include <cblas.h>
//#include <common.h> // do not know when is needed or not
#else
#error Neither mkl nor openBLAS have been defined
#endif

#include <cublas_v2.h>

namespace cBLAS{
  //========================================================//
  template<typename Float>
  void axpy(int NN, Float val[2], Float *x, Float *y){}

  template<>
  void axpy<float>(int NN, float val[2], float *x, float *y){
    cblas_caxpy(NN,val,x,1,y,1);
  }

  template<>
  void axpy<double>(int NN, double val[2], double *x, double *y){
    cblas_zaxpy(NN,val,x,1,y,1);
  }

  //=================================================================//
  template<typename Float>
  void cscal(int NN, Float val[2], Float *x){}

  template<>
  void cscal<float>(int NN, float val[2], float *x){
    cblas_cscal(NN,val,x,1);
  }

  template<>
  void cscal<double>(int NN, double val[2], double *x){
    cblas_zscal(NN,val,x,1);
  }
}
//=================================================================//


namespace cuBLAS{
  template<typename Float>
  void axpy(int NN, Float val[2], Float *x, Float *y){}

  template<>
  void axpy<float>(int NN, float val[2], float *x, float *y){
    cuComplex cu_val = make_cuComplex(val[0],val[1]);
    cublasStatus_t error =  cublasCaxpy(cublas_handle,NN, &cu_val, (cuComplex*) x, 1, (cuComplex*) y, 1);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
  }

  template<>
  void axpy<double>(int NN, double val[2], double *x, double *y){
    cuDoubleComplex cu_val = make_cuDoubleComplex(val[0],val[1]);
    cublasStatus_t error =  cublasZaxpy(cublas_handle, NN, &cu_val,(cuDoubleComplex*) x, 1, (cuDoubleComplex*) y, 1);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
  }
}
