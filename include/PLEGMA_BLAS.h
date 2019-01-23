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
#include <mpi.h>

namespace cBLAS{
  //========================================================//
  template<typename Float>
  inline void axpy(int NN, Float val[2], Float *x, Float *y){}

  template<>
  inline void axpy<float>(int NN, float val[2], float *x, float *y){
    cblas_caxpy(NN,val,x,1,y,1);
  }

  template<>
  inline void axpy<double>(int NN, double val[2], double *x, double *y){
    cblas_zaxpy(NN,val,x,1,y,1);
  }

  //=================================================================//
  template<typename Float>
  inline void cscal(int NN, Float val[2], Float *x){}

  template<>
  inline void cscal<float>(int NN, float val[2], float *x){
    cblas_cscal(NN,val,x,1);
  }

  template<>
  inline void cscal<double>(int NN, double val[2], double *x){
    cblas_zscal(NN,val,x,1);
  }
}
//=================================================================//


namespace cuBLAS{
  template<typename Float>
  inline void axpy(int NN, Float val[2], Float *x, Float *y){}

  template<>
  inline void axpy<float>(int NN, float val[2], float *x, float *y){
    cuComplex cu_val = make_cuComplex(val[0],val[1]);
    cublasStatus_t error =  cublasCaxpy(cublas_handle,NN, &cu_val, (cuComplex*) x, 1, (cuComplex*) y, 1);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCaxpy failed with error %d", error);
  }

  template<>
  inline void axpy<double>(int NN, double val[2], double *x, double *y){
    cuDoubleComplex cu_val = make_cuDoubleComplex(val[0],val[1]);
    cublasStatus_t error =  cublasZaxpy(cublas_handle, NN, &cu_val,(cuDoubleComplex*) x, 1, (cuDoubleComplex*) y, 1);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasZaxpy failed with error %d", error);
  }

  template<typename Float>
  inline void scal(int NN, const Float val, Float *x);

  template<>
  inline void scal<float>(int NN, const float val, float *x){
    cublasStatus_t error = cublasCsscal(cublas_handle, NN, &val, (cuComplex*) x, 1);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCsscal failed with error %d", error);
  }

  template<>
  inline void scal<double>(int NN, const double val, double *x){
    cublasStatus_t error = cublasZdscal(cublas_handle, NN, &val, (cuDoubleComplex*) x, 1);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasZdscal failed with error %d", error);
  }

  template<typename Float>
  inline void dot(Float res[2],int NN, const Float *x, const Float *y, MPI_Comm comm);

  template<>
  inline void dot<float>(float res[2], int NN, const float *x, const float *y, MPI_Comm comm){
    cuComplex cu_res;
    cublasStatus_t error = cublasCdotc(cublas_handle, NN,(cuComplex*)x, 1, (cuComplex*)y,1,&cu_res);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCdotc failed with error %d", error);
    int mpiErr = MPI_Allreduce((float*) &cu_res, res, 2, MPI_FLOAT, MPI_SUM, comm);
    if(mpiErr != MPI_SUCCESS) errorQuda("MPI_Allreduce failed with error %d\n", mpiErr);
  }

  template<>
  inline void dot<double>(double res[2], int NN, const double *x, const double *y, MPI_Comm comm){
    cuDoubleComplex cu_res;
    cublasStatus_t error = cublasZdotc(cublas_handle, NN,(cuDoubleComplex*)x, 1, (cuDoubleComplex*)y,1,&cu_res);
    if(error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasZdotc failed with error %d", error);
    int mpiErr = MPI_Allreduce((double*) &cu_res, res, 2, MPI_DOUBLE, MPI_SUM, comm);
    if(mpiErr != MPI_SUCCESS) errorQuda("MPI_Allreduce failed with error %d\n", mpiErr);
  }  
}

//=================================================================//

namespace plegma{
  template<typename Float>
  void elemWiseMul(int NN, Float* x, Float* y);

}
