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
#pragma once
enum OPER_MATR_BLAS {NOTRANS, TRANS, DAGGER};
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
  //--------------------------------------------------------------------
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

  //--------------------------------------------------------
  template<typename Float>
  inline void scal(int NN, Float val, Float *x){}

  template<>
  inline void scal<float>(int NN, float val, float *x){
    cblas_csscal(NN,val,x,1);
  }

  template<>
  inline void scal<double>(int NN, double val, double *x){
    cblas_zdscal(NN,val,x,1);
  }

  //---------------------------------------------------------
  template<typename Float>
  inline void gemv_(OPER_MATR_BLAS trans, int m, int n, Float alpha[2], Float* A, Float* x, Float beta[2], Float* y){}

  template<>
  inline void gemv_<float>(OPER_MATR_BLAS trans, int m, int n, float alpha[2], float* A, float* x, float beta[2], float* y){
    CBLAS_TRANSPOSE Oper;
    switch(trans){case(NOTRANS): Oper=CblasNoTrans; break; case(TRANS): Oper=CblasTrans; break; case(DAGGER): Oper=CblasConjTrans; break;}
    cblas_cgemv(CblasColMajor, Oper, m, n, (float*) alpha, A, m, x, 1, (float*) beta, y, 1);
  }

  template<>
  inline void gemv_<double>(OPER_MATR_BLAS trans, int m, int n, double alpha[2], double* A, double* x, double beta[2], double* y){
    CBLAS_TRANSPOSE Oper;
    switch(trans){case(NOTRANS): Oper=CblasNoTrans; break; case(TRANS): Oper=CblasTrans; break; case(DAGGER): Oper=CblasConjTrans; break;}
    cblas_zgemv(CblasColMajor, Oper, m, n, (double*) alpha, A, m, x, 1, (double*) beta, y, 1);
  }

  // In case of m is partitioned and trans=NOTRANS OR m is not partitioned one can use whatever trans
  // Cannot work if n is partitioned
  template<typename Float>
  inline void gemv(OPER_MATR_BLAS trans, int m, int n, Float alpha[2], Float* A, Float* x, Float beta[2], Float* y){
    gemv_(trans, m, n, alpha, A, x, beta, y);
  }
  // In case that m is partitioned and we take trans of dagger then reduction is needed
  // Cannot work if n is partitioned
  template<typename Float>
  inline void gemv(OPER_MATR_BLAS trans, int m, int n, Float alpha[2], Float* A, Float* x, Float beta[2], Float* y, MPI_Comm comm){
    if(trans == NOTRANS) PLEGMA_error("Use gemv without MPI comm");
    if(comm == MPI_COMM_NULL) PLEGMA_error("Communicator is NULL and cannot be used for MPI reduction");
    Float *yr = nullptr;
    try{yr = new Float[n*2];} catch (std::bad_alloc& err){ PLEGMA_error(err.what());}
    gemv_(trans, m, n, alpha, A, x, beta, y);
    int mpiErr = MPI_Allreduce(y,yr,n*2,MPI_Type(yr),MPI_SUM,comm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
    memcpy(y,yr,n*2*sizeof(Float));
    delete[] yr;
  }
  
}
//=================================================================//


namespace cuBLAS{
  
  template<typename Float>
  inline void axpy(int NN, Float val[2], Float *x, Float *y){}
  template<>
  inline void axpy<float>(int NN, float val[2], float *x, float *y){
    cuComplex cu_val = make_cuComplex(val[0],val[1]);
    cublasStatus_t error =  cublasCaxpy(HGC_cublas_handle,NN, &cu_val, (cuComplex*) x, 1, (cuComplex*) y, 1);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCaxpy failed with error %d", error);
  }
  template<>
  inline void axpy<double>(int NN, double val[2], double *x, double *y){
    cuDoubleComplex cu_val = make_cuDoubleComplex(val[0],val[1]);
    cublasStatus_t error =  cublasZaxpy(HGC_cublas_handle, NN, &cu_val,(cuDoubleComplex*) x, 1, (cuDoubleComplex*) y, 1);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasZaxpy failed with error %d", error);
  }
  
  //------------------------------------------------------------------
  template<typename Float>
  inline void scal(int NN, const Float val, Float *x);
  template<>
  inline void scal<float>(int NN, const float val, float *x){
    cublasStatus_t error = cublasCsscal(HGC_cublas_handle, NN, &val, (cuComplex*) x, 1);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCsscal failed with error %d", error);
  }
  template<>
  inline void scal<double>(int NN, const double val, double *x){
    cublasStatus_t error = cublasZdscal(HGC_cublas_handle, NN, &val, (cuDoubleComplex*) x, 1);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasZdscal failed with error %d", error);
  }
  //------------------------------------------------------------------
  template<typename Float>
  inline void cscal(int NN, const Float val[2], Float *x);
  template<>
  inline void cscal<float>(int NN, const float val[2], float *x){
    cuComplex cu_val = make_cuComplex(val[0],val[1]);
    cublasStatus_t error = cublasCscal(HGC_cublas_handle, NN, &cu_val, (cuComplex*) x, 1);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCscal failed with error %d", error);
  }
  template<>
  inline void cscal<double>(int NN, const double val[2], double *x){
    cuDoubleComplex cu_val = make_cuDoubleComplex(val[0],val[1]);
    cublasStatus_t error = cublasZscal(HGC_cublas_handle, NN, &cu_val, (cuDoubleComplex*) x, 1);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasZscal failed with error %d", error);
  }
  //-----------------------------------------------------------------
  template<typename Float>
  inline std::complex<Float> dot(int NN, const Float *x, const Float *y);
  template<>
  inline std::complex<float> dot<float>(int NN, const float *x, const float *y){
    cuComplex cu_res;
    cublasStatus_t error = cublasCdotc(HGC_cublas_handle, NN,(cuComplex*)x, 1, (cuComplex*)y,1,&cu_res);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCdotc failed with error %d", error);
    return std::complex<float>(cu_res.x,cu_res.y);
  }
  template<>
  inline std::complex<double> dot<double>(int NN, const double *x, const double *y){
    cuDoubleComplex cu_res;
    cublasStatus_t error = cublasZdotc(HGC_cublas_handle, NN,(cuDoubleComplex*)x, 1, (cuDoubleComplex*)y,1,&cu_res);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasZdotc failed with error %d", error);
    return std::complex<double>(cu_res.x,cu_res.y);
  }  
  template<typename Float>
  inline std::complex<Float> dot(int NN, const Float *x, const Float *y, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) PLEGMA_error("Communicator is NULL and cannot be used for MPI reduction");
    std::complex<Float> result, res = cuBLAS::dot(NN, x, y);
    int mpiErr = MPI_Allreduce((Float*) &res, (Float*) &result, 2,
  			       MPI_Type<Float>(), MPI_SUM, comm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
    return result;
  }  
  //-----------------------------------------------------------------
  template<typename Float>
  inline Float norm(int NN, const Float *x);
  template<>
  inline float norm<float>(int NN, const float *x){
    float res;
    cublasStatus_t error = cublasScnrm2(HGC_cublas_handle, NN, (cuComplex*)x, 1, &res);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCdotc failed with error %d", error);
    return res;
  }
  template<>
  inline double norm<double>(int NN, const double *x){
    double res;
    cublasStatus_t error = cublasDznrm2(HGC_cublas_handle, NN, (cuDoubleComplex*)x, 1, &res);
    if(error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCdotc failed with error %d", error);
    return res;
  }
  template<typename Float>
  inline Float norm(int NN, const Float *x, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) PLEGMA_error("Communicator is NULL and cannot be used for MPI reduction");
    Float result, loc_res = cuBLAS::norm(NN, x);
    int mpiErr = MPI_Allreduce(&result, &loc_res, 1, MPI_Type<Float>(), MPI_SUM,
			       comm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
    return result;
  }
}
//=================================================================//

namespace plegma{
  template<typename Float>
  void elemWiseMul(int NN, Float* x, Float* y);

  template<typename Float>
  void axpbypcz(int NN, Float a[2], Float* x, Float b[2], Float* y, Float c[2], Float* z);
}
