#include <hipblas.h>
#include <hip/hip_complex.h>
#pragma once
namespace cuBLAS{
  
  template<typename Float>
  inline void axpy(int NN, Float val[2], Float *x, Float *y){}
  template<>
  inline void axpy<float>(int NN, float val[2], float *x, float *y){
    hipblasComplex cu_val(val[0],val[1]);
    hipblasStatus_t error =  hipblasCaxpy(HGC.hipblas_handle,NN, &cu_val, (hipblasComplex*) x, 1, (hipblasComplex*) y, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCaxpy failed with error %d", error);
  }
  template<>
  inline void axpy<double>(int NN, double val[2], double *x, double *y){
    hipblasDoubleComplex cu_val(val[0],val[1]);
    hipblasStatus_t error =  hipblasZaxpy(HGC.hipblas_handle, NN, &cu_val,(hipblasDoubleComplex*) x, 1, (hipblasDoubleComplex*) y, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasZaxpy failed with error %d", error);
  }
  
  //------------------------------------------------------------------
  template<typename Float>
  inline void scal(int NN, const Float val, Float *x);
  template<>
  inline void scal<float>(int NN, const float val, float *x){
    hipblasStatus_t error = hipblasCsscal(HGC.hipblas_handle, NN, &val, (hipblasComplex*) x, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCsscal failed with error %d", error);
  }
  template<>
  inline void scal<double>(int NN, const double val, double *x){
    hipblasStatus_t error = hipblasZdscal(HGC.hipblas_handle, NN, &val, (hipblasDoubleComplex*) x, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasZdscal failed with error %d", error);
  }
  //------------------------------------------------------------------
  template<typename Float>
  inline void cscal(int NN, const Float val[2], Float *x);
  template<>
  inline void cscal<float>(int NN, const float val[2], float *x){
    hipblasComplex cu_val(val[0],val[1]);
    hipblasStatus_t error = hipblasCscal(HGC.hipblas_handle, NN, &cu_val, (hipblasComplex*) x, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCscal failed with error %d", error);
  }
  template<>
  inline void cscal<double>(int NN, const double val[2], double *x){
    hipblasDoubleComplex cu_val(val[0],val[1]);
    hipblasStatus_t error = hipblasZscal(HGC.hipblas_handle, NN, &cu_val, (hipblasDoubleComplex*) x, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasZscal failed with error %d", error);
  }
  //-----------------------------------------------------------------
  template<typename Float>
  inline std::complex<Float> dot(int NN, const Float *x, const Float *y);
  template<>
  inline std::complex<float> dot<float>(int NN, const float *x, const float *y){
    hipblasComplex cu_res;
    hipblasStatus_t error = hipblasCdotc(HGC.hipblas_handle, NN,(hipblasComplex*)x, 1, (hipblasComplex*)y,1,&cu_res);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCdotc failed with error %d", error);
    return std::complex<float>(cu_res.real(),cu_res.imag());
  }
  template<>
  inline std::complex<double> dot<double>(int NN, const double *x, const double *y){
    hipblasDoubleComplex cu_res;
    hipblasStatus_t error = hipblasZdotc(HGC.hipblas_handle, NN,(hipblasDoubleComplex*)x, 1, (hipblasDoubleComplex*)y,1,&cu_res);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasZdotc failed with error %d", error);
    return std::complex<double>(cu_res.real(),cu_res.imag());
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
    hipblasStatus_t error = hipblasScnrm2(HGC.hipblas_handle, NN, (hipblasComplex*)x, 1, &res);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCdotc failed with error %d", error);
    return res;
  }
  template<>
  inline double norm<double>(int NN, const double *x){
    double res;
    hipblasStatus_t error = hipblasDznrm2(HGC.hipblas_handle, NN, (hipblasDoubleComplex*)x, 1, &res);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCdotc failed with error %d", error);
    return res;
  }
  template<typename Float>
  inline Float norm(int NN, const Float *x, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) PLEGMA_error("Communicator is NULL and cannot be used for MPI reduction");
    Float result, loc_res = std::pow(cuBLAS::norm(NN, x),2);
    int mpiErr = MPI_Allreduce(&loc_res, &result, 1, MPI_Type<Float>(), MPI_SUM,
			       comm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
    return sqrt(result);
  }


  //---------------------------------------------------------
  template<typename Float>
  inline void gemv_(OPER_MATR_BLAS trans, int m, int n, Float alpha[2], Float* A, Float* x, Float beta[2], Float* y){}

  template<>
  inline void gemv_<float>(OPER_MATR_BLAS trans, int m, int n, float alpha[2], float* A, float* x, float beta[2], float* y){
    hipblasOperation_t Oper;
    hipblasComplex cu_alpha(alpha[0],alpha[1]);
    hipblasComplex cu_beta(beta[0],beta[1]);
    switch(trans){case(NOTRANS): Oper=HIPBLAS_OP_N; break; case(TRANS): Oper=HIPBLAS_OP_T; break; case(DAGGER): Oper=HIPBLAS_OP_C; break;}
    hipblasStatus_t error = hipblasCgemv(HGC.hipblas_handle, Oper, m, n, &cu_alpha, (hipblasComplex*) A, m, (hipblasComplex*) x, 1, &cu_beta, (hipblasComplex*) y, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCgemv failed with error %d", error);
  }

  template<>
  inline void gemv_<double>(OPER_MATR_BLAS trans, int m, int n, double alpha[2], double* A, double* x, double beta[2], double* y){
    hipblasOperation_t Oper;
    hipblasDoubleComplex cu_alpha(alpha[0],alpha[1]);
    hipblasDoubleComplex cu_beta(beta[0],beta[1]);
    switch(trans){case(NOTRANS): Oper=HIPBLAS_OP_N; break; case(TRANS): Oper=HIPBLAS_OP_T; break; case(DAGGER): Oper=HIPBLAS_OP_C; break;}
    hipblasStatus_t error = hipblasZgemv(HGC.hipblas_handle, Oper, m, n, &cu_alpha, (hipblasDoubleComplex*) A, m,(hipblasDoubleComplex*) x, 1, &cu_beta,(hipblasDoubleComplex*) y, 1);
    if(error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasZgemv failed with error %d", error);
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
  inline void gemv(OPER_MATR_BLAS trans, int m, int n, Float alpha[2], Float* A, Float* x, Float beta[2], Float* y, Float *yHost, MPI_Comm comm){
    if(trans == NOTRANS) PLEGMA_error("Use gemv without MPI comm");
    if(comm == MPI_COMM_NULL) PLEGMA_error("Communicator is NULL and cannot be used for MPI reduction");
    cuBLAS::gemv_(trans, m, n, alpha, A, x, beta, y);
    qudaMemcpy(yHost,y,n*2*sizeof(Float),qudaMemcpyDeviceToHost);
    checkQudaError();
    int mpiErr = MPI_Allreduce(MPI_IN_PLACE,yHost,n*2,MPI_Type(yHost),MPI_SUM,comm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Allreduce failed with error %d\n", mpiErr);
  }

  // In case that m is partitioned and we take trans of dagger then reduction is needed
  // Cannot work if n is partitioned
  template<typename Float>
  inline void gemv(OPER_MATR_BLAS trans, int m, int n, Float alpha[2], Float* A, Float* x, Float beta[2], Float* y, MPI_Comm comm){
    Float yHost[n*2];
    cuBLAS::gemv(trans,m, n, alpha, A, x, beta, y, yHost,comm);
    qudaMemcpy(y,yHost,sizeof(yHost),qudaMemcpyHostToDevice);
    checkQudaError();
  }
}
//=================================================================//

