/* cblas.h — minimal self-contained LP64 declarations for PLEGMA.
 *
 * Why: some systems (e.g. OpenBLAS 0.3.30 ILP64 build) ship a cblas.h that
 * only declares the 64-bit-integer variants (cblas_caxpy64_ etc.).  PLEGMA
 * uses only the 8 standard LP64 complex BLAS-1/2 routines listed below; we
 * declare them directly so that this header works with any LP64 BLAS library
 * (OpenBLAS, FlexiBLAS, MKL, …) without touching system headers.
 *
 * Linker side: set PLEGMA_OPENBLAS_LIB to any library that exports these
 * symbols (libopenblas.so, libflexiblas.so, libmkl_rt.so, …).
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_ORDER;
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;

/* BLAS-1 complex single */
void cblas_caxpy(int N, const void *alpha, const void *X, int incX, void *Y, int incY);
void cblas_cscal (int N, const void *alpha, void *X, int incX);
void cblas_csscal(int N, float alpha,        void *X, int incX);

/* BLAS-1 complex double */
void cblas_zaxpy(int N, const void *alpha, const void *X, int incX, void *Y, int incY);
void cblas_zscal (int N, const void *alpha, void *X, int incX);
void cblas_zdscal(int N, double alpha,       void *X, int incX);

/* BLAS-2 complex single */
void cblas_cgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                 int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *X, int incX,
                 const void *beta, void *Y, int incY);

/* BLAS-2 complex double */
void cblas_zgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                 int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *X, int incX,
                 const void *beta, void *Y, int incY);

#ifdef __cplusplus
}
#endif
