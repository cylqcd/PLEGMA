#include <PLEGMA_utils.h>
#include <PLEGMA_Vector.h>
#ifndef PLEGMA_EIGSOLVER_H
#define PLEGMA_EIGSOLVER_H

#if defined(HAVE_ARPACK) && defined(HAVE_PRIMME)
#error Cannot have both ARPACK and PRIMME
#endif

#if defined(HAVE_PRIMME)
#include <primme.h>
#elif defined(HAVE_ARPACK)
extern "C"{
  extern int initlog_(int*, char*, int);
  extern int finilog_(int*);
  extern int pznaupd_(int *comm, int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
			    std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, 
			    int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, 
			    int *lworkl, double *rwork, int *info, int bmat_size, int which_size );
  extern int pzneupd_(int *comm, int *comp_evecs, char *howmany, int *select, std::complex<double> *evals, 
			    std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev, 
			    char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, 
			    int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, 
			    std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info,
			    int howmany_size, int bmat_size, int which_size);
  extern int pmcinitdebug_(int*,int*,int*,int*,int*,int*,int*,int*);
}
#else
#error Neither PRIMME nor ARPACK have been defined
#endif

struct  EigSolverParams{
  int NeV; // total number of eigenvalues & eigenvectors
  int NkV; // Krylov space size > NeV

  bool isACC; // In case we want to use Polymonial accelarator
  int PolyDeg; // Order of the Polynomial
  double amin; // Low boundary for polymonial accelator
  double amax; // High boundary for polynomial accelator
#if defined(HAVE_ARPACK)
  std::string spectrumPart; // available options for arpack are (SR,LR)
  double tol;
  int maxIters;
  int mode; 
#elif defined(HAVE_PRIMME)
  
#else
#endif
};

class PLEGMA_EigSolver{
 private:
  EigSolverParams p;
  bool verbose;
  int field_length;
  size_t size_per_Vec;
  size_t size_NeV;
  size_t size_NkV;
  size_t size_total;
  size_t bytes_per_Vec;
  size_t bytes_NeV;
  size_t bytes_NkV;
  size_t bytes_total;
  
  double *h_eigVecs;
  double *h_eigVals;

  quda::QUDA_dirac *dOp;

  PLEGMA_Vector<double> *d_in;
  PLEGMA_Vector<double> *d_out;
  PLEGMA_Vector<double> *tmp1;
  PLEGMA_Vector<double> *tmp2;

  void applyOperator(double *out, double *in);
  void initEigSolver();
  void computeEigVecs();
  void computeEigVals();
  void print();
 public:
  PLEGMA_EigSolver(EigSolverParams params, QudaDslashType dslashType, bool verbose=false);
  virtual ~PLEGMA_EigSolver();
  void projectVector(PLEGMA_Vector<double> &vecOut, PLEGMA_Vector<double> &vecIn);
};
#endif /* PLEGMA_EIGSOLVER_H */
