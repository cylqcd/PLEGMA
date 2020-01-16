#include <PLEGMA_utils.h>
#include <PLEGMA_Vector.h>
#ifndef EIGSOLVER_H
#define EIGSOLVER_H

#ifdef HAVE_EIGENSOLVER

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

namespace plegma{

  struct  EigSolverParams{
    int NeV; // total number of eigenvalues & eigenvectors
    std::string spectrumPart; // available options for arpack are (SR,LR)
    bool isACC; // In case we want to use Polymonial accelerator
    int PolyDeg; // Order of the Polynomial
    double amin; // Low boundary for polymonial accelerator
    double amax; // High boundary for polynomial accelerator
    double tol;          // tolerance of the eigen solver
    int maxIters;        // maximum number of iterations for solver
#if defined(HAVE_ARPACK)
    int NkV; // Krylov space size should be > NeV
    std::string logFile; // path to the eigensolver log file
#elif defined(HAVE_PRIMME)
    int printLevel; // primme level of print (0 for no printing at all), (5, for printing everything)
    primme_preset_method primme_method; // method to eigenSolver
    /* Available methods
       PRIMME_DYNAMIC
       PRIMME_DEFAULT_MIN_TIME
       PRIMME_DEFAULT_MIN_MATVECS
       PRIMME_Arnoldi
       PRIMME_GD
       PRIMME_GD_plusK
       PRIMME_GD_Olsen_plusK
       PRIMME_JD_Olsen_plusK
       PRIMME_RQI
       PRIMME_JDQR
       PRIMME_JDQMR
       PRIMME_JDQMR_ETol
       PRIMME_STEEPEST_DESCENT
       PRIMME_LOBPCG_OrthoBasis
       PRIMME_LOBPCG_OrthoBasis_Window
    */
#else
#endif
  };

  class EigSolver{
  private:
    EigSolverParams p;
    bool verbose;
    int field_length;
    int size_per_Vec;
    size_t size_NeV;
#if defined(HAVE_ARPACK)
    size_t size_NkV;
#endif
    size_t bytes_per_Vec;
    size_t bytes_NeV;
#if defined(HAVE_ARPACK)
    size_t bytes_NkV;
#endif
  
    double *h_eigVecs;
    double *h_eigVals;
#if defined(HAVE_PRIMME)
    double *h_rnorms;
    primme_params primme_pars;
#endif
    std::vector< std::tuple<double,double,double,int> > evalsOrdered; // real, imag, residual, orderInd
#if defined(HAVE_ARPACK)  
    void applyOperator(double *out, double *in);
#endif
    void initEigSolver();
    void computeEigVecs();
    void computeEigVals();
    void print();
    void writeEigenVectors(std::string filenamePrefix);
    void readEigenVectors(std::string filenamePrefix);
  public:
    EigSolver(EigSolverParams params, QudaDslashType dslashType,bool isReadEigenVectors = false,
	      bool isWriteEigenVectors = false, std::string filenamePrefix = "", bool verbose=false);
    ~EigSolver();
    void projectVector(PLEGMA_Vector<double> &vecOut, PLEGMA_Vector<double> &vecIn);
    void projectVector(PLEGMA_Vector<double> &vec, int nvecs=0);
    void dumpEvalsVdagG5V(std::string filename);
    double* getEigVecs() const{return h_eigVecs;}
    std::vector< std::tuple<double,double,double,int> > getEigVals() const{return evalsOrdered;}
    int getSize_per_Vec() const{return size_per_Vec;}
    size_t getBytes_per_Vec() const{return bytes_per_Vec;}
  };
}
#endif /* HAVE_EIGENSOLVER */
#endif /* EIGSOLVER_H */
