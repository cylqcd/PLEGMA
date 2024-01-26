#include <PLEGMA_utils.h>
#include <PLEGMA_Vector.h>
#ifndef EIGSOLVER_H
#define EIGSOLVER_H

#ifdef HAVE_EIGENSOLVER

#if defined(HAVE_ARPACK) && defined(HAVE_PRIMME)
#error Cannot have both ARPACK and PRIMME
#endif

#if defined(QUDAEIG) && defined(HAVE_ARPACK)
#error Cannot have both QUDAEIG and ARPACK
#endif

#if defined(QUDAEIG) && defined(HAVE_PRIMME)
#error Cannot have both QUDAEIG and PRIMME
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
#elif defined(QUDAEIG)
#include <quda.h>
#else
#error Neither PRIMME, ARPACK or QUDAEIG have been defined
#endif

namespace plegma{

  struct  EigSolverParams{
    int NeV; // total number of eigenvalues & eigenvectors
    std::string spectrumPart; // available options for arpack are (SR,LR)
    bool littleD; // In case we want to compute little Dirac and use it in the projection
    bool isACC; // In case we want to use Polymonial accelerator
    int PolyDeg; // Order of the Polynomial
    double amin; // Low boundary for polymonial accelerator
    double amax; // High boundary for polynomial accelerator
    double tol;          // tolerance of the eigen solver
    int maxIters;        // maximum number of iterations for solver
#if defined(QUDAEIG)
    int batched_rotate; // batched size of TRLM. Set 1 for small memory need but loose of performance
#endif
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
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
    size_t size_per_Vec;
    size_t size_NeV;
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
    size_t size_NkV;
#endif
    size_t bytes_per_Vec;
    size_t bytes_NeV;
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
    size_t bytes_NkV;
#endif
#if defined(QUDAEIG)
    QudaInvertParam eig_inv_param;
    QudaEigParam eig_param;
    double **h_eigVecs_p;
#endif
    double *h_eigVecs;
    double *h_eigVals;
#if defined(HAVE_PRIMME)
    double *h_rnorms;
    primme_params primme_pars;
#endif
    std::complex<double> *littleD;
    std::complex<double> *littleD_inv;
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
    void projectVector(PLEGMA_Vector<double> &vec, double* spinVals=nullptr, int global_t=-1, int spin=-1, int col=-1);
    void dumpEvalsVdagG5V(std::string filename);
    double* getEigVecs() const{return h_eigVecs;}
    std::complex<double>* getLittleD() const{return littleD;}
    std::vector< std::tuple<double,double,double,int> > getEigVals() const{return evalsOrdered;}
    int getSize_per_Vec() const{return size_per_Vec;}
    size_t getBytes_per_Vec() const{return bytes_per_Vec;}
  };
}
#endif /* HAVE_EIGENSOLVER */
#endif /* EIGSOLVER_H */
