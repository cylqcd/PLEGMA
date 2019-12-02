#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <algorithm>
#include <PLEGMA_BLAS.h>

using namespace plegma;
using namespace quda;

#ifdef HAVE_EIGENSOLVER

static quda::QUDA_dirac *dOp;
static PLEGMA_Vector<double> *d_in;
static PLEGMA_Vector<double> *d_out;
static PLEGMA_Vector<double> *tmp1;
static PLEGMA_Vector<double> *tmp2;
static int G_PolyDeg;
static bool G_isACC;
static double G_amin;
static double G_amax;

EigSolver::EigSolver(EigSolverParams params, QudaDslashType dslashType,bool isReadEigenVectors,bool isWriteEigenVectors,std::string filenamePrefix,
		     bool verbose):verbose(verbose),p(params),
				   h_eigVecs(nullptr),h_eigVals(nullptr)
#ifdef HAVE_QUDAEIG
				  ,dslashT(dslashType)
#endif
{
  if(!HGC_init_PLEGMA_flag) PLEGMA_error("Initialize PLEGMA first");
  if(isReadEigenVectors && isWriteEigenVectors) PLEGMA_warning("Read and write eigenvectors is a strange choice...");
  if(p.NeV <=0 ){
    PLEGMA_printf("Warning: Eigensolver instructed to use NeV=%d, skipping eigenvectors calculation\n",p.NeV);
    return;
  }
  G_PolyDeg = p.PolyDeg;
  G_isACC = p.isACC;
  G_amax = p.amax;
  G_amin = p.amin;
  
  field_length = N_SPINS * N_COLS;
  size_per_Vec = HGC_localVolume * field_length;
  size_NeV = ((size_t) p.NeV) * size_per_Vec;
  bytes_per_Vec = size_per_Vec * 2 * sizeof(double);
  bytes_NeV = size_NeV * 2 * sizeof(double);
#if defined(HAVE_ARPACK) || defined(HAVE_QUDAEIG)
  if(isReadEigenVectors) p.NkV = p.NeV;
  if(p.NkV < p.NeV) PLEGMA_error("The NkV should be larger equal than NeV");
  size_NkV = ((size_t) p.NkV) * size_per_Vec;
  bytes_NkV = size_NkV * 2 * sizeof(double);
#endif

#if defined(HAVE_ARPACK)
  hostMalloc(h_eigVecs,size_NkV*2*sizeof(double));
  if(!isReadEigenVectors) hostMalloc(h_eigVals,p.NkV*2*sizeof(double));
#elif defined(HAVE_PRIMME)
  hostMalloc(h_eigVecs,size_NeV*2*sizeof(double));
  if(!isReadEigenVectors) hostMalloc(h_eigVals,p.NeV*2*sizeof(double));
  if(!isReadEigenVectors) hostMalloc(h_rnorms,p.NeV*2*sizeof(double));
#elif defined(HAVE_QUDAEIG)//test app appars to use the full field...
  hostMalloc(q_eigVecs,((size_t) p.NeV)*sizeof(void *));
  hostMalloc(h_eigVecs,size_NeV*2*sizeof(double));
  for(int i=0; i < p.NeV; i++)
    q_eigVecs[i] = (void *) (h_eigVecs+i*size_per_Vec*2);
  if(!isReadEigenVectors){
    hostMalloc(h_eigVals,p.NeV*2*sizeof(double));
    q_eigVals=h_eigVals;
  }
#else
  PLEGMA_error("Not implemented");
#endif

  if(!isReadEigenVectors){
    d_in = new PLEGMA_Vector<double>(DEVICE);
    d_out = new PLEGMA_Vector<double>(DEVICE);
  }
  dOp = new QUDA_dirac(dslashType);
  tmp1 = new PLEGMA_Vector<double>(DEVICE);
  tmp2 = new PLEGMA_Vector<double>(DEVICE);
  // check the spectrumPart
  if(p.spectrumPart != "SR" && p.spectrumPart != "LR") PLEGMA_error("Allowed values for spectrum part are SR or LR");
  if(p.isACC){
    if(p.spectrumPart == "SR") p.spectrumPart = "LR";
    else if(p.spectrumPart == "LR") p.spectrumPart = "SR";
    else PLEGMA_error("Not implemented");
  }

  /* compute eigenvectors */
  if(!isReadEigenVectors){
    initEigSolver();
    if(verbose) print();
    computeEigVecs();printf("d12\n");
  }
  else readEigenVectors(filenamePrefix);
    
  if(isWriteEigenVectors) writeEigenVectors(filenamePrefix);

  computeEigVals();
  if(!isReadEigenVectors){
#if defined(HAVE_ARPACK)
    hostFree(h_eigVals,p.NkV*2*sizeof(double));
#elif defined(HAVE_PRIMME)
    hostFree(h_eigVals,p.NeV*2*sizeof(double));
#elif defined(HAVE_QUDAEIG)
    hostFree(h_eigVals,p.NeV*2*sizeof(double));
#else
  PLEGMA_error("Not implemented");
#endif    
    delete d_in;
    delete d_out;
    d_in = nullptr; d_out = nullptr;
  }
  delete dOp;
  delete tmp1;
  delete tmp2;
  tmp1 = nullptr; tmp2 = nullptr; dOp = nullptr;
#if defined(HAVE_PRIMME)
  if(!isReadEigenVectors){
    hostFree(h_rnorms,p.NeV*2*sizeof(double));
    primme_free(&primme_pars);
  }
#endif
  
}

EigSolver::~EigSolver(){
#if defined(HAVE_ARPACK)
    hostFree(h_eigVecs,size_NkV*2*sizeof(double));
#elif defined(HAVE_PRIMME)
    hostFree(h_eigVecs,size_NeV*2*sizeof(double));
#elif defined(HAVE_QUDAEIG)//might move to constructor and combine with ARPACK
    hostFree(q_eigVecs,p.NeV*sizeof(void *));
    hostFree(h_eigVecs,size_NeV*2*sizeof(double));
#else
  PLEGMA_error("Not implemented");
#endif    
  delete[] h_eigVecs;
}


#ifndef HAVE_QUDAEIG
#if defined(HAVE_PRIMME) && defined(HAVE_MAGMA)
//under construction
#elif defined(HAVE_PRIMME)
static void applyOperator(double *out, double *in, int size_per_Vec){
  size_t bytes_per_Vec = size_per_Vec * 2 * sizeof(double);
#else
void EigSolver::applyOperator(double *out, double *in){  
#endif
  cudaMemcpy(d_in->D_elem(),in,bytes_per_Vec,cudaMemcpyHostToDevice);
  checkCudaError();
  
  if(!G_isACC) dOp->apply<MdagM>(*d_out,*d_in);
  else{
    if(G_PolyDeg < 1) PLEGMA_error("Degree of the Polynomial shoud be >= 1");
    double delta,theta;
    double sigma,sigma1,sigma_old;
    std::complex<double> d1(0.,0.),d2(0.,0.),d3(0.,0.);
    double a = G_amin;
    double b = G_amax;
    delta = (b-a)/2.0;
    theta = (b+a)/2.0;
    sigma1 = -delta/theta;
    d1.real(sigma1/delta);
    d2.real(1.0);
    dOp->apply<MdagM>(*d_out,*d_in);
    cuBLAS::scal(size_per_Vec, d1.real(), d_out->D_elem() );
    cuBLAS::axpy(size_per_Vec, reinterpret_cast<double(&)[2]>(d2), d_in->D_elem(), d_out->D_elem());
    if(G_PolyDeg > 1){
      tmp1->copy(*d_in);
      tmp2->copy(*d_out);
      sigma_old = sigma1;
      for(int i=2; i <= G_PolyDeg; i++){
	sigma = 1.0/(2.0/sigma1-sigma_old);
	d1.real(2.0*sigma/delta);
	d2.real(-d1.real()*theta);
	d3.real(-sigma*sigma_old);
	dOp->apply<MdagM>(*d_out, *tmp2);
	plegma::axpbypcz(size_per_Vec,reinterpret_cast<double(&)[2]>(d3),tmp1->D_elem(),
			 reinterpret_cast<double(&)[2]>(d2),tmp2->D_elem(),
			 reinterpret_cast<double(&)[2]>(d1),d_out->D_elem());
	tmp1->copy(*tmp2);
	tmp2->copy(*d_out);
	sigma_old=sigma;
      }
    }
  }
  cudaMemcpy(out,d_out->D_elem(),bytes_per_Vec,cudaMemcpyDeviceToHost);
  checkCudaError();
}
#endif

#if defined(HAVE_PRIMME)
static void applyOperator(void *in, PRIMME_INT *ldx, void *out, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr){
  *blockSize=1;
  applyOperator((double*) out, (double*) in, primme->nLocal);
  *ierr=0;
}

static void par_GlobalSumForDouble(void *sendBuf, void *recvBuf, int *count, primme_params *primme, int *ierr) {
  MPI_Comm communicator = MPI_COMM_WORLD;	
  if (sendBuf == recvBuf)
    *ierr = MPI_Allreduce(MPI_IN_PLACE, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
  else
    *ierr = MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
}
#endif

void EigSolver::initEigSolver(){
#ifdef HAVE_ARPACK
  int arpack_log_u = 9999;
  if(!p.logFile.empty() && comm_rank() == 0){
    char *tmps = strdup(p.logFile.c_str());
    initlog_(&arpack_log_u, tmps, p.logFile.length());
    free(tmps);
    int msglvl0 = 0;//, msglvl1 = 1, msglvl2 = 2, msglvl3 = 3;
    int msglvl3 = 3;
    pmcinitdebug_(&arpack_log_u, &msglvl3, &msglvl3, &msglvl0, &msglvl3, &msglvl0, &msglvl0, &msglvl3);
  }
#elif HAVE_PRIMME
#ifdef HAVE_MAGMA
  /* Initialize MAGMA and create some LA structures */
  magma_init();
  magma_queue_t queue;
  magma_queue_create(0, &queue);
  
  magma_d_matrix A={Magma_CSR}, dA={Magma_CSR};
  
  /* Pass the matrix to MAGMA and copy it to the GPU */
  // row, col, val for TM Dirac needs to be specified in CSR format.
  magma_dcsrset(n, n, row, col, val, &A, queue);
  magma_dmtransfer(A, &dA, Magma_CPU, Magma_DEV, queue);
#endif
  /* Set default values in PRIMME configuration struct */
  primme_initialize(&primme_pars);
  /* Set problem parameters */
  primme_pars.matrixMatvec = applyOperator;
  MPI_Comm commPRIMME = MPI_COMM_WORLD;
  primme_pars.commInfo=&commPRIMME;
  primme_pars.globalSumReal=par_GlobalSumForDouble;
  primme_pars.numProcs=HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3];
  primme_pars.procID=comm_rank();
  primme_pars.nLocal = size_per_Vec;
  primme_pars.n = size_per_Vec * HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3];
  primme_pars.numEvals = p.NeV;
  primme_pars.eps = p.tol;
  primme_pars.maxOuterIterations = p.maxIters;
  /*  Tuning */
  primme_pars.correctionParams.robustShifts = 1; // led to faster convergence with 0 for tol=1e-8
  primme_pars.locking=1;
  /* James and Xiao-Yong's optimal setting */
  if(primme_pars.locking != 0){ /* relevant if locking != 0 */
    primme_pars.minRestartSize=120;
    primme_pars.maxBasisSize=192;
  }
  primme_pars.maxBlockSize=8;
  primme_pars.restartingParams.maxPrevRetain=2;
  /*  END: Tuning  */
  if(p.spectrumPart == "SR") primme_pars.target = primme_smallest;
  else if(p.spectrumPart == "LR") primme_pars.target = primme_largest;
  else PLEGMA_error("Not implemented");
  primme_pars.printLevel = p.printLevel;
  primme_set_method(p.primme_method, &primme_pars);
#elif defined(HAVE_QUDAEIG)
  // It could have been already checked...
  if(dslashT != QUDA_WILSON_DSLASH
     && dslashT != QUDA_CLOVER_WILSON_DSLASH
     && dslashT != QUDA_TWISTED_MASS_DSLASH
     && dslashT != QUDA_TWISTED_CLOVER_DSLASH) PLEGMA_error("Error dslashType is not allowed in PLEGMA");

  eig_param = newQudaEigParam();
  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator. We encapsualte the
  // inv_param structure inside the eig_param structure
  // to avoid any confusion
  eig_inv_param = newQudaInvertParam();
  //cpu_prec = QUDA_DOUBLE_PRECISION;  //This is not necessary.
  setInvertParam(eig_inv_param);
  eig_inv_param.dslash_type = dslashT;
  eig_inv_param.solve_type=QUDA_DIRECT_SOLVE;
  eig_param.invert_param = &eig_inv_param;
  setQUDAEigParam(eig_param,p);
#else
  PLEGMA_error("Not implemented")
#endif
}

void EigSolver::print(){
  PLEGMA_printf("Number of eigenvalues requested: %d\n",p.NeV);
#if defined(HAVE_ARPACK)
  PLEGMA_printf("Size of Krylov space requested: %d\n",p.NkV);
#endif
  if(p.isACC) PLEGMA_printf("Using Polynomial acceleration with parameters: Degree=%d, amin=%+e, amax=%+e\n",p.PolyDeg,p.amin,p.amax);
  PLEGMA_printf("Part of the spectrum to be computed: %s",p.spectrumPart.c_str());
  if(p.isACC){ PLEGMA_printf("\n Flipped due to polynomial acceleration\n");}
  else{ PLEGMA_printf("\n");}
  PLEGMA_printf("Tolerance for eigenSolver is %+e\n",p.tol);
  PLEGMA_printf("Max number of iterations for eigenSolver is %d\n",p.maxIters);
#if defined(HAVE_ARPACK)
  PLEGMA_printf("The eigensolver logfile is: %s\n",p.logFile.c_str());
#endif
#if defined(HAVE_PRIMME)
  primme_display_params(primme_pars);
#endif
}

void EigSolver::computeEigVecs(){
#if defined(HAVE_ARPACK)
  MPI_Fint mpi_comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
  char *bmat = strdup("I");
  int rvec = 1;
  int lworkl = (3*p.NkV*p.NkV+5*p.NkV)*2;
  std::complex<double> sigma;
  char *howmany = strdup("P");
  char *which_evals = strdup(p.spectrumPart.c_str());
  std::complex<double> * pin=nullptr;
  std::complex<double> * pout=nullptr;
  int *ipntr, *select, *iparam; 
  double *rwork; 
  std::complex<double> *resid, *workd, *workl, *workev;
  try{
    ipntr              = new int[14];
    select             = new int[p.NkV];
    iparam             = new int[11];
    rwork        = new double[p.NkV];
  
    resid  = new std::complex<double>[size_per_Vec];
    workd  = new std::complex<double>[3*size_per_Vec];
    workl  = new std::complex<double>[lworkl];
    workev = new std::complex<double>[2*p.NkV];
  }
  catch (std::bad_alloc& err){
    PLEGMA_error(err.what());
  }
  
  iparam[0] = 1;  //use exact shifts
  iparam[2] = p.maxIters;
  iparam[3] = 1;
  iparam[6] = 1;
  iparam[7] = 1; // arpack mode

  int info = 0; // random initial guess
  // int info = 1;
  // for (int i = 0; i < size_per_Vec; ++i) {
  //   resid[i].real(1.);
  //   resid[i].imag(0.);
  // }

  int nconv;
  bool checkIdo = true;
  int arpack_iter = 0;
  int ido = 0;

  // start solver
  do{
    pznaupd_(&mpi_comm_f, &ido,bmat, &size_per_Vec, which_evals, 
	     &p.NeV, &p.tol, resid, &p.NkV,
	     (std::complex<double>*)h_eigVecs, &size_per_Vec, iparam, ipntr, workd, 
	     workl, &lworkl,rwork,&info,1,2);
    if(checkIdo){
      pin=workd+ipntr[0]-1;
      pout=workd+ipntr[1]-1;
      checkIdo = false;
    }
    if (ido == 99 || info == 1)break;
    if( (ido==-1) || (ido==1) ) applyOperator((double*)pout, (double*)pin);
    if(verbose)PLEGMA_printf("arpack iter = %d\n", arpack_iter);
    arpack_iter++;
  }while(ido != 99);
    
  if(info < 0) PLEGMA_error("EigSolver error: info = %d\n",info);
  nconv = iparam[4];
  if(nconv != p.NeV) PLEGMA_error("Converged number of eigvalues %d != %d\n", nconv, p.NeV);
  if(verbose) PLEGMA_printf("EigSolver: Number of converged eigenvalues: %d\n", nconv);

  // compute eigenvectors
  pzneupd_(&mpi_comm_f,&rvec,howmany, select, (std::complex<double>*) h_eigVals,(std::complex<double>*) h_eigVecs, &size_per_Vec,&sigma, 
	   workev,bmat,&size_per_Vec,which_evals,&p.NeV,&p.tol, resid,&p.NkV, 
	   (std::complex<double>*) h_eigVecs,&size_per_Vec,iparam,ipntr,workd,workl,&lworkl,rwork,&info,1,1,2);
  if(info == 1) PLEGMA_printf("Warning: Maximum number of iterations reached.\n");
  if(info == 3) PLEGMA_error("No shifts could be applied during implicit, Arnoldi update, try increasing NkV\n");
  int arpack_log_u = 9999;
  if(!p.logFile.empty() && comm_rank() == 0)finilog_(&arpack_log_u);

  free(bmat);
  free(howmany);
  free(which_evals);
  delete[] ipntr;
  delete[] select;
  delete[] iparam;
  delete[] rwork;
  delete[] resid;
  delete[] workd;
  delete[] workl;
  delete[] workev;

#elif defined(HAVE_PRIMME)
  zprimme(h_eigVals, (std::complex<double> *) h_eigVecs, h_rnorms, &primme_pars);
#elif defined(HAVE_QUDAEIG)
  // Thick Restarted Lanczos Eigensolver

  // This function returns the host_evecs and host_evals pointers, populated with the
  // requested data, at the requested prec. All the information needed to perfom the
  // solve is in the eig_param container. If eig_param.arpack_check == true and
  // precision is double, the routine will use ARPACK rather than the GPU.
  double time = -((double)clock());
  //printQudaInvertParam(eig_param.invert_param);
  eigensolveQuda(q_eigVecs, (double _Complex*) q_eigVals, &eig_param);
  //for(int i=0; i<p.NeV; i++) memcpy(h_eigVals+i*2*sizeof(double),q_eigVals+i,sizeof(double _Complex));
  time += (double)clock();
  PLEGMA_printf("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", time / CLOCKS_PER_SEC);
#else
  PLEGMA_error("Not implemented");
#endif
}

void EigSolver::computeEigVals(){
  double* ptr_tmp = h_eigVecs;
  for(int j = 0 ; j < p.NeV; j++){
    double one[2] = {1.,0.};
    cudaMemcpy(tmp1->D_elem(),ptr_tmp,bytes_per_Vec,cudaMemcpyHostToDevice);
    checkCudaError();
    dOp->apply<MdagM>(*tmp2,*tmp1);
    std::complex<double> eval = cuBLAS::dot(size_per_Vec, tmp1->D_elem(), tmp2->D_elem(), MPI_COMM_WORLD);
    cuBLAS::scal(size_per_Vec,-eval.real(),tmp1->D_elem());
    cuBLAS::axpy(size_per_Vec,one,tmp2->D_elem(),tmp1->D_elem());
    std::complex<double> res = cuBLAS::dot(size_per_Vec, tmp1->D_elem(), tmp1->D_elem(), MPI_COMM_WORLD);
    evalsOrdered.push_back(std::make_tuple(eval.real(), eval.imag(), std::sqrt(res.real()), j));
    ptr_tmp += size_per_Vec*2;
  }
  std::sort(evalsOrdered.begin(), evalsOrdered.end());
  if(verbose)
    for (int j = 0; j < p.NeV; ++j)
      PLEGMA_printf("Eval[%04d] = (%+e,%+e), Residual: %+e, Order Index: %d\n", j, std::get<0>(evalsOrdered[j]), std::get<1>(evalsOrdered[j]),
		 std::get<2>(evalsOrdered[j]), std::get<3>(evalsOrdered[j]));
}

// vecOut = (1 - U * U^\dag) vecIn
void EigSolver::projectVector(PLEGMA_Vector<double> &vecOut, PLEGMA_Vector<double> &vecIn){
  if(p.NeV <= 0){
    if(verbose) PLEGMA_printf("Skipping deflation of source vector since NeV=%d\n",p.NeV);
    vecOut.copy(vecIn);
    return;
  }
  if(!vecOut.IsAllocHost() || !vecIn.IsAllocHost()) PLEGMA_error("This functions needs both vecs to have also Host allocation");
  vecIn.unload();
  double aP[2]={1.,0.}, b[2]={0.,0.}, aM[2]={-1.,0.};
  double *tmpArr = nullptr;
  try { tmpArr = new double[p.NeV*2]; } catch (std::bad_alloc &err) { PLEGMA_error(err.what());}
  memset(tmpArr,0,p.NeV*2*sizeof(double));
  cBLAS::gemv(DAGGER, size_per_Vec, p.NeV, aP, h_eigVecs, vecIn.H_elem(), b, tmpArr, MPI_COMM_WORLD);
  cBLAS::gemv(NOTRANS, size_per_Vec, p.NeV, aM, h_eigVecs, tmpArr, b, vecOut.H_elem());
  cBLAS::axpy(size_per_Vec, aP, vecIn.H_elem(), vecOut.H_elem());
  vecOut.load();
  delete[] tmpArr;
}


 // vecOut = (1 - U * U^\dag) vecIn where out and in are the same
void EigSolver::projectVector(PLEGMA_Vector<double> &vec){
  if(p.NeV <= 0){
    if(verbose) PLEGMA_printf("Skipping deflation of source vector since NeV=%d\n",p.NeV);
    return;
  }
  if(!vec.IsAllocHost()) PLEGMA_error("This functions needs vec to have also Host allocation");
  vec.unload();
  double aP[2]={1.,0.}, b[2]={0.,0.}, aM[2]={-1.,0.};
  double *tmpArr = nullptr;
  try { tmpArr = new double[p.NeV*2]; } catch (std::bad_alloc &err) { PLEGMA_error(err.what());}
  memset(tmpArr,0,p.NeV*2*sizeof(double));
  cBLAS::gemv(DAGGER, size_per_Vec, p.NeV, aP, h_eigVecs, vec.H_elem(), b, tmpArr, MPI_COMM_WORLD);
  cBLAS::gemv(NOTRANS, size_per_Vec, p.NeV, aM, h_eigVecs, tmpArr, aP, vec.H_elem());
  vec.load();
  delete[] tmpArr;
}

void EigSolver::dumpEvalsVdagG5V(std::string filename){
  if(p.NeV <= 0){ PLEGMA_printf("Skipping dumping of evals v^+ g5 v since NeV=%d\n",p.NeV); return;}
  PLEGMA_Vector<double> g5V(DEVICE);
  PLEGMA_Vector<double> V(DEVICE);
  std::vector<double> VdagG5V;
  double* ptr_tmp = h_eigVecs;
  for (int j = 0; j < p.NeV; ++j) {
    cudaMemcpy(g5V.D_elem(),ptr_tmp,bytes_per_Vec,cudaMemcpyHostToDevice);
    checkCudaError();
    V.copy(g5V);
    g5V.apply_gamma(G5);
    std::complex<double> res = cuBLAS::dot(size_per_Vec, V.D_elem(), g5V.D_elem(),MPI_COMM_WORLD);
    VdagG5V.push_back(res.real());
    ptr_tmp += size_per_Vec*2;
  }
  if(comm_rank() == 0){
    FILE *ptr = fopen(filename.c_str(), "w");
    if(ptr == NULL) PLEGMA_error("Cannot open file:%s for writting\n",filename.c_str());
    for (int i = 0; i < p.NeV; ++i) fprintf(ptr, "%+e %+e\n", std::get<0>(evalsOrdered[i]), VdagG5V[std::get<3>(evalsOrdered[i])]);
    fclose(ptr);
  }
}

 void EigSolver::readEigenVectors(std::string filenamePrefix){
   if(filenamePrefix.empty()) PLEGMA_error("Filename for eigenVectors is empty");
   PLEGMA_Vector<double> tmp(HOST);
   for(int i = 0 ; i < p.NeV; i++){
     double *eigVec = h_eigVecs + ((long int) i) * size_per_Vec*2;
     tmp.readFile(filenamePrefix + "_eV" + std::to_string(i),LIME_FORMAT);
     memcpy(eigVec,tmp.H_elem(),bytes_per_Vec);
     if(verbose) PLEGMA_printf("Eigenvector %d loaded\n", i);
   }
 }

 void EigSolver::writeEigenVectors(std::string filenamePrefix){
   if(filenamePrefix.empty()) PLEGMA_error("Filename for eigenVectors is empty");
   PLEGMA_Vector<double> tmp(HOST);
   for(int i = 0 ; i < p.NeV; i++){
     double *eigVec = h_eigVecs + ((long int) i) * size_per_Vec*2;
     memcpy(tmp.H_elem(),eigVec,bytes_per_Vec);
     tmp.writeFile(filenamePrefix + "_eV" + std::to_string(i), LIME_FORMAT);
     if(verbose) PLEGMA_printf("Eigenvector %d writen\n", i);
   }   
 }
#endif
