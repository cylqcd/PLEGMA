#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_BLAS.h>

static std::vector<double> runtime;
#define TIC()  runtime.push_back(MPI_Wtime())
#define TOC(str)  PLEGMA_printf("TIME for %s %f sec\n", str, MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

#define TIME(fnc)  TIC(); fnc;	TOC(#fnc)

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
					   "Eig-NkV", "Eig-logFile",
#endif 
            "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					  "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks", "Projs", "threep-filename"};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::vector<double> mu_s;
  std::vector<double> mu_c;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int nsmearGauss_c = 0;
  int startSource = 0;
  std::string prOrNt = "neutron";
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
    options.set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
	};

  std::string Eig_outputFile = "./eigsVdagG5V.dat";
  HGC_options->set("Eig-outputFile", "Path to dump the eigenvalues and vdag g5 v if low-modes-recon is enabled",verbosity, Eig_outputFile);
  bool isReadEigenVecs = false, isWriteEigenVecs = false, isDeviceEigenVecs = false;
  std::string fnameEigenVecsPrefix="";
  HGC_options->set("deviceEigenVectors", "Where we want to store EigenVectors on device", verbosity, isDeviceEigenVecs);
  HGC_options->set("readEigenVectors", "Where we want to read EigenVectors from file", verbosity, isReadEigenVecs);
  HGC_options->set("writeEigenVectors", "Where we want to read EigenVectors from file", verbosity, isWriteEigenVecs);
  HGC_options->set("prefixEigenVecsFile", "Path with prefix for the filenames of the eigenvectors", verbosity, fnameEigenVecsPrefix);
  #ifdef QUDAEIG
    int batched_rotate = 1;
    HGC_options->set("batched-rotate", "The size of the batch during Ritz rotation", verbosity, batched_rotate);
  #endif
  std::vector<double> mus;
  HGC_options->set("extra-mu", "List of additional mu to run", verbosity, mus);
  std::vector<double> nevs;
  HGC_options->set("extra-nev", "List of additional nev to run", verbosity, nevs);
  bool fastio = false;
  HGC_options->set("fastio", "Whether eigenvectors are read using fastio", verbosity, fastio);
  bool single_prec = false;
  HGC_options->set("single-prec", "Whether to use single precision eigenvectors", verbosity, single_prec);

  add_options(*HGC_options);
  if(prOrNt != "proton" && prOrNt != "neutron") PLEGMA_error("This exec is only for nucleon, %s is not allowed",prOrNt.c_str());
  //=========================================================================================================//
  TIME(initializePLEGMA());

  nevs.insert(nevs.begin(), Eig_NeV);
  for(int inev=1; inev < nevs.size(); inev++){ // Loop over NeV
    if(nevs[inev]>nevs[0]) PLEGMA_error("extra-nev can only be smaller");
  }

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();
      
      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();

      // Gauge for contractions
      contractGauge.copy(gauge);
      // apply boundary conditions since is needed for the covariant derivative
      applyBoundaryConditions(contractGauge,true);
   }

    EigSolver *eigSol = nullptr;
    if(Eig_NeV>0) {
      #if defined(HAVE_EIGENSOLVER)
        EigSolverParams eigParam;
        eigParam.NeV = Eig_NeV;
        eigParam.isACC = Eig_isACC;
        eigParam.littleD = true;
        eigParam.fastio = fastio;
        eigParam.single_prec = single_prec;
        eigParam.deviceAlloc = isDeviceEigenVecs;
        eigParam.PolyDeg = Eig_PolyDeg;
        eigParam.amin = Eig_amin;
        eigParam.amax = Eig_amax;
        eigParam.spectrumPart = Eig_spectrumPart;
        eigParam.tol = Eig_tol;
        eigParam.maxIters = Eig_maxIters;
        #ifdef QUDAEIG
          eigParam.batched_rotate = batched_rotate;
        #endif
        #if defined(HAVE_ARPACK) || defined(QUDAEIG)
          eigParam.NkV = Eig_NkV;
          eigParam.logFile = Eig_logFile;
        #else
          PLEGMA_error("No arpack or primme is compiled");
        #endif
        TIME(eigSol = new EigSolver(eigParam, dslash_type, isReadEigenVecs, isWriteEigenVecs, fnameEigenVecsPrefix, true));
      #else
        PLEGMA_error("No eigenSolver is compiled");
      #endif
    }


    // updateOptions(LIGHT);
    // TIME(QUDA_solver solver(mu));

    std::string given_twop_filename = twop_filename;
    std::string given_threep_filename = threep_filename;

    double aP[2]={1.,0.}, b[2]={0.,0.}, aM[2]={-1.,0.};
	  size_t size_per_Vec = eigSol->getSize_per_Vec();
    
    // Smear the eigenvectors but also keep track of unsmeared ones
    size_t vec_size = size_per_Vec*2*sizeof(double);
    double *eigVecs_d = eigSol->getEigVecs();
    double *eigVecs_hL = eigSol->getHEigVecs();
    PLEGMA_Vector<double> tmp1;
    PLEGMA_Vector<double> tmp2;
    PLEGMA_Vector<double> tmp3(NONE);
    PLEGMA_printf("\n ### Smearing the eigenvectors ###\n\n");
    for(int i=0; i<Eig_NeV; i++){
      tmp3.D_elem(eigSol->getEigVecs()+i*size_per_Vec*2);
      tmp1.copy(tmp3);
      TIME(tmp2.gaussianSmearing(tmp1, smearedGauge, nsmearGauss, alphaGauss));
      tmp3.copy(tmp2);
    }
    double *eigVecs_hS;
    eigVecs_hS = (double*)malloc(Eig_NeV*vec_size);
    cudaMemcpy(eigVecs_hS, eigVecs_d, Eig_NeV*vec_size, cudaMemcpyDeviceToHost);
    checkCudaError();
    std::complex<double> spinEVals[12*Eig_NeV];
    double *spinEVals_d;
    size_t V4 = HGC_localVolume;
    size_t V3 = V4/HGC_localL[3];
    cudaMalloc((void**)&spinEVals_d, 12*2*Eig_NeV*sizeof(double));
    checkCudaError();
    std::complex<double> evals[Eig_NeV];
    for(int i=0; i<Eig_NeV; i++) {
      evals[i] = eigSol->getLittleD()[i*(Eig_NeV+1)];
    }
    PLEGMA_Vector<double> vec;

    std::complex<double>tmp[Eig_NeV];
    double spinVals[Eig_NeV*2];
    double *spinVals_d, *evecs_d;
    cudaMalloc((void**)&spinVals_d, 2*Eig_NeV*sizeof(double));
    checkCudaError();
    cudaMalloc((void**)&evecs_d, 12*2*Eig_NeV*V3*sizeof(double));
    checkCudaError();

    PLEGMA_Propagator<float> propUP;
    PLEGMA_Propagator<float> propDN;
    PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE);
    PLEGMA_Propagator<float> propDN_SL(tSinks.size()>0 ? BOTH:NONE);

    PLEGMA_Propagator3D<float> prop13D;
    PLEGMA_Propagator3D<float> prop23D;
    PLEGMA_Vector3D<double> vectorAuxD;
    PLEGMA_Vector3D<float> vectorAuxF3D;
    PLEGMA_Vector<float> vectorAuxF;
    std::complex<double> *vals;
    PLEGMA_Propagator<float> seqProp;
    
    
    for(int isource = startSource; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

      bool all_exist=true;
      for(int inev=0; inev < nevs.size(); inev++){ // Loop over NeV
        int nev = nevs[inev];
      
        char * src_string;
        asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d_nev%03d", source[0], source[1], source[2], source[3], nev);
        twop_filename = given_twop_filename + src_string + ".h5";
        threep_filename = given_threep_filename + src_string;
        free(src_string);
        if(access( twop_filename.c_str(), F_OK ) == -1)
            all_exist=false;
      }
      if(all_exist) continue;

      cudaMemcpy(eigVecs_d, eigVecs_hS, Eig_NeV*vec_size, cudaMemcpyHostToDevice);
      checkCudaError();
      // create prop up and dn
      memset(spinEVals, 0, 12*Eig_NeV*2*sizeof(double));
      cudaMemset(spinEVals_d, 0, 12*Eig_NeV*2*sizeof(double));
      checkCudaError();
      int my_src[N_DIMS];
      bool mine = true;
      size_t id=0;
      for(int i = N_DIMS-1; i >= 0; i--) {
        my_src[i] = (source[i] - HGC_procPosition[i] * HGC_localL[i]);

        // if out of the local lattice we break
        if((my_src[i]<0) || (my_src[i]>=HGC_localL[i])) mine=false;

        id = id * HGC_localL[i] + my_src[i];
      }
      // This makes it work also for vector3D
      //id = id % this->Total_length();

      if(mine) {
        for(int ivec = 0; ivec < Eig_NeV; ivec++){
          for(int spin = 0; spin < N_SPINS; spin++){
            for(int color = 0; color < N_COLS; color++){
              cudaMemcpy(spinEVals_d + (ivec*N_SPINS*N_COLS + spin*N_COLS+color)*2, eigVecs_d + (ivec*N_SPINS*N_COLS*V4 + (((spin+2)%N_SPINS)*N_COLS+color)*V4 + id)*2, 2*sizeof(double), cudaMemcpyDeviceToDevice); //((spin+2)%N_SPINS) because of gamma_5 
              checkCudaError();
            }
          }
        }
        cudaMemcpy(spinEVals, spinEVals_d, 12*Eig_NeV*2*sizeof(double), cudaMemcpyDeviceToHost);
	      checkCudaError();
      }
      MPI_Allreduce(MPI_IN_PLACE,spinEVals,12*Eig_NeV*2,MPI_DOUBLE,MPI_SUM,HGC_fullComm);

      for(int inev=0; inev < nevs.size(); inev++){ // Loop over NeV
        PLEGMA_printf("Allocating device memory");
        int nev = nevs[inev];

        char * src_string;
        asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d_nev%03d", source[0], source[1], source[2], source[3], nev);
        twop_filename = given_twop_filename + src_string + ".h5";
        threep_filename = given_threep_filename + src_string;
        free(src_string);

        auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS_UP, PLEGMA_Propagator<float>& prop_SL_UP, PLEGMA_Propagator<float>& prop_SS_DN, PLEGMA_Propagator<float>& prop_SL_DN, bool finalize) { //What about WHICHFLAVOR fl? How does choosing a different flavor affect the computation?
          // PLEGMA_Gauge3D<double> smearedGauge3D;
          // smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

          // Inverting
          cudaMemcpy(eigVecs_d, eigVecs_hS, Eig_NeV*vec_size, cudaMemcpyHostToDevice);
          checkCudaError();
          for(int spin = 0; spin < N_SPINS; spin++){
            for(int color = 0; color < N_COLS; color++){
              for(int ivec = 0; ivec < nev; ivec++){
                tmp[ivec] = std::conj(spinEVals[ivec*N_SPINS*N_COLS + spin*N_COLS + color])/evals[ivec];
              }
              cudaMemcpy(spinEVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
              checkCudaError();

              cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aM, eigVecs_d, spinEVals_d, b, vec.D_elem());
              // PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(vec);
              prop_SS_UP.absorb(vectorAuxF, spin, color);
            }
          }

          for(int spin = 0; spin < N_SPINS; spin++){
            for(int color = 0; color < N_COLS; color++){
              for(int ivec = 0; ivec < nev; ivec++){
                tmp[ivec] = std::conj(spinEVals[ivec*N_SPINS*N_COLS + spin*N_COLS+color])/std::conj(evals[ivec]);
              }
              cudaMemcpy(spinEVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
              checkCudaError();

              cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aM, eigVecs_d, spinEVals_d, b, vec.D_elem());
              // PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(vec);
              prop_SS_DN.absorb(vectorAuxF, spin, color);
            }
          }

          cudaMemcpy(eigVecs_d, eigVecs_hL, Eig_NeV*vec_size, cudaMemcpyHostToDevice);
          checkCudaError();
          for(int spin = 0; spin < N_SPINS; spin++){
            for(int color = 0; color < N_COLS; color++){
              for(int ivec = 0; ivec < nev; ivec++){
                tmp[ivec] = std::conj(spinEVals[ivec*N_SPINS*N_COLS + spin*N_COLS + color])/evals[ivec];
              }
              cudaMemcpy(spinEVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
              checkCudaError();

              cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aM, eigVecs_d, spinEVals_d, b, vec.D_elem()); 
              // PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(vec);
              prop_SL_UP.absorb(vectorAuxF, spin, color);
            }
          }

          for(int spin = 0; spin < N_SPINS; spin++){
            for(int color = 0; color < N_COLS; color++){
              for(int ivec = 0; ivec < nev; ivec++){
                tmp[ivec] = std::conj(spinEVals[ivec*N_SPINS*N_COLS + spin*N_COLS+color])/std::conj(evals[ivec]);
              }
              cudaMemcpy(spinEVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
              checkCudaError();

              cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aM, eigVecs_d, spinEVals_d, b, vec.D_elem()); 
              // PLEGMA_Vector<float> vectorAuxF;
              vectorAuxF.copy(vec);
              prop_SL_DN.absorb(vectorAuxF, spin, color);
            }
          }
            
          if(finalize) {
            prop_SS_UP.rotateToPhysicalBase_device(+1);
            prop_SS_DN.rotateToPhysicalBase_device(-1);
            prop_SS_UP.applyBoundaries_device(source[DIM_T]);
            prop_SS_DN.applyBoundaries_device(source[DIM_T]);
          }
        };

        {
          bool computed_light = false;
          // If twop_filename exists we hold the computation of the light props
          if(access( twop_filename.c_str(), F_OK ) == -1) {
            TIME(computePropagator(propUP, propUP_SL, propDN, propDN_SL, false));
            computed_light = true;
          }
	
          //#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
            cudaMemset(spinVals_d, 0, Eig_NeV*2*sizeof(double));
            cudaMemset(evecs_d, 0, 12*Eig_NeV*V3*2*sizeof(double));

            for(size_t its = 0; its < tSinks.size(); its++){
              int tsinkMtsource = tSinks[its];
              if(tsinkMtsource >= HGC_totalL[3])
                PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
              int signPer = (tsinkMtsource+source[3]) >= HGC_totalL[3] ? -1 : +1;
              int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3]; 

              PLEGMA_Correlator<float> corr(corr_space, source, maxQsq, tsinkMtsource+1);

              WHICHPARTICLE nucleon = get_particle(prOrNt); 
              std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
              for(size_t iproj = 0; iproj < Projs.size(); iproj++){
                auto computeThreep = [&](double run_mu, PLEGMA_Propagator<float>& prop1, PLEGMA_Propagator<float>& prop2, int signProps, PLEGMA_Propagator<float> &propF, PLEGMA_Propagator<float> &propF2, std::string fl) {
                  std::string filename = threep_filename + "_" + Projs[iproj] + "_dt" + std::to_string(tsinkMtsource) + "_" + fl + ".h5";
                  if(access( filename.c_str(), F_OK ) != -1) {
                    PLEGMA_printf("File %s already exists. Skipping...", filename.c_str());
                    return;
                  }
                  if(not computed_light) {
                    TIME(computePropagator(propUP, propUP_SL, propDN, propDN_SL, false));
                    computed_light = true;
                  }
                  cudaMemcpy(eigVecs_d, eigVecs_hL, Eig_NeV*vec_size, cudaMemcpyHostToDevice);
                  checkCudaError();
        
                  {
                    // 3D propagators at t_sink
                    prop13D.absorb(prop1, global_fixSinkTime);
                    prop23D.absorb(prop2, global_fixSinkTime);
                    // PLEGMA_Gauge3D<double> smearedGauge3D_sink;
                    // smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

                    for(int nu = 0 ; nu < 4 ; nu++)
                    for(int c2 = 0 ; c2 < 3 ; c2++){

                      // PLEGMA_Vector<double> vectorInOut;
                      {
                        // PLEGMA_Vector3D<double> vectorAuxD1,vectorAuxD2;
                        if(&prop1 != &prop2)
                          vectorAuxF3D.seqSourceNucleon(prop13D, prop23D, get_projector(Projs[iproj]), nucleon, nu, c2);
                        else
                          vectorAuxF3D.seqSourceNucleon(prop13D, get_projector(Projs[iproj]), nucleon, nu, c2);
                    
                        // put a momentum in the sink later
                        vectorAuxF3D.conjugate();
                        // vectorAuxF3D.apply_gamma(G5); // I think this cancels with the G5 from the left eigenvectors
                        vectorAuxD.copy(vectorAuxF3D);
                        // TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nsmearGauss, alphaGauss));
                        // vectorInOut.absorb(vectorAuxD, global_fixSinkTime);
                      }
                      // if(nu==0 && c2==0){
                      //   vectorAuxD.unload();
                      //   vectorAuxD.writeHDF5("/leonardo_scratch/large/userexternal/cschneid/B64/nucl_defl_3pt/vecAuxD.h5");
                      // }
                      //Invert
                      // Projecting the source
                      int my_it = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
                      bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );
                      
                      TIC();
                      if(not is_myIt) {
                        memset(spinVals, 0, nev*2*sizeof(double));
                      } else {
                        // Copy the non-zero part of the source
                        // double *dst = source_d;
                        // double *src = vectorAuxD.D_elem() + nu*3*V3*2 + c2*V3*2;
                        // cudaMemcpy(dst, src, 2*V3*sizeof(double), cudaMemcpyDeviceToDevice);
                        // checkCudaError();
                          
                        // Copy the needed part of the evecs
                        for(int iv = 0 ; iv < nev ; iv++){
                          for (int spinindex=0; spinindex<4; ++spinindex){
                            for(int c1 = 0 ; c1 < N_COLS ; c1++){
                              double *dst = evecs_d + iv*4*3*V3*2 + spinindex*3*V3*2 + c1*V3*2;
                              double *src = eigVecs_hS + iv*4*3*V4*2 + spinindex*3*V4*2 + c1*V4*2 + my_it*V3*2;
                              cudaMemcpy(dst, src, 2*V3*sizeof(double), cudaMemcpyHostToDevice);
                            }
                          }
                        }  
                        checkCudaError();

                        cuBLAS::gemv(DAGGER, 12*V3, nev, aP, evecs_d, vectorAuxD.D_elem(), b, spinVals_d);
                        checkCudaError();
                        cudaMemcpy(spinVals, spinVals_d, nev*2*sizeof(double), cudaMemcpyDeviceToHost);
                        checkCudaError();
                      }
                      MPI_Allreduce(MPI_IN_PLACE,spinVals,nev*2,MPI_DOUBLE,MPI_SUM,HGC_fullComm);
                      TOC("projecting the source");
                      checkCudaError();

                      // if(nu==0 && c2==0){
                      //   PLEGMA_printf("spinVals:\n");
                      //   for(int i=0; i<2*nev; i++){
                      //     PLEGMA_printf("%e\n", spinVals[i]);
                      //   }
                      // }

                      // Building propagators
                      TIC();
                      for(int i=0; i<nev; i++) {
                        evals[i].imag(run_mu);
                      }
                      vals = (std::complex<double> *) (spinVals);
                      // if(fl=="up") {
                      for(int i=0; i<nev; i++) {
                        vals[i] /= evals[i];
                      }
                      // }
                      // else if(fl=="dn"){
                      //   for(int i=0; i<nev; i++) {
                      //     tmp[i] = vals[i]/std::conj(evals_test[i]);
                      //   }
                      // }
                      // else {
                      //   PLEGMA_error("Flavor %s not recognized",fl.c_str());
                      // }
                      cudaMemcpy(spinVals_d, vals, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
                      checkCudaError();
                      cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aM, eigVecs_d, spinVals_d, b, vec.D_elem());

                      vectorAuxF.copy(vec);
                      // if(nu==0 && c2==0){
                      //   vectorAuxF.unload();
                      //   vectorAuxF.writeHDF5("/leonardo_scratch/large/userexternal/cschneid/B64/nucl_defl_3pt/vecAuxF.h5");
                      // }
                      seqProp.absorb(vectorAuxF, nu, c2);
                      TOC("building propagators");
                    }
                  }
                  seqProp.apply_gamma(G5);
                  seqProp.conjugate();
              
                  // LOCAL contractions
                  TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
                  THREAD(corr.writeFile(filename, corr_file_format));
                      
                  // ONED contractions
                  TIME(corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas));
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
                  THREAD(corr.writeFile( filename, corr_file_format));
                      
                  // noe contractions
                  TIME(corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps));
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
                  THREAD(corr.writeFile( filename, corr_file_format));

                  // LOCAL contractions
                  TIME(corr.contractNucleonThrp_local(seqProp, propF2, signProps, gammas));
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
                  corr.setDatasets((std::vector<std::string>) {"threep_OS"});
                  THREAD(corr.writeFile(filename, corr_file_format));
                      
                  // ONED contractions
                  TIME(corr.contractNucleonThrp_oneD(seqProp, propF2, contractGauge, signProps, gammas));
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
                  corr.setDatasets((std::vector<std::string>) {"threep_OS"});
                  THREAD(corr.writeFile( filename, corr_file_format));
                      
                  // noe contractions
                  TIME(corr.contractNucleonThrp_noe(seqProp, propF2, contractGauge, signProps));
                  if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
                  corr.setDatasets((std::vector<std::string>) {"threep_OS"});
                  THREAD(corr.writeFile( filename, corr_file_format));
                };

                if(nucleon == PROTON) {
                  TIME(computeThreep(-mu_ud, propUP, propDN, +1, propUP_SL, propDN_SL, "up"));
                  TIME(computeThreep( mu_ud, propUP, propUP, -1, propDN_SL, propUP_SL, "dn"));
                } else {
                  TIME(computeThreep( mu_ud, propDN, propUP, -1, propDN_SL, propUP_SL, "dn"));
                  TIME(computeThreep(-mu_ud, propDN, propDN, +1, propUP_SL, propDN_SL, "up"));
                }
              }
            }
          //#endif
        }
        // If twop_filename exists we skip the rest
        if(access( twop_filename.c_str(), F_OK ) != -1) {
          PLEGMA_printf("File %s already exists. Skipping...", twop_filename.c_str());
          continue;
        }
        
        propUP.rotateToPhysicalBase_device(+1);
        propDN.rotateToPhysicalBase_device(-1);
        propUP.applyBoundaries_device(source[3]);
        propDN.applyBoundaries_device(source[3]);
        
        {
          PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
          // TIME(corr.contractMesonsNew(propUP, propDN));
          // char *dset;
          // asprintf(&dset, "twop_mesons_new_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
          // corr.setDatasets((std::vector<std::string>) {dset});
          // free(dset);
          // THREAD(corr.writeFile(twop_filename, corr_file_format));

          // TIME(corr.contractMesonsNew(propUP, propUP));
          // asprintf(&dset, "twop_mesons_new_u[%+1.1e]u[%+1.1e]", mu_ud, mu_ud);
          // corr.setDatasets((std::vector<std::string>) {dset});
          // free(dset);
          // THREAD(corr.writeFile(twop_filename, corr_file_format));

          // TIME(corr.contractMesonsNew(propDN, propDN));
          // asprintf(&dset, "twop_mesons_new_d[%+1.1e]d[%+1.1e]", -mu_ud, -mu_ud);
          // corr.setDatasets((std::vector<std::string>) {dset});
          // free(dset);
          // THREAD(corr.writeFile(twop_filename, corr_file_format));
          
          TIME(corr.contractBaryons(propUP, propDN));
          THREAD(corr.writeFile(twop_filename, corr_file_format));
        }
      }
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();
  return 0;
}

