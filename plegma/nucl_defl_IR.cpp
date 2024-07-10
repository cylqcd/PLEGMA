#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_BLAS.h>
#include <stdio.h>
using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIC()  runtime.push_back(MPI_Wtime())
#define TOC(str)  PLEGMA_printf("TIME for %s %f sec\n", str, MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

#define TIME(fnc)  TIC(); fnc;	TOC(#fnc)

#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
					   "Eig-NkV", "Eig-logFile",
#endif
					  "maxQsq", "rng-seed", "corr-file-format", "twop-filename",
            "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					  "nsrc", "src-filename"
};


int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::string loopsPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
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
  int nroots=2;
  int rand_seed1=1234;
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  std::vector<double> mus;
  HGC_options->set("extra-mu", "List of additional mu to run", verbosity, mus);
  std::vector<double> nevs;
  HGC_options->set("extra-nev", "List of additional nev to run", verbosity, nevs);
  bool fastio = false;
  HGC_options->set("fastio", "Whether eigenvectors are read using fastio", verbosity, fastio);
  int startSource = 0;
  HGC_options->set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
  bool exact = false;
  HGC_options->set("exact", "Whether the exact part should be computed", verbosity, exact);
  bool pointToAll = false;
  HGC_options->set("point-to-all", "Whether the point-to-all sources part should be computed", verbosity, pointToAll);
  //=========================================================================================================//
  TIME(initializePLEGMA());

  mus.insert(mus.begin(), mu);
  int nmus = mus.size();
  nevs.insert(nevs.begin(), Eig_NeV);
  for(int inev=1; inev < nevs.size(); inev++){ // Loop over NeV
    if(nevs[inev]>nevs[0]) PLEGMA_error("extra-nev can only be smaller");
  }

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  TIME(gauge.readFile(latfile, LIME_FORMAT));
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  TIME(initGaugeQuda(gauge, true));
  plaqQuda();


  // apply boundary conditions since is needed for the covariant derivative
  // this needs to be done after initGaugeQuda otherwise causes troubles
  // applyBoundaryConditions(gauge,true);

  // In case we need LMR we need to compute the eigenvectors
  EigSolver *eigSol = nullptr;
  if(Eig_NeV>0) {
#if defined(HAVE_EIGENSOLVER)
    EigSolverParams eigParam;
    eigParam.NeV = Eig_NeV;
    eigParam.isACC = Eig_isACC;
    eigParam.littleD = true;
    eigParam.fastio = fastio;
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

    if(nsmearGauss>0){
      PLEGMA_Gauge<double> smearedGauge(BOTH);
      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
      
      PLEGMA_Vector<double> tmp1;
      PLEGMA_Vector<double> tmp2;
      PLEGMA_Vector<double> tmp3(NONE);
      size_t vec_size = eigSol->getSize_per_Vec()*2;
      for(int i=0; i<Eig_NeV; i++){
        tmp3.D_elem(eigSol->getEigVecs()+i*vec_size);
        tmp1.copy(tmp3);
        TIME(tmp2.gaussianSmearing(tmp1, smearedGauge, nsmearGauss, alphaGauss));
        tmp3.copy(tmp2);
      }
    }


    if(exact){
      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,0}), 0); //maxQsq set to 0 here!

      for(int inev=0; inev < nevs.size(); inev++){ // Loop over NeV //inev set to one because we have to read 400evs with fastio but actuall want to use less!
        TIC();
        int nev = nevs[inev];
      
        char * src_string;
        asprintf(&src_string, "_nucl_exact_nev%03d", nev);
        std::string outfilename = twop_filename + src_string + ".h5";
        free(src_string);

        if(access( outfilename.c_str(), F_OK ) == -1) {
          std::complex<double> evals[nev];
          for(int i=0; i<nev; i++) {
            evals[i] = eigSol->getLittleD()[i*(Eig_NeV+1)];
        }

          TIME(corr.contractBaryonsEEE(eigSol->getEigVecs(), (double*) evals, nev, eigSol->getSize_per_Vec()*2, isDeviceEigenVecs));
          TIME(corr.writeHDF5( outfilename ));
        }
        asprintf(&src_string, "contracting %d EVs", nev);
        TOC(src_string);
        free(src_string);
      }
    }
#else
    PLEGMA_error("No eigenSolver is compiled");
#endif
  }
  
  if(pointToAll){
    std::string given_twop_filename = twop_filename;
    std::string given_threep_filename = threep_filename;

    double aP[2]={1.,0.}, b[2]={0.,0.};
	  size_t size_per_Vec = eigSol->getSize_per_Vec();
    PLEGMA_Vector<double> vec;
    PLEGMA_Propagator<double> propUP;
    PLEGMA_Propagator<double> propDN;

    double *eigVecs = eigSol->getEigVecs();
    std::complex<double> spinEVals[12*Eig_NeV];
    double *spinEVals_d;
    size_t V4 = HGC_localVolume;
    cudaMalloc((void**)&spinEVals_d, 12*2*Eig_NeV*sizeof(double));
    std::complex<double> evals[Eig_NeV], tmp[Eig_NeV];
    for(int i=0; i<Eig_NeV; i++) {
      evals[i] = eigSol->getLittleD()[i*(Eig_NeV+1)];
    }

    for(int isource = startSource; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);


      // create prop up and dn
      memset(spinEVals, 0, 12*Eig_NeV*2*sizeof(double));
      cudaMemset(spinEVals_d, 0, 12*Eig_NeV*2*sizeof(double));
      int my_src[N_DIMS];
      bool mine = true;
      size_t id=0;
      for(int i = N_DIMS-1; i >= 0; i--) {
        my_src[i] = (source[i] - HGC_procPosition[i] * HGC_localL[i]);

        // if out of the local lattice we break
        if((my_src[i]<0) || (my_src[i]>=HGC_localL[i])) mine=false;

        id = id * HGC_localL[i] + my_src[i];
      }
      // This make it work also for vector3D
      //id = id % this->Total_length();

      if(mine) {
        for(int ivec = 0; ivec < Eig_NeV; ivec++){
          for(int spin = 0; spin < N_SPINS; spin++){
            for(int color = 0; color < N_COLS; color++){
              cudaMemcpy(spinEVals_d + (ivec*N_SPINS*N_COLS + spin*N_COLS+color)*2, eigVecs + (ivec*N_SPINS*N_COLS*V4 + (((spin+2)%N_SPINS)*N_COLS+color)*V4 + id)*2, 2*sizeof(double), cudaMemcpyDeviceToDevice); //((spin+2)%N_SPINS) because of gamma_5 
            }
          }
        }
        cudaMemcpy(spinEVals, spinEVals_d, 12*Eig_NeV*2*sizeof(double), cudaMemcpyDeviceToHost);
	      checkCudaError();
      }
      MPI_Allreduce(MPI_IN_PLACE,spinEVals,12*Eig_NeV*2,MPI_DOUBLE,MPI_SUM,HGC_fullComm);
      // PLEGMA_printf("spinEVals:\n");
      // for(int i=0; i<12*Eig_NeV; i++){
      //   PLEGMA_printf("%f+%f\n", spinEVals[i].real(), spinEVals[i].imag());
      // }
      
      for(int inev=0; inev < nevs.size(); inev++){ // Loop over NeV
        TIC();
        int nev = nevs[inev];

        char * src_string;
        asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d_nev%03d", source[0], source[1], source[2], source[3], nev);
        twop_filename = given_twop_filename + src_string + ".h5";
        threep_filename = given_threep_filename + src_string;
        free(src_string);
        
        TIC();
        // PLEGMA_printf("tmp up:\n");
        for(int spin = 0; spin < N_SPINS; spin++){
          for(int color = 0; color < N_COLS; color++){
            for(int ivec = 0; ivec < nev; ivec++){
              tmp[ivec] = std::conj(spinEVals[ivec*N_SPINS*N_COLS + spin*N_COLS+color])/evals[ivec];
              // PLEGMA_printf("%f\n", tmp[ivec]);
            }
            cudaMemcpy(spinEVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
            cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aP, eigVecs, spinEVals_d, b, vec.D_elem());
            propUP.absorb(vec, spin, color);
            // if(spin==0 && color==0){
            //   vec.unload();
            //   vec.writeHDF5("/leonardo_scratch/large/userexternal/cschneid/B64/nucl_defl/vec.h5");
            // }

          }
        }

        // PLEGMA_printf("tmp down:\n");
        for(int spin = 0; spin < N_SPINS; spin++){
          for(int color = 0; color < N_COLS; color++){
            for(int ivec = 0; ivec < nev; ivec++){
              tmp[ivec] = std::conj(spinEVals[ivec*N_SPINS*N_COLS + spin*N_COLS+color])/std::conj(evals[ivec]);
              // PLEGMA_printf("%f\n", tmp[ivec]);
            }
            cudaMemcpy(spinEVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
            cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aP, eigVecs, spinEVals_d, b, vec.D_elem());
            propDN.absorb(vec, spin, color);
          }
        }

        propUP.rotateToPhysicalBase_device(+1);
        propDN.rotateToPhysicalBase_device(-1);
        propUP.applyBoundaries_device(source[3]);
        propDN.applyBoundaries_device(source[3]);
        TOC("Constructing propagators");
        
        {
          PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
          // TIME(corr.contractMesonsNew(propUP, propDN));
          // char *dset;
          // asprintf(&dset, "twop_mesons_new_u[%+1.1e]d[%+1.1e]", mu, -1*mu);
          // corr.setDatasets((std::vector<std::string>) {dset});
          // free(dset);
          // THREAD(corr.writeFile(twop_filename, corr_file_format));

          // TIME(corr.contractMesonsNew(propUP, propUP));
          // asprintf(&dset, "twop_mesons_new_u[%+1.1e]u[%+1.1e]", mu, mu);
          // corr.setDatasets((std::vector<std::string>) {dset});
          // free(dset);
          // THREAD(corr.writeFile(twop_filename, corr_file_format));

          // TIME(corr.contractMesonsNew(propDN, propDN));
          // asprintf(&dset, "twop_mesons_new_d[%+1.1e]d[%+1.1e]", -mu, -mu);
          // corr.setDatasets((std::vector<std::string>) {dset});
          // free(dset);
          // THREAD(corr.writeFile(twop_filename, corr_file_format));

          TIME(corr.contractBaryons(propUP, propDN));
          THREAD(corr.writeFile(twop_filename, corr_file_format));
        }
        asprintf(&src_string, "contracting %d EVs", nev);
        TOC(src_string);
        free(src_string);
      }
    }
  }
  finalize();
  return 0;
}
