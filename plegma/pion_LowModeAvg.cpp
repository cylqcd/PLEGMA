#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

using namespace plegma;
using namespace quda;

//For calculating runtime.
std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back() 

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsrc", "src-filename", "tSinks", "twop-filename", "maxQsq", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
					   "Eig-NkV", "Eig-logFile",
#elif defined(HAVE_PRIMME)
					   "Eig-printLevel", "Eig-method-PRIMME",
#endif
					   "rng-seed", "corr-file-format"
};


int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int n_stochastic_samples;
  int nroots=2; //
  int rand_seed1=1234; //random seed for initialization of stochastic sources.
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  std::vector<double> mus; //Twisted mass mus
  HGC_options->set("extra-mu", "List of additional mu to run", verbosity, mus);
  //std::string Eig_outputFile = "./eigsVdagG5V.dat";
  //HGC_options->set("Eig-outputFile", "Path to dump the eigenvalues and vdag g5 v if low-modes-recon is enabled",verbosity, Eig_outputFile);
  bool isReadEigenVecs = false;
  bool isWriteEigenVecs = false;
  std::string fnameEigenVecsPrefix="";
  HGC_options->set("readEigenVectors", "Where we want to read EigenVectors from file", verbosity, isReadEigenVecs);
  HGC_options->set("writeEigenVectors", "Where we want to read EigenVectors from file", verbosity, isWriteEigenVecs);
  HGC_options->set("prefixEigenVecsFile", "Path with prefix for the filenames of the eigenvectors", verbosity, fnameEigenVecsPrefix);

  //=========================================================================================================//

  initializePLEGMA();

  mus.insert(mus.begin(), mu); //Prepend mu to vector mus.
  int nmus = mus.size();

  //Storing only the smeared gauge
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  //Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true);
  plaqQuda();

  //Initializing solver.
  updateOptions(LIGHT);
  TIME(QUDA_solver solver(mu));

  //Initializing stochastic vector using given random seed.
  PLEGMA_Vector<double> vector_stoc;
  vector_stoc.randInit(rand_seed1);

  //Initialize gamma list.
  std::vector<GAMMAS_SCATT> glist_src={ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};
  std::vector<GAMMAS_SCATT> glist_sink={ID};

  //Eigensolver to get eigenvalues
  #if defined(HAVE_EIGENSOLVER)
    EigSolver *eigSol = nullptr;  

    EigSolverParams eigParam;
    eigParam.NeV = Eig_NeV;
    eigParam.isACC = Eig_isACC;
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
    #elif defined(HAVE_PRIMME)
        eigParam.printLevel = Eig_printLevel;
        eigParam.primme_method=getMethod(Eig_method);
    #else
        PLEGMA_error("No arpack or primme is compiled");
    #endif
        TIME(eigSol = new EigSolver(eigParam, dslash_type, isReadEigenVecs, isWriteEigenVecs, fnameEigenVecsPrefix, true));
        //eigSol->dumpEvalsVdagG5V(Eig_outputFile);
    #else
        PLEGMA_error("No eigenSolver is compiled");
  #endif

  //QUDA_TWISTED_CLOVER_DSLASH is used, so we determine eigenvalues of D(\mu)=V(\Lambda+i\mu)U^\dagger
  //V=\Gamma_5 U*
  //Need to do this also for -i\mu
  //list of lambda, v gamma u', v gamma phi, v gamma eta
  #if defined(HAVE_EIGENSOLVER)
  //exact-exact part
  //=========================================================================================================//
  //for(size_t its = 0; its < tSinks.size(); its++){
    //int tsink = tSinks[its];
      int its = 0;
      int tsink = 0;

      PLEGMA_Vector<double> eigVec;
      PLEGMA_Vector<double> eigVecP;

      for(int i=0; i < 1; i++){ //eigSol->getEigVals().size(); i++){
        //double eigVal = std::get<0>(eigSol->getEigVals()[i]); //Doesn't this only get the real part of the eigenvector? Only real eigenvalues if Dirac operator is hermitian. However, here we have to take the twisted mass parameter into account!
        long int iorder = std::get<3>(eigSol->getEigVals()[i]);
        double *eigVec_tmp = eigSol->getEigVecs() + iorder*eigSol->getSize_per_Vec()*2;
        cudaMemcpy(eigVec.D_elem(), eigVec_tmp, eigSol->getBytes_per_Vec(), cudaMemcpyHostToDevice); //Copy eigVec_tmp (host) to eigVec (device).
        checkCudaError();
        
        for(int j=0; j < 1; j++){//eigSol->getEigVals().size(); j++){
          PLEGMA_ScattCorrelator<double> corr(site({0,0,0,tsink}), maxQsq);
          //std::vector<GAMMAS_SCATT> glist_src={ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};
          //std::vector<GAMMAS_SCATT> glist_sink={ID};
          std::string dataset_name = std::to_string(i) + std::to_string(j);
          TIME(corr.initialize_diagram(glist_src, glist_sink, "P"));

          char * src_string;
          asprintf(&src_string, "exact-exact");
          std::string outfilename = twop_filename + src_string + ".h5";
          free(src_string);

          //double eigValP = std::get<0>(eigSol->getEigVals()[j]); 
          iorder = std::get<3>(eigSol->getEigVals()[j]);
          double *eigVecP_tmp = eigSol->getEigVecs() + iorder*eigSol->getSize_per_Vec()*2;
          cudaMemcpy(eigVecP.D_elem(), eigVecP_tmp, eigSol->getBytes_per_Vec(), cudaMemcpyHostToDevice);
          checkCudaError();
          eigVecP.apply_gamma5(); //Make eigVecP a left eigenvector by applying gamma5.

          PLEGMA_printf("eigVecNorm: %f", eigVec.norm());
          PLEGMA_printf("eigVecPNorm: %f", eigVecP.norm());
	  eigVecP.unload();
	  eigVecP.writeHDF5("eigVecTest");

          TIME(corr.PhiPhi(eigVec, glist_src, eigVecP));
          corr.setDatasets((std::vector<std::string>) {dataset_name});
          TIME(corr.writeHDF5( outfilename )); 
        }
      }
    //=========================================================================================================//

    //exact-stochastic part and stochastic-stochastic part
    //=========================================================================================================//  
    //{
      PLEGMA_printf("\n ### Calculations for stochastic source %d - %02d begin now ###\n\n",
                    its, tsink);

      //We draw a different random vector for every source position
      vector_stoc.stochastic_Z(nroots);
          
      //Take current time slice given by tsink and dilute spins.
      //Dilution
      PLEGMA_Vector<double> vectortmp1;
      PLEGMA_Vector<double> vectortmp2;
      vectortmp1.absorbTimeslice(vector_stoc, tsink);
      vector_stoc.dilutespin(vectortmp1,0); //put 0 everywhere in the stochastic vector except for the position corresponding to one specific timeslice and mu.

      PLEGMA_Propagator<double> propUP; //Declaration of the two propagators. UP = +mu and DN = -mu
      PLEGMA_Propagator<double> propDN;
      PLEGMA_Propagator<double> propUP_phi; //The propagators containing phi.
      PLEGMA_Propagator<double> propDN_phi;
      //PLEGMA_Vector<double> eigVec; //Declaration of eigenvector

      for(int imu=0; imu<nmus; imu++){ //loop over all mus.
        //Prepare ouput file.
        mu = mus[imu];

        char * src_string;
        asprintf(&src_string, "_%.8f_id%02d_st%03d_exactstoc", mu, its, tsink);
        std::string outfilename_exactstoc = twop_filename + src_string + ".h5";

        //Skip this "imu" if output file already exists.
        if(access( outfilename_exactstoc.c_str(), F_OK ) != -1) {
          PLEGMA_printf("File %s already exists. Skipping...", outfilename_exactstoc.c_str());
          continue;
        }

        asprintf(&src_string, "_%.8f_id%02d_st%03d_stocstoc", mu, its, tsink);
        std::string outfilename_stocstoc = twop_filename + src_string + ".h5";
        free(src_string);

        //Skip this "imu" if output file already exists.
        if(access( outfilename_stocstoc.c_str(), F_OK ) != -1) {
          PLEGMA_printf("File %s already exists. Skipping...", outfilename_stocstoc.c_str());
          continue;
        }

        for(int fl=0; fl<2; fl++){
          char datasetPrefix = '+';
          if(fl==1) {
            mu *= -1; //For index fl==1 flip sign of mu.
            datasetPrefix = '-'; //Use prefix "-" for -mu data sets.
          }
          solver.UpdateSolver();

          for (int spinindex=0; spinindex<4; ++spinindex){ //Push stochastic source one forward.
            if (spinindex==0)
              vectortmp1.copy(vector_stoc);
            else
              vectortmp1.diluteSpinDisplace(vector_stoc,spinindex,0);
              TIME(solver.solve(vectortmp1, vectortmp1));

            if(fl==0)
              propUP.absorb(vectortmp1, spinindex, 0); //Last argument is the color index, put to 0 here.
            else
              propDN.absorb(vectortmp1, spinindex, 0);

            eigSol->projectVector(vectortmp1); //Project vector to obtain "phi".

            if(fl==0)
              propUP_phi.absorb(vectortmp1, spinindex, 0); //Last argument is the color index, put to 0 here.
            else
              propDN_phi.absorb(vectortmp1, spinindex, 0);
          }
            
          propUP.rotateToPhysicalBase_device(+1);
          propUP.applyBoundaries_device(tsink); //take into account (antiperiodic) boundary conditions.
          propDN.rotateToPhysicalBase_device(-1);
          propDN.applyBoundaries_device(tsink);
          propUP_phi.rotateToPhysicalBase_device(+1);
          propUP_phi.applyBoundaries_device(tsink);
          propDN_phi.rotateToPhysicalBase_device(-1);
          propDN_phi.applyBoundaries_device(tsink);


          //exact-stochastic contractions for v gamma eta and v gamma phi terms
          //=========================================================================================================//
          for(int i=0; i < eigSol->getEigVals().size(); i++){
            //double eigVal = std::get<0>(eigSol->getEigVals()[i]); //Doesn't this only get the real part of the eigenvector? Only real eigenvalues if Dirac operator is hermitian. However, here we have to take the twisted mass parameter into account!
            long int iorder = std::get<3>(eigSol->getEigVals()[i]);
            double *eigVec_tmp = eigSol->getEigVecs() + iorder*eigSol->getSize_per_Vec()*2;
            cudaMemcpy(eigVec.D_elem(), eigVec_tmp, eigSol->getBytes_per_Vec(), cudaMemcpyHostToDevice);
            checkCudaError();

            //Initialize new Scatt_Correlator.
            PLEGMA_ScattCorrelator<double> mix_corr(site({0,0,0,tsink}), maxQsq);
            //std::vector<GAMMAS_SCATT> glist_src={ID,G_1,G_2,G_3,G_4,G_5,G_5_G_1,G_5_G_2,G_5_G_3,G_5_G_4};
            //std::vector<GAMMAS_SCATT> glist_sink={ID};
            TIME(mix_corr.initialize_diagram(glist_src, glist_sink, "P"));

            TIME(mix_corr.V3(eigVec, glist_src, propUP));
            mix_corr.setDatasets((std::vector<std::string>) {datasetPrefix + std::to_string(i) + "u"});
            TIME(mix_corr.writeHDF5( outfilename_exactstoc ));
            TIME(mix_corr.V3(eigVec, glist_src, propDN));
            mix_corr.setDatasets((std::vector<std::string>) {datasetPrefix + std::to_string(i) + "d"});
            TIME(mix_corr.writeHDF5( outfilename_exactstoc ));
            TIME(mix_corr.V3(eigVec, glist_src, propUP_phi));
            mix_corr.setDatasets((std::vector<std::string>) {datasetPrefix + std::to_string(i) + "uphi"});
            TIME(mix_corr.writeHDF5( outfilename_exactstoc ));
            TIME(mix_corr.V3(eigVec, glist_src, propDN_phi));
            mix_corr.setDatasets((std::vector<std::string>) {datasetPrefix + std::to_string(i) + "dphi"});
            TIME(mix_corr.writeHDF5( outfilename_exactstoc ));
          }

          PLEGMA_printf("Calculating stoc-stoc contributions... \n\n");
          //=========================================================================================================//

          //stochastic-stochastic contractions
          //=========================================================================================================//
          PLEGMA_Correlator<double> stoc_corr(corr_space, site({0,0,0,tsink}), maxQsq);
          TIME(stoc_corr.contractMesonsNew(propUP, propUP, false));
          stoc_corr.setDatasets((std::vector<std::string>) {datasetPrefix + "uu"});
          TIME(stoc_corr.writeHDF5( outfilename_stocstoc ));
          TIME(stoc_corr.contractMesonsOpen(propUP, propUP, false));
          stoc_corr.setDatasets((std::vector<std::string>) {datasetPrefix + "uu_open"}); //Leave spin indices open (16x16 indices)
          TIME(stoc_corr.writeHDF5( outfilename_stocstoc ));
          TIME(stoc_corr.contractMesonsNew(propDN, propDN, false));
          stoc_corr.setDatasets((std::vector<std::string>) {datasetPrefix + "dd"});
          TIME(stoc_corr.writeHDF5( outfilename_stocstoc ));
          TIME(stoc_corr.contractMesonsOpen(propDN, propDN, false));
          stoc_corr.setDatasets((std::vector<std::string>) {datasetPrefix + "dd_open"});
          TIME(stoc_corr.writeHDF5( outfilename_stocstoc ));
          TIME(stoc_corr.contractMesonsNew(propUP, propDN, false));
          stoc_corr.setDatasets((std::vector<std::string>) {datasetPrefix + "ud"}); //"du" is just the conjugated one, so no need to compute it separately.
          TIME(stoc_corr.writeHDF5( outfilename_stocstoc ));
          TIME(stoc_corr.contractMesonsOpen(propUP, propDN, false));
          stoc_corr.setDatasets((std::vector<std::string>) {datasetPrefix + "ud_open"});
          TIME(stoc_corr.writeHDF5( outfilename_stocstoc ));
          //=========================================================================================================//
        }
      }
    //}
  //}
  #endif

  //finalize();

  return 0;
}

  /*Todo: 1) Find smallest eigenvalue of Dirac operator. 
          2) Deflate Dirac operator.
          3) Repeat until sufficient number of eigenvalues has been found. 
            -> Eigensolver does all this (step 1)-3)). Accessible via eigSol->getEigVals().
              Have only right eigenvectors. -> To get left one transpose and apply gamma5 from the left.
              But which eigenvalues are those? Are these for both propagators, or only one of them? Eigenvalues and -vectors are actually the same. Have to perform double sum over them.
              Are the vectors ordered in the same way as the eigenvalues? Line 96 gets the index of the eigenvector corresponding to a given eigenvalue.
          4) Use eigenvalues and corresponding eigenvectors to construct terms (a,b)_t
            -> Need to find a way to make the contractions. Specifically need to know how to apply gamma_5. Doable by copying the eigenvector (double) to a vector of type PLEGMA_vector.
              How do we do this for each lattice site?
              What is Gamma in our case? For pions its just gamma_5.
          5) Contract propagators to find the exact-exact part of the correlation function.
  */
