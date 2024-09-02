#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <cuda_profiler_api.h>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge","src-filename", "twop-filename", "corr-file-format", "corr-space"};


int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  {
    cudaProfilerStart();
    PLEGMA_Gauge<double> gauge(BOTH,THIRD_SIDE);
    // Reading from Lime file and loading to device
    gauge.readFile(latfile, LIME_FORMAT);

    // Computing plaquette on device in three different way for crosschecking
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, false, QUDA_SU3_LINKS);
    plaqQuda();

    //updateOptions(LIGHT);
    //QUDA_solver solver(mu);


    site& source = sourcePositions[0];

    PLEGMA_Propagator<double> prop(BOTH,THIRD_SIDE);

    for (int isc=0; isc<12; isc++){
      PLEGMA_Vector<double> vectorRead(BOTH);
      char buff[100];
      snprintf(buff, sizeof(buff), "%02d", isc);
      std::string buffAsStdStr = buff;
      std::string inputfilename="/eagle/NucleonForm/ckummer/test2d/lime"+buffAsStdStr;
      PLEGMA_printf("Read stochastic source from: %s\n",inputfilename.c_str());
      vectorRead.readFile(inputfilename,LIME_FORMAT);
      vectorRead.load();
      //vectorRead.readFile("srcExp.down.2940.00003.inverted.bin",LIME_FORMAT);

      prop.absorb(vectorRead, isc/3, isc%3);
    }
    //prop.load();
    double norm=prop.norm();
    PLEGMA_printf("Norm2 %e \n",norm);

    //prop.writeFile("prop.lime",LIME_FORMAT); //Used to write a propagator which could then get replaced by Giannis' Propagator
    //prop.readFile("Giannis_prop.lime", LIME_FORMAT); //Read in Giannis' Propagator
    //prop.writeHDF5("/eagle/NucleonForm/ckummer/test2d/Giannis_prop.hdf5"); //Print Propagator for crosscheck

    //Contractions
    std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};

    PLEGMA_Correlator<double> corr(corr_space,source);
    corr.contractNucleonThrp_local(prop,prop,1,gammas,false);
    corr.writeFile(twop_filename,corr_file_format);

    corr.contractNucleonThrp_oneD(prop,prop,gauge,1,gammas,false);
    corr.writeFile(twop_filename,corr_file_format);

    corr.contractNucleonThrp_twoD(prop,prop,gauge,1,gammas,false);
    corr.writeFile(twop_filename,corr_file_format);

    corr.contractNucleonThrp_threeD(prop,prop,gauge,1,gammas,false);
    corr.writeFile(twop_filename,corr_file_format);

    cudaProfilerStop();
  }
  finalize();

  return 0;
}
               
