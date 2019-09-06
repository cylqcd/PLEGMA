#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge"};

int main(int argc, char **argv){
  initializeOptions(argc, argv, false, listOpt);
  //==========================//
  //std::string filesPrefix="./";
  //HGC_options->set("output-path", "Path to the directory to dump results", verbosity, filesPrefix);
  double overelaxPar = 0.2;
  HGC_options->set("overelax-param", "The value of the parameter will be used for the overelaxation",verbosity,overelaxPar);
  
  //==========================//
  initializePLEGMA();
  PLEGMA_Gauge<double> *gauge = new PLEGMA_Gauge<double>();
  gauge->readFile(latfile, LIME_FORMAT);
  PLEGMA_printf("Unsmeared Plaquette is: ");
  gauge->calculatePlaq();
  PLEGMA_Gauge<double> gaugeFixed;
  gaugeFixed.gFixingLandau(*gauge,overelaxPar);
  delete gauge;
  finalize();
}
