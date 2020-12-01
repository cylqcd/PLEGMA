#include <PLEGMA.h>
#include <PLEGMA_utils.h>

static std::vector<std::string> listOpt = {"verbosity", "load-gauge-list-filename"};


int main(int argc, char **argv){
  initializeOptions(argc, argv, false, listOpt);
  //==========================//
  std::string overelaxType = "exact";
  HGC_options->set("overelaxType", "Choose between exact overrelaxation and stochastic, options (stoch,exact). Exact is provided from QUDA and stochastic from PLEGMA",verbosity,overelaxType);
  double tolerance = 1e-12;
  HGC_options->set("tolerance", "Tolerance to use for the gauge fixing procedure", verbosity, tolerance);
  double stochoverelaxPar = 0.2;
  HGC_options->set("stoch-overelax-param", "The value of this parameter will be used for the stochastic overelaxation (PLEGMA)",verbosity,stochoverelaxPar);
  double exactoverelaxPar = 1.;
  HGC_options->set("exact-overelax-param", "The value of the parameter that will be used for the exact overelaxation in (QUDA). Set it to one to turn off overelaxation",verbosity,exactoverelaxPar);
  int gaugeFixType = 4;
  HGC_options->set("gaugeFixType", "The type of gauge fixing we want to do. Default is 4. (3-> Coulomb, 4-> Landau)",verbosity,gaugeFixType);
  //==========================//
  initializePLEGMA();
  PLEGMA_Gauge<double> G1,G2;
  if(HGC_verbosity > 1) PLEGMA_printf("Will work on %d confs",listGaugeConfs.size());
  for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
    std::string prefix=splitStrBwd(listGaugeConfs[iconf],'.');
    std::string confStr=splitStrFwd(listGaugeConfs[iconf],'.');
    G1.readFile(listGaugeConfs[iconf], LIME_FORMAT);
    PLEGMA_printf("Plaquette before gauge fixing is: ");
    G1.calculatePlaq();
    if(overelaxType == "exact") gFixingLandauOVR_QUDA(G2,G1,gaugeFixType,
						      exactoverelaxPar,tolerance,20000,1000);
    else if (overelaxType == "stoch") G2.gFixingLandau(G1,stochoverelaxPar,tolerance);
    else PLEGMA_error("Overrelaxation type %s not implemented",overelaxType.c_str());
    G2.writeLIME(prefix+"_lgfix."+confStr);
  }
  finalize();
}
