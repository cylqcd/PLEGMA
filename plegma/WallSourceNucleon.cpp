#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","Projs","xiMomSm","sinkMom","which_particle","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  double mu_ud;

  int tsrc;
  HGC_options->set("t-source", "Time of source vector", verbosity, tsrc);

  std::string ft1_filename;
  std::string ft2_filename;
  std::string ft3_filename;
  std::string ft4_filename;
  HGC_options->set("ft1-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft1_filename);
  HGC_options->set("ft2-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft2_filename);
  HGC_options->set("ft3-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft3_filename);
  HGC_options->set("ft4-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft4_filename);
  
  initializePLEGMA();

  std::vector<int> zero_mom = {0,0,0};
  
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();

  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  
  if(mu<0)  mu*=-1.;
  mu_ud = mu;
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;

  PLEGMA_Propagator<float> *propUPp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDNp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propUPPp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDNPp = new PLEGMA_Propagator<float>(BOTH);

  site& source = sourcePositions[0];
  
  if(mu != mu_ud) {
    mu = mu_ud;
    solver->UpdateSolver();
  }
    
  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsrc,true);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propUPp->absorb(vectorAuxF, isc/3, isc%3);
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsrc,true);
    vectorIn.mulMomentumPhases((std::vector<int>) {sinkMom[0],sinkMom[1],sinkMom[2],0},+1);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propUPPp->absorb(vectorAuxF, isc/3, isc%3);
  }

  propUPPp->rotateToPhysicalBase_device(+1);
  propUPp->rotateToPhysicalBase_device(+1);
  propUPPp->applyBoundaries_device(0);
  propUPp->applyBoundaries_device(0);

  PLEGMA_FT<float> ftUPp(zero_mom,3);
  PLEGMA_FT<float> ftUPPp(sinkMom,3);
  ftUPp.apply(*propUPp);
  ftUPPp.apply(*propUPPp);
  ftUPp.writeFile(ft1_filename, corr_file_format);
  ftUPPp.writeFile(ft2_filename, corr_file_format);

  if(mu != -mu_ud) {
    mu = -mu_ud;
    solver->UpdateSolver();
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsrc,true);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propDNp->absorb(vectorAuxF, isc/3, isc%3);
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsrc,true);
    vectorIn.mulMomentumPhases((std::vector<int>) {sinkMom[0],sinkMom[1],sinkMom[2],0},+1);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propDNPp->absorb(vectorAuxF, isc/3, isc%3);
  }

  propDNp->rotateToPhysicalBase_device(-1);
  propDNp->applyBoundaries_device(source[0]);
  propDNPp->rotateToPhysicalBase_device(-1);
  propDNPp->applyBoundaries_device(source[0]);

  PLEGMA_FT<float> ftDNp(zero_mom,3);
  PLEGMA_FT<float> ftDNPp(sinkMom,3);
  ftDNp.apply(*propDNp);
  ftDNPp.apply(*propDNPp);
  ftDNp.writeFile(ft3_filename, corr_file_format);
  ftDNPp.writeFile(ft4_filename, corr_file_format);

  PLEGMA_Correlator<float> corr(corr_space, source);
  corr.setFixMomVec(sinkMom);
  corr.contractBaryonsWall(*propUPp,*propDNPp,*propDNp,*propUPPp);
  corr.writeFile(twop_filename, corr_file_format);
  
  delete propUPp;
  delete propUPPp;
  delete propDNp;
  delete propDNPp;

  delete solver;
  
  finalize();
  
  return 0;
}
