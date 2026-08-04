#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq",  "corr-file-format",
					     "corr-space","xiMomSm","gammas", "twop-filename"};
  
  initializeOptions(argc, argv, true, listOpt);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  int nStout;
  HGC_options->set("nstout", "n parameter stout smearing", verbosity, nStout);

  size_t WilsDir;
  HGC_options->set("wilson_direction", "Direction of the wilson line", verbosity, WilsDir);
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the WIlson line has to be smaller than 3");

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::string proj;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());

  int tsrc;
  HGC_options->set("t-source", "Time of source vector", verbosity, tsrc);

  int tsnk;
  HGC_options->set("t-sink", "Time of sink vector", verbosity, tsnk);

  std::string fourp_filename;
  HGC_options->set("fourp-filename", "Filename of the four-points correlator", verbosity, fourp_filename);

  std::string qwf_filename;
  HGC_options->set("qwf-filename", "Filename of the quasi-wave function", verbosity, qwf_filename);

  /* std::string ft1_filename;
  std::string ft2_filename;
  HGC_options->set("ft1-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft1_filename);
  HGC_options->set("ft2-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft2_filename); */
  
  initializePLEGMA();

  //set the 2 transverse directions given WilsDir
  size_t Bdir1, Bdir2;
  switch(WilsDir) {
    case 0:
      Bdir1 = 1;
      Bdir2 = 2;
      break;
    case 1:
      Bdir1 = 0;
      Bdir2 = 2;
      break;
    case 2:
      Bdir1 = 0;
      Bdir2 = 1;
      break;
    default:
      PLEGMA_error("The direction of the WIlson line has to be smaller than 3");
  }

  size_t B_max = 8;
  std::vector<int> Lvals = {8};

  std::vector<int> PMom = {0,0,0};
  for(int pi = 0; pi < 3; pi++) {
    PMom[pi] = sinkMom[pi]/2;
  }
  PLEGMA_printf("PMom = %d %d %d\n",PMom[0],PMom[1],PMom[2]);

  std::vector<int> P2Mom = {0,0,0};
  for(int pi = 0; pi < 3; pi++) {
    P2Mom[pi] = 2*sinkMom[pi];
  }
  PLEGMA_printf("P2Mom = %d %d %d\n",P2Mom[0],P2Mom[1],P2Mom[2]);

  std::vector<int> zero_mom = {0,0,0};
  
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.load();
  gauge.calculatePlaq();

  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();

  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);
  gaugeWL.stoutSmearing(gaugeWL,nStout,rhoStout,3);

  if(mu<0)  mu*=-1.;
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;

  PLEGMA_Propagator<float> *propUPp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propUPm = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDNp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDNm = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *prop1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *prop2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;

  site& source =  sourcePositions[0];

  PLEGMA_Correlator<float> corr4p(corr_space, source);
  corr4p.setFixMomVec(P2Mom);

  for(int isource = 0; isource < numSourcePositions; isource++) { //loop over source positions
    int tsrci = tsrc + 2*isource;
    int tsnki = tsnk + 2*isource;

    if(mu < 0) {
      mu *= -1;
      solver->UpdateSolver();
    }

    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsrci,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},+1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propUPp->absorb(vectorAuxF, isc/3, isc%3);
    }

    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsnki,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},-1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propUPm->absorb(vectorAuxF, isc/3, isc%3);
    }
    propUPp->rotateToPhysicalBase_device(+1);
    propUPp->applyBoundaries_device(0);
    propUPm->rotateToPhysicalBase_device(+1);
    propUPm->applyBoundaries_device(0);

    if(mu > 0) {
      mu *= -1;
      solver->UpdateSolver();
    }

    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsrci,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},-1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propDNp->absorb(vectorAuxF, isc/3, isc%3);
    }

    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsnki,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},+1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propDNm->absorb(vectorAuxF, isc/3, isc%3);
    }
    propDNp->rotateToPhysicalBase_device(-1);
    propDNp->applyBoundaries_device(0);
    propDNm->rotateToPhysicalBase_device(-1);
    propDNm->applyBoundaries_device(0);

    //positive b Bdir1
    for(int b = 0; b <= B_max; b++) { //loop over values of B
      
      if(b > 0) {
        propExchange = propIn; propIn = propUPp; propUPp = propExchange;
        propUPp->shift(*propIn, 4 + Bdir1);
        propExchange = propIn; propIn = propUPm; propUPm = propExchange;
        propUPm->shift(*propIn, 4 + Bdir1);
      }

      corr4p.contractMesonsFourp_ultralocal(*propUPm, *propUPp, *propDNp, *propDNm, +b);
      corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);

    } //loop over values of B
    
    //shift back the props
    for(int b = 0; b < B_max; b++) { //loop over values of B

      propExchange = propIn; propIn = propUPp; propUPp = propExchange;
      propUPp->shift(*propIn, Bdir1);
      propExchange = propIn; propIn = propUPm; propUPm = propExchange;
      propUPm->shift(*propIn, Bdir1);

    } //loop over values of B

    //positive b Bdir2
    for(int b = 0; b <= B_max; b++) { //loop over values of B

      if(b > 0) {
        propExchange = propIn; propIn = propUPp; propUPp = propExchange;
        propUPp->shift(*propIn, 4 + Bdir2);
        propExchange = propIn; propIn = propUPm; propUPm = propExchange;
        propUPm->shift(*propIn, 4 + Bdir2);
      }

      corr4p.contractMesonsFourp_ultralocal(*propUPm, *propUPp, *propDNp, *propDNm, +b);
      corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);

    } //loop over values of B
    
    //shift back the props
    for(int b = 0; b < B_max; b++) { //loop over values of B

      propExchange = propIn; propIn = propUPp; propUPp = propExchange;
      propUPp->shift(*propIn, Bdir2);
      propExchange = propIn; propIn = propUPm; propUPm = propExchange;
      propUPm->shift(*propIn, Bdir2);

    } //loop over values of B

    //negative b Bdir1
    for(int b = 0; b <= B_max; b++) { //loop over values of B

      if(b > 0) {
        propExchange = propIn; propIn = propUPp; propUPp = propExchange;
        propUPp->shift(*propIn, Bdir1);
        propExchange = propIn; propIn = propUPm; propUPm = propExchange;
        propUPm->shift(*propIn, Bdir1);

        corr4p.contractMesonsFourp_ultralocal(*propUPm, *propUPp, *propDNp, *propDNm, -b);
        corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
      }

    } //loop over values of B
    
    //shift back the props
    for(int b = 0; b < B_max; b++) { //loop over values of B

      propExchange = propIn; propIn = propUPp; propUPp = propExchange;
      propUPp->shift(*propIn, 4 + Bdir1);
      propExchange = propIn; propIn = propUPm; propUPm = propExchange;
      propUPm->shift(*propIn, 4 + Bdir1);

    } //loop over values of B

    //negative b Bdir2
    for(int b = 0; b <= B_max; b++) { //loop over values of B

      if(b > 0) {
        propExchange = propIn; propIn = propUPp; propUPp = propExchange;
        propUPp->shift(*propIn, Bdir2);
        propExchange = propIn; propIn = propUPm; propUPm = propExchange;
        propUPm->shift(*propIn, Bdir2);

        corr4p.contractMesonsFourp_ultralocal(*propUPm, *propUPp, *propDNp, *propDNm, -b);
        corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
      }

    } //loop over values of B
    
    //shift back the props
    for(int b = 0; b < B_max; b++) { //loop over values of B

      propExchange = propIn; propIn = propUPp; propUPp = propExchange;
      propUPp->shift(*propIn, 4 + Bdir2);
      propExchange = propIn; propIn = propUPm; propUPm = propExchange;
      propUPm->shift(*propIn, 4 + Bdir2);

    } //loop over values of B

  } //loop over source positions

  delete solver;

  finalize();

  return 0;

}