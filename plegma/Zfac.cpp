#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsrc","src-filename"};
// we use src-filename to provide the momenta we want to do

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::string boundaryCond = "antiperiodic";
  HGC_options->set("boundary-condition", "If we want periodic or antiperiodic in the temporal direction", verbosity, boundaryCond);
  std::string filesPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, filesPrefix);
  //==========================//
  initializePLEGMA();

  // Read the configurations //
  // Warning gauge fixing is needed //
  PLEGMA_Gauge<double> gauge(BOTH,FIRST_VERTEX);
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();
  initGaugeQuda(gauge, boundaryCond == "antiperiodic");
  plaqQuda();
  std::string confStr=splitStrFwd(latfile,'.');
  if(boundaryCond == "antiperiodic") applyBoundaryConditions(gauge,true);
  /////////////////////////////

  if(mu < 0) mu = -mu; // we invert only for up quark with UKQCD conventions
  QUDA_solver solver(mu);
  PLEGMA_Vector<double> vecIn;
  PLEGMA_Vector<double> vecOut;
  PLEGMA_Propagator<double> rprop(BOTH,FIRST_VERTEX);
  PLEGMA_Propagator<double> lprop(BOTH,FIRST_VERTEX);
  for(int i = 0; i < numSourcePositions; i++){ // iterate over the momenta we want to do
    site& sS = sourcePositions[i];
    std::vector<int> mom= {sS[0],sS[1],sS[2],sS[3]};
    std::string pxpypzpt = "px" + std::to_string(sS[0]) + "py" + std::to_string(sS[1]) +
      "pz" + std::to_string(sS[2]) + "pt" + std::to_string(sS[3]);
    //      outName = pathOut + sourceType + "_" + pxpypzpt + "." + confStr + ".data"; 	

    for(int isc =0; isc < 12; isc++){ // compute the prop with mom source
      std::vector<int> scInd = {isc};
      vecIn.setUnit(scInd);

      vecIn.mulMomentumPhases((std::vector<int>) {sS[0],sS[1],sS[2],sS[3]},+1);
      //      vecIn.rotate_uk_ch();
      if(boundaryCond == "antiperiodic") vecIn.mulThetaPhase(1.,true);
      vecIn.apply_gamma(G4,LEFT);
      vecIn.apply_gamma(G5,LEFT);
      solver.solve(vecOut,vecIn);
      vecOut.apply_gamma(G5,LEFT);
      vecOut.apply_gamma(G4,LEFT);
      rprop.absorb(vecOut, isc/3, isc%3); // propagator in g0 convention in UKQCD (implicit flavor flip from g5g4)
    }
    //rprop.rotateToPhysicalBase_device(-1); // propagator in g0 convention in chiral (flipped flavor sign)
    // compute the prop in momentum space
    PLEGMA_FT<double> ft((std::vector<double>) {(double)mom[0], (double)mom[1], (double)mom[2], mom[3]+0.5},4,false); // half twist in temporal direction
    //    rprop.scale(1./HGC_totalVolume);
    ft.apply(rprop, FT_GEMV,-1);
    //    rprop.scale(HGC_totalVolume);
    ft.writeASCII(filesPrefix+"zfac_GpropMom_"+pxpypzpt+"_conf_"+confStr+".dat",0);
    ///////////
    lprop.copy(rprop);

    // lprop.rotateToPhysicalBase_device(+1); // this needs to be in (1+ig5)

    // lprop.rotateToPhysicalBase_device(+1);
    lprop.apply_gamma(G5,LEFT);
    lprop.apply_gamma(G5,RIGHT);
    // lprop.rotateToPhysicalBase_device(-1);

    //lprop.rotateToPhysicalBase_device(+1);
    //lprop.rotateToPhysicalBase_device(+1);
    lprop.conjugate(); // this is to make the dagger, trans is in the code

    site source({0,0,0,0});
    std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};     

    PLEGMA_Correlator<double> corr(MOMENTUM_SPACE, source, 0, HGC_totalL[3]);
    corr.contractNucleonThrp_local(lprop,rprop,0,gammas,true);
    corr.writeFile(filesPrefix+"zfac_Vloc_"+pxpypzpt+"_conf_"+confStr,HDF5_FORMAT);

    /***corr.contractNucleonThrp_oneD(lprop,rprop,gauge,0,gammas,true);
    corr.writeFile(filesPrefix+"zfac_VoneD_"+pxpypzpt+"_conf_"+confStr,HDF5_FORMAT);

    corr.contractNucleonThrp_twoD(lprop,rprop,gauge,0,gammas,true);
    corr.writeFile(filesPrefix+"zfac_VtwoD_"+pxpypzpt+"_conf_"+confStr,HDF5_FORMAT);
    
    corr.contractNucleonThrp_threeD(lprop,rprop,gauge,0,gammas,true);
    corr.writeFile(filesPrefix+"zfac_VthreeD_"+pxpypzpt+"_conf_"+confStr,HDF5_FORMAT);***/
  }

    
  finalize();
  return 0;
}
