#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv, true);

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();

  // Smearing
  PLEGMA_Gauge<double> smearedGauge(BOTH);
  smearedGauge.APEsmearing(gauge, 0, 0.1, 3); // TODO: here should go the smearing params
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  
  PLEGMA_Gauge<float> contractGauge;
  contractGauge.copy(gauge);

  // apply boundary conditions since is needed for the covariant derivative
  // this needs to be done after initGaugeQuda otherwise causes troubles
  applyBoundaryConditions(contractGauge,true);

  
  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver *solverUP = new QUDA_solver(mu);
  mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> propUP;
  PLEGMA_Propagator<float> propDN;
  PLEGMA_Propagator<float> seqProp;
  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  PLEGMA_Correlator<float> corr(MOMENTUM_SPACE,1);

  // for the test use sinkSourceSep = 10;
  int  isource=0;

  int tsinkMtsource = 10;//test with tsink 10
  int signPer = (tsinkMtsource+sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
  int global_fixSinkTime = (tsinkMtsource + sourcePositions[isource][3])%HGC_totalL[3]; 

  for(int isc = 0 ; isc < 12 ; isc++){
    vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
    vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
    
    PLEGMA_printf("Going to invert UP for component %d\n", isc);
    solverUP->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propUP.absorb(vectorAuxF, isc/3, isc%3);
    vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
    vectorAuxF.copy(vectorAuxD);
    propUP3D.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
    
    PLEGMA_printf("Going to invert DN for component %d\n", isc);
    solverDN->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propDN.absorb(vectorAuxF, isc/3, isc%3);
    vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
    vectorAuxF.copy(vectorAuxD);
    propDN3D.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
  }

  WHICHPARTICLE nucleon = NEUTRON; // for the test is NEUTRON, later we can provide an option
  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
  //seq source part 2Props and contraction block
  {
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	if(nucleon == PROTON)
	  vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, P4_P, nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
	else
	  vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, P4_P, nucleon, global_fixSinkTime, nu, c2);
	// put a momentum in the sink later
	vectorAuxF.conjugate();
	vectorAuxF.apply_gamma(G5);
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverDN->solve(vectorOut, vectorIn); else solverUP->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqProp.absorb(vectorAuxF, nu, c2);
      }
    seqProp.apply_gamma(G5);
    seqProp.conjugate();
    int signProps = (nucleon == PROTON) ? +1: -1;

    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propUP : propDN;
    // LOCAL contractions
    corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas, sourcePositions[isource]); // 0 is isource change later 
    if(signPer < 0) for(int iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;      
    corr.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_local_CP2_new.dat");

    // ONED contractions
    corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas, sourcePositions[isource]); // 0 is isource change lat
    if(signPer < 0) for(int iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;      
    corr.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_oneD_CP2_new.dat");

    // noe contractions
    corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps, sourcePositions[isource]); // 0 is isource change lat
    if(signPer < 0) for(int iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;      
    corr.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_noe_CP2_new.dat");

    // do the contractions also for the conserved
  }
  
  //seq source part 1Prop contraction
  {
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	if(nucleon == PROTON)
	  vectorAuxF.seqSourceNucleon(propUP3D, P4_P, nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
	else
	  vectorAuxF.seqSourceNucleon(propDN3D, P4_P, nucleon, global_fixSinkTime, nu, c2);
	// put a momentum in the sink later
	vectorAuxF.conjugate();
	vectorAuxF.apply_gamma(G5);
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverUP->solve(vectorOut, vectorIn); else solverDN->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqProp.absorb(vectorAuxF, nu, c2);
      }
    seqProp.apply_gamma(G5);
    seqProp.conjugate();
    int signProps = (nucleon == PROTON) ? -1: +1;
    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propDN : propUP;
    
    //LOCAL
    corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas, sourcePositions[isource]);
    if(signPer < 0) for(int iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
    corr.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_local_CP1_new.dat");

    //ONED
    corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas, sourcePositions[isource]);
    if(signPer < 0) for(int iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
    corr.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_oneD_CP1_new.dat");

    //ONED
    corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps, sourcePositions[isource]);
    if(signPer < 0) for(int iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
    corr.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_noe_CP1_new.dat");

    // do the contractions also for the conserved
  }

  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c2 = 0 ; c2 < 3 ; c2++){
      vectorAuxF.absorb(propUP, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propUP.absorb(vectorAuxF, nu, c2);

      vectorAuxF.absorb(propDN, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propDN.absorb(vectorAuxF, nu, c2);
    }
  
  propUP.rotateToPhysicalBase_device(+1);
  propDN.rotateToPhysicalBase_device(-1);
  propUP.applyBoundaries_device(sourcePositions[isource][3]);
  propDN.applyBoundaries_device(sourcePositions[isource][3]);
  
  corr.contractMesons(propUP, propDN, sourcePositions[isource]);
  corr.writeFile(twop_filename.c_str(), corr_file_format);

  corr.contractBaryons(propUP, propDN, sourcePositions[isource]);
  corr.writeFile(twop_filename.c_str(), corr_file_format);
  
  delete solverUP;
  delete solverDN;
  finalize();

  return 0;
}

