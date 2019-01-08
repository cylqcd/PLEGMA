#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];
extern int nsmearAPE;
extern double alphaAPE;

int main(int argc, char **argv)
{
  PLEGMA_params params;
  initialize(argc, argv, &params);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  //-Read the gauge field in lime format
  GaugeBuffer<double> gauge(params);
  readLimeGauge(gauge.get_ptr(), latfile, &gauge_param, params.procs);

  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  initGaugeQuda((void*)gauge.get_ptr(), gauge_param);
  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  // The gauge is loaded in a format suitable for QUDA. We need to re-map it
  mapEvenOddToNormalGauge(gauge.get_ptr(),gauge_param,params.lL);

  PLEGMA_Gauge<double> pGauge;
  pGauge.pack(gauge.get_ptr());
  pGauge.load();
  pGauge.calculatePlaq();
  
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(pGauge, nsmearAPE, alphaAPE, 3);
  smearedGauge.calculatePlaq();

  // pGauge will be used later for the derivatives also in the temporal directions so should have the sign
  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  pGauge.pack(gauge.get_ptr());
  pGauge.load();
  
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
  PLEGMA_Correlator<float> nucleonThrpLocal_CP1;
  PLEGMA_Correlator<float> nucleonThrpLocal_CP2;
  // for the test use sinkSourceSep = 10;
  int  isource=0;

  int tsinkMtsource = 10;//test with tsink 10
  int signPer = (tsinkMtsource+GK_sourcePosition[isource][3]) >= GK_totalL[3] ? -1 : +1;
  int global_fixSinkTime = (tsinkMtsource + params.sourcePosition[isource][3])%GK_totalL[3]; 
  int my_fixSinkTime = global_fixSinkTime - comm_coords(default_topo)[3] * GK_localL[3];
  bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < GK_localL[3] );

  for(int isc = 0 ; isc < 12 ; isc++){
    vectorAuxD.pointSource(params.sourcePosition[isource], isc/3, isc%3, DEVICE);
    vectorIn.gaussianSmearing(vectorAuxD,smearedGauge); // later pass here the parameters for gaussian smearing
    
    printfQuda("Going to invert UP for component %d\n", isc);
    solverUP->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propUP.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);
    propUP3D.absorbVectorTimeSlice(vectorAuxF,global_fixSinkTime,isc/3, isc%3); // later when we have copy from 4D specific time slice we change it
    
    printfQuda("Going to invert DN for component %d\n", isc);
    solverDN->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propDN.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);
    propDN3D.absorbVectorTimeSlice(vectorAuxF,global_fixSinkTime,isc/3, isc%3);// later when we have copy from 4D specific time slice we change it
  }


  //smear the 3D propagators
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c2 = 0 ; c2 < 3 ; c2++){
      // later when we have copy from 4D specific time slice we change it
      vectorAuxF.copyPropagator3D(propUP3D, global_fixSinkTime, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge);
      vectorAuxF.copy(vectorOut);
      propUP3D.absorbVectorTimeSlice(vectorAuxF,global_fixSinkTime,nu, c2);

      vectorAuxF.copyPropagator3D(propDN3D, global_fixSinkTime, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge);
      vectorAuxF.copy(vectorOut);
      propDN3D.absorbVectorTimeSlice(vectorAuxF,global_fixSinkTime,nu, c2);
    }

  WHICHPARTICLE nucleon = NEUTRON; // for the test is NEUTRON, later we can provide an option
  
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
	vectorAuxF.apply_gamma5();
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverDN->solve(vectorOut, vectorIn); else solverUP->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqProp.absorbVectorToDevice(vectorAuxF, nu, c2);
      }
    seqProp.apply_gamma5();
    seqProp.conjugate();
    seqProp.communicateGhost();
    pGauge.communicateGhost();
    if(nucleon == PROTON) propUP.communicateGhost(); else propDN.communicateGhost();
    int signProps = (nucleon == PROTON) ? +1: -1;
    std::vector<GAMMAS> lgammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
    if(nucleon == PROTON) nucleonThrpLocal_CP2.contractNucleonThrp(seqProp, propUP, signProps, lgammas, 0, MOMENTUM_SPACE); // 0 is isource change later to do many source pos
    else nucleonThrpLocal_CP2.contractNucleonThrp(seqProp, propDN, signProps, lgammas, 0, MOMENTUM_SPACE);
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpLocal_CP2.getTotalSize()*2; iv++) (nucleonThrpLocal_CP2.getCorr())[iv] *= signPer;
      
    nucleonThrpLocal_CP2.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_local_CP2.dat");
    // do the contractions also for the conserved and oneD
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
	vectorAuxF.apply_gamma5();
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverUP->solve(vectorOut, vectorIn); else solverDN->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqProp.absorbVectorToDevice(vectorAuxF, nu, c2);
      }
    seqProp.apply_gamma5();
    seqProp.conjugate();
    seqProp.communicateGhost();
    pGauge.communicateGhost();
    if(nucleon == PROTON) propDN.communicateGhost(); else propUP.communicateGhost();
    int signProps = (nucleon == PROTON) ? -1: +1;
    std::vector<GAMMAS> lgammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
    if(nucleon == PROTON) nucleonThrpLocal_CP1.contractNucleonThrp(seqProp, propDN, signProps, lgammas, 0, MOMENTUM_SPACE); // 0 is isource change later to do many source pos
    else nucleonThrpLocal_CP1.contractNucleonThrp(seqProp, propUP, signProps, lgammas, 0, MOMENTUM_SPACE);
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpLocal_CP1.getTotalSize()*2; iv++) (nucleonThrpLocal_CP1.getCorr())[iv] *= signPer;
    nucleonThrpLocal_CP1.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_local_CP1.dat");
    // do the contractions also for the conserved and oneD
  }

  // smear the forward props
  // rotate to physical basis
  // do contractions for the two point functions
  
  delete solverUP;
  delete solverDN;
  finalize();

  return 0;
}

  // int v3 = GK_localVolume/GK_localL[3];
  // int v = GK_localVolume;
  // propDN3D.unload();
  
  // printfQuda("%+e %+e\n",propDN3D.H_elem()[0], propDN3D.H_elem()[1]);

  // FILE *ptr_out = NULL;
  // ptr_out = fopen("/onyx/noether/h/khadjiyiannakou/runs/seqSource.dat","w");
  // if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
  // WHICHPARTICLE nucleon = NEUTRON;

  // for(int isc = 0; isc < 12; isc++){ 
  //   vectorAuxF.zero_device();
  //   if(nucleon == PROTON)
  //     vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, P4_P, nucleon, global_fixSinkTime, isc/3, isc%3);
  //   else
  //     vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, P4_P, nucleon, global_fixSinkTime, isc/3, isc%3);
  //   vectorAuxF.unload();
  //   for(int nu = 0; nu < 4 ; nu++)
  //     for(int c2 = 0; c2 < 3; c2++)
  // 	for(int iv3 = 0 ; iv3 < v3; iv3++)
  // 	  if(is_myST) fprintf(ptr_out, "%+e %+e\n", vectorAuxF.H_elem()[nu*3*v*2 + c2*v*2 + my_fixSinkTime*v3*2 + iv3*2], vectorAuxF.H_elem()[nu*3*v*2 + c2*v*2 + my_fixSinkTime*v3*2 + iv3*2 + 1]);
  // }

  // for(int isc = 0; isc < 1; isc++){ 
  //   vectorAuxF.zero_device();
  //   if(nucleon == PROTON)
  //     vectorAuxF.seqSourceNucleon(propUP3D, P4_P, nucleon, global_fixSinkTime, isc/3, isc%3);
  //   else
  //     vectorAuxF.seqSourceNucleon(propDN3D, P4_P, nucleon, global_fixSinkTime, isc/3, isc%3);
  //   vectorAuxF.unload();
  //   for(int nu = 0; nu < 4 ; nu++)
  //     for(int c2 = 0; c2 < 3; c2++)
  // 	for(int iv3 = 0 ; iv3 < v3; iv3++)
  // 	  if(is_myST) fprintf(ptr_out, "%+e %+e\n", vectorAuxF.H_elem()[nu*3*v*2 + c2*v*2 + my_fixSinkTime*v3*2 + iv3*2], vectorAuxF.H_elem()[nu*3*v*2 + c2*v*2 + my_fixSinkTime*v3*2 + iv3*2 + 1]);
  // }
