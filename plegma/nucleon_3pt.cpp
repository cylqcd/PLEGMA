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

  PLEGMA_Gauge<double> *pGauge = new PLEGMA_Gauge<double>();
  pGauge->pack(gauge.get_ptr());
  pGauge->load();
  pGauge->calculatePlaq();
  
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(*pGauge, nsmearAPE, alphaAPE, 3);
  smearedGauge.calculatePlaq();
  delete pGauge;
  
  PLEGMA_Gauge<float> contractGauge;
  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  contractGauge.pack(gauge.get_ptr());
  contractGauge.load();
  
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
  PLEGMA_Correlator<float> nucleonThrpOneD_CP1;
  PLEGMA_Correlator<float> nucleonThrpOneD_CP2;
  PLEGMA_Correlator<float> nucleonThrpNoe_CP1;
  PLEGMA_Correlator<float> nucleonThrpNoe_CP2;

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
    propUP.absorb(vectorAuxF, isc/3, isc%3);
    //    propUP3D.absorbVectorTimeSlice(vectorAuxF,global_fixSinkTime,isc/3, isc%3); // later when we have copy from 4D specific time slice we change it
    
    printfQuda("Going to invert DN for component %d\n", isc);
    solverDN->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propDN.absorb(vectorAuxF, isc/3, isc%3);
    //    propDN3D.absorbVectorTimeSlice(vectorAuxF,global_fixSinkTime,isc/3, isc%3);// later when we have copy from 4D specific time slice we change it
  }


  //smear the 3D propagators
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c2 = 0 ; c2 < 3 ; c2++){
      // later when we have copy from 4D specific time slice we change it
      //      vectorAuxF.copyPropagator3D(propUP3D, global_fixSinkTime, nu, c2);
      vectorAuxF.absorb(propUP, global_fixSinkTime, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge);
      vectorAuxF.copy(vectorOut);
      propUP3D.absorb(vectorAuxF,global_fixSinkTime,nu, c2);

      //      vectorAuxF.copyPropagator3D(propDN3D, global_fixSinkTime, nu, c2);
      vectorAuxF.absorb(propDN, global_fixSinkTime, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge);
      vectorAuxF.copy(vectorOut);
      propDN3D.absorb(vectorAuxF,global_fixSinkTime,nu, c2);
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
	vectorAuxF.apply_gamma(G5);
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
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
    nucleonThrpLocal_CP2.contractNucleonThrp_local(seqProp, propF, signProps, 0, MOMENTUM_SPACE); // 0 is isource change later 
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpLocal_CP2.getTotalSize()*2; iv++) (nucleonThrpLocal_CP2.getCorr())[iv] *= signPer;      
    nucleonThrpLocal_CP2.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_local_CP2.dat");

    // ONED contractions
    nucleonThrpOneD_CP2.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, 0, MOMENTUM_SPACE); // 0 is isource change lat
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpOneD_CP2.getTotalSize()*2; iv++) (nucleonThrpOneD_CP2.getCorr())[iv] *= signPer;      
    nucleonThrpOneD_CP2.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_oneD_CP2.dat");

    // noe contractions
    nucleonThrpNoe_CP2.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps, 0, MOMENTUM_SPACE); // 0 is isource change lat
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpNoe_CP2.getTotalSize()*2; iv++) (nucleonThrpNoe_CP2.getCorr())[iv] *= signPer;      
    nucleonThrpNoe_CP2.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_noe_CP2.dat");

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
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
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
    nucleonThrpLocal_CP1.contractNucleonThrp_local(seqProp, propF, signProps, 0, MOMENTUM_SPACE); // 0 is isource change later
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpLocal_CP1.getTotalSize()*2; iv++) (nucleonThrpLocal_CP1.getCorr())[iv] *= signPer;
    nucleonThrpLocal_CP1.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_local_CP1.dat");

    //ONED
    nucleonThrpOneD_CP1.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, 0, MOMENTUM_SPACE); // 0 is isource change la
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpOneD_CP1.getTotalSize()*2; iv++) (nucleonThrpOneD_CP1.getCorr())[iv] *= signPer;
    nucleonThrpOneD_CP1.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_oneD_CP1.dat");

    //ONED
    nucleonThrpNoe_CP1.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps, 0, MOMENTUM_SPACE); // 0 is isource change la
    if(signPer < 0) for(int iv = 0 ; iv < nucleonThrpNoe_CP1.getTotalSize()*2; iv++) (nucleonThrpNoe_CP1.getCorr())[iv] *= signPer;
    nucleonThrpNoe_CP1.writeASCII("/onyx/noether/h/khadjiyiannakou/runs/threep_noe_CP1.dat");

    // do the contractions also for the conserved
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
