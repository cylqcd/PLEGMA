#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  int nsmearAPE = 20;
  double alphaAPE = 0.5;
  double xiMomSm = 0.6; //remember the sign later
  std::vector<int> moms = {0,0,1,0};
    
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

  // put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*moms[i]/GK_totalL[i])*I);
  smearedGauge.scaleDirWise(momSmScale);

  PLEGMA_Gauge<float> WLGauge;
  WLGauge.pack(gauge.get_ptr());
  WLGauge.load();

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
  PLEGMA_Propagator<float> *seqPropOut = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;
  
  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  PLEGMA_Correlator<float> *nucleonThrpWLP_CP1 = new PLEGMA_Correlator<float>[GK_totalL[2]]; // if is the z direction
  PLEGMA_Correlator<float> *nucleonThrpWLP_CP2 = new PLEGMA_Correlator<float>[GK_totalL[2]];

  PLEGMA_Correlator<float> *nucleonThrpWLM_CP1 = new PLEGMA_Correlator<float>[GK_totalL[2]];
  PLEGMA_Correlator<float> *nucleonThrpWLM_CP2 = new PLEGMA_Correlator<float>[GK_totalL[2]];

  PLEGMA_Correlator<float> *nucleonThrpWL_CP2[2];
  nucleonThrpWL_CP2[0] = nucleonThrpWLP_CP2;
  nucleonThrpWL_CP2[1] = nucleonThrpWLM_CP2;

  PLEGMA_Correlator<float> *nucleonThrpWL_CP1[2];
  nucleonThrpWL_CP1[0] = nucleonThrpWLP_CP1;
  nucleonThrpWL_CP1[1] = nucleonThrpWLM_CP1;


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

  PLEGMA_Su3field<float> su3;
  PLEGMA_Su3field<float> WL;
  PLEGMA_Su3field<float> tmp;

  //seq source part 2Props and contraction block
  {
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	if(nucleon == PROTON)
	  vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, P4_P, nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
	else
	  vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, P4_P, nucleon, global_fixSinkTime, nu, c2);
	// !!!!!!!!!!!!!!!!!! put a momentum in the sink later
	vectorAuxF.conjugate();
	vectorAuxF.apply_gamma5();
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverDN->solve(vectorOut, vectorIn); else solverUP->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqPropOut->absorbVectorToDevice(vectorAuxF, nu, c2);
      }
    seqPropOut->apply_gamma5();
    seqPropOut->conjugate();
    
    int signProps = (nucleon == PROTON) ? +1: -1;

    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propUP : propDN;
    seqPropOut->unload();
    std::vector<GAMMAS> gammas = {G4};
    
    su3.absorbDir_device(WLGauge, 2); // only for z direction
    WL.setUnit( (std::vector<int>) {0,4,8});
    for(int i = 0 ; i < GK_totalL[2]/2;i++){ // GK_totalL[2] only for z direction
      nucleonThrpWL_CP2[0][i].contractNucleonThrp_wilsonLine(*seqPropOut, propF, signProps, WL, gammas, 0, MOMENTUM_SPACE);
      nucleonThrpWL_CP2[0][i].writeASCII( ("/onyx/noether/h/khadjiyiannakou/runs/threep_PDFs_CP2_Plus" + std::to_string(i) + ".dat").c_str() );
      propExchange = seqPropIn; seqPropIn = seqPropOut; seqPropOut = propExchange;
      WL.wilsonLineUpdate(su3, tmp, 4+2); // build Wilson line in the +z direction
      seqPropOut->shift(*seqPropIn, 4+2);      
    }

    seqPropOut->load();
    su3.absorbDir_device(WLGauge, 2); // only for z direction
    WL.setUnit( (std::vector<int>) {0,4,8});
    for(int i = 0 ; i < GK_totalL[2]/2;i++){ // GK_totalL[2] only for z direction
      nucleonThrpWL_CP2[1][i].contractNucleonThrp_wilsonLine(*seqPropOut, propF, signProps, WL, gammas, 0, MOMENTUM_SPACE);
      nucleonThrpWL_CP2[1][i].writeASCII( ("/onyx/noether/h/khadjiyiannakou/runs/threep_PDFs_CP2_Minus" + std::to_string(i) + ".dat").c_str() );
      propExchange = seqPropIn; seqPropIn = seqPropOut; seqPropOut = propExchange;
      WL.wilsonLineUpdate(su3, tmp, 2); // build Wilson line in the +z direction
      seqPropOut->shift(*seqPropIn, 2);      
    }

  }

  delete seqPropOut;
  delete seqPropIn;
  
  delete[] nucleonThrpWLP_CP1;
  delete[] nucleonThrpWLP_CP2;

  delete[] nucleonThrpWLM_CP1;
  delete[] nucleonThrpWLM_CP2;

  finalize();
  return 0;
}

