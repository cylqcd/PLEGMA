#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

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
  smearedGauge.APEsmearing(pGauge, 10, 0.1, 3);
  smearedGauge.calculatePlaq();

  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver *solverUP = new QUDA_solver(mu);
  mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  //  PLEGMA_Propagator<float> propUP;
  // PLEGMA_Propagator<float> propDN;
  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  // for the test use tsink = 10;
  int  isource=0;
  int my_fixSinkTime = (10 + params.sourcePosition[isource][3])%GK_totalL[3] - comm_coords(default_topo)[3] * GK_localL[3];
  //  int my_fixSinkTime = (10 + 0)%GK_totalL[3] - comm_coords(default_topo)[3] * GK_localL[3];
  bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < GK_localL[3] );

  for(int isc = 0 ; isc < 12 ; isc++){
    vectorAuxD.pointSource(params.sourcePosition[isource], isc/3, isc%3, DEVICE);
    vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
    
    printfQuda("Going to invert UP for component %d\n", isc);
    solverUP->solve(vectorOut, vectorIn);
    vectorAuxD.gaussianSmearing(vectorOut,smearedGauge); // Fix later to do not do smearing here
    vectorAuxF.copy(vectorAuxD);
    //propUP.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);
    propUP3D.absorbVectorTimeSlice(vectorAuxF,my_fixSinkTime,isc/3, isc%3);
    
    printfQuda("Going to invert DN for component %d\n", isc);
    solverDN->solve(vectorOut, vectorIn);
    vectorAuxD.gaussianSmearing(vectorOut,smearedGauge); // Fix later to do not do smearing here
    vectorAuxF.copy(vectorAuxD);
    propDN3D.absorbVectorTimeSlice(vectorAuxF,my_fixSinkTime,isc/3, isc%3);
    //    propDN.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);
  }

  FILE *ptr_out = NULL;
  ptr_out = fopen("/onyx/noether/h/khadjiyiannakou/runs/seqSource.dat","w");
  if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
  WHICHPARTICLE nucleon = NEUTRON;

  if(is_myST)
    for(int isc = 0; isc < 1; isc++){ // do mu = 0 and c1 =0 for now
      if(nucleon == PROTON)
	vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, P4_P, nucleon, my_fixSinkTime, isc/3, isc%3);
      else
	vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, P4_P, nucleon, my_fixSinkTime, isc/3, isc%3);
      vectorAuxF.unload();
      int v3 = GK_localVolume/GK_localL[3];
      for(int nu = 0; nu < 4 ; nu++)
	for(int c2 = 0; c2 < 3; c2++)
	  for(int iv3 = 0 ; iv3 < v3; iv3++)
	    fprintf(ptr_out, "%+e %+e\n", vectorAuxF.H_elem()[nu*3*v3*2 + c2*v3*2 + iv3*2], vectorAuxF.H_elem()[nu*3*v3*2 + c2*v3*2 + iv3*2 + 1]);
    }
  
  finalize();

  return 0;
}
