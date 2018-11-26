#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  PLEGMA_params params;
  initialize(argc, argv, &params);

  // Setting the QUDA params as read from command line
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  
  //-Read the gauge field in lime format
  GaugeBuffer<double> gauge(params);
  readLimeGauge(gauge.get_ptr(), latfile, &gauge_param, params.procs);

  // This gauge will be used for the inversions.
  // We need to apply the anti-periodic boundaries.
  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);

  // Load the gauge field into QUDA
  initGaugeQuda((void*)gauge.get_ptr(), gauge_param);

  // Removing anti-periodic boundaries.
  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  mapEvenOddToNormalGauge(gauge.get_ptr(),gauge_param,params.lL);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> readGauge(BOTH);
  PLEGMA_Gauge<double> smearedGauge(BOTH);
  readGauge.pack(gauge.get_ptr());
  readGauge.load();
  printfQuda("Plaquette before smearing:\n");
  readGauge.calculatePlaq();
  smearedGauge.APEsmearing(readGauge, 0, 0.1, 3); // TODO: here should go the smearing params
  printfQuda("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  
  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver *solverUP = new QUDA_solver(mu);
  mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn(BOTH);
  PLEGMA_Vector<double> vectorOut(BOTH);
  PLEGMA_Vector<double> vectorAuxD(BOTH);
  PLEGMA_Vector<float> vectorAuxF(BOTH);
  PLEGMA_Propagator<float> propUP(BOTH);
  PLEGMA_Propagator<float> propDN(BOTH);
  PLEGMA_Correlator<float> corr;

  for(int isource = 0 ; isource < params.Nsources ; isource++){
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
	       isource, params.sourcePosition[isource][0], params.sourcePosition[isource][1],
	       params.sourcePosition[isource][2], params.sourcePosition[isource][3]);

    for(int isc = 0 ; isc < 12 ; isc++){
      vectorAuxD.pointSource(params.sourcePosition[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
      
      printfQuda("Going to invert UP for component %d\n", isc);
      solverUP->solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut,smearedGauge);
      vectorAuxF.copy(vectorAuxD);
      propUP.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);

      printfQuda("Going to invert DN for component %d\n", isc);
      solverDN->solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut,smearedGauge);
      vectorAuxF.copy(vectorAuxD);
      propDN.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);
    }

    propUP.rotateToPhysicalBase_device(+1);
    propDN.rotateToPhysicalBase_device(-1);
    propUP.removeBoundaries_device(params.sourcePosition[isource][3]);
    propDN.removeBoundaries_device(params.sourcePosition[isource][3]);

    corr.contractMesons(propUP, propDN, isource, params.CorrSpace);
    corr.writeFile(params);

    corr.contractBaryons(propUP, propDN, isource, params.CorrSpace);
    corr.writeFile(params);
  }

  delete solverUP;
  delete solverDN;
  
  finalize();
  return 0;
}

