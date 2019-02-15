#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true);
  plaqQuda();

  // Smearing
  PLEGMA_Gauge<double> smearedGauge(BOTH);
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
      propUP.absorb(vectorAuxF, isc/3, isc%3);

      printfQuda("Going to invert DN for component %d\n", isc);
      solverDN->solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut,smearedGauge);
      vectorAuxF.copy(vectorAuxD);
      propDN.absorb(vectorAuxF, isc/3, isc%3);
    }

    propUP.rotateToPhysicalBase_device(+1);
    propDN.rotateToPhysicalBase_device(-1);
    propUP.applyBoundaries_device(params.sourcePosition[isource][3]);
    propDN.applyBoundaries_device(params.sourcePosition[isource][3]);

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

