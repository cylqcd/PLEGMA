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
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  printfQuda("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  
  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver solverUP(mu);
  mu*=-1.;
  QUDA_solver solverDN(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> propUP;
  PLEGMA_Propagator<float> propDN;

  for(int isource = 0 ; isource < numSourcePositions; isource++){
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
	       isource, sourcePositions[isource][0], sourcePositions[isource][1],
	       sourcePositions[isource][2], sourcePositions[isource][3]);

    for(int isc = 0 ; isc < 12 ; isc++){
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      
      printfQuda("Going to invert UP for component %d\n", isc);
      solverUP.solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propUP.absorb(vectorAuxF, isc/3, isc%3);

      printfQuda("Going to invert DN for component %d\n", isc);
      solverDN.solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut,smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propDN.absorb(vectorAuxF, isc/3, isc%3);
    }

    propUP.rotateToPhysicalBase_device(+1);
    propDN.rotateToPhysicalBase_device(-1);
    propUP.applyBoundaries_device(sourcePositions[isource][3]);
    propDN.applyBoundaries_device(sourcePositions[isource][3]);

    PLEGMA_Correlator<float> corr(corr_space, maxQsq);
    corr.contractMesons(propUP, propDN, sourcePositions[isource]);
    corr.writeFile(twop_filename.c_str(), corr_file_format);

    corr.contractBaryons(propUP, propDN, sourcePositions[isource]);
    corr.writeFile(twop_filename.c_str(), corr_file_format);
  }

  finalize();
  return 0;
}

