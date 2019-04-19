#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};
  
int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFromLime(latfile.c_str());
      gauge.load();
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();
      
      // Smearing
      smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
    }
    QUDA_solver solver(mu);
    
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      PLEGMA_Propagator<float> propUP;
      // ensuring mu positive
      if(mu<0) {
	mu*=-1.;
	solver.UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	PLEGMA_Vector<float> vectorAuxF;
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
	
	PLEGMA_printf("Going to invert UP for component %d\n", isc);
	solver.solve(vectorInOut, vectorInOut);
	vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
	vectorAuxF.copy(vectorAuxD);
	propUP.absorb(vectorAuxF, isc/3, isc%3);
      }	

      PLEGMA_Propagator<float> propDN;
      // ensuring mu negative
      if(mu>0) {
	mu*=-1.;
	solver.UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	PLEGMA_Vector<float> vectorAuxF;
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
	
	PLEGMA_printf("Going to invert DN for component %d\n", isc);
	solver.solve(vectorInOut, vectorInOut);
	vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmearGauss, alphaGauss);
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

      corr.contractBaryonsProj(propUP, propDN, propUP, propDN, sourcePositions[isource]);
      corr.writeFile(twop_filename.c_str(), corr_file_format);
    }
  }
  
  finalize();
  return 0;
}

