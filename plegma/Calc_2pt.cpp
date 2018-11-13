#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <quda_params.h>
#include <quda_solver.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  PLEGMA_params params;
  read_command_line(argc, argv, &params);
  
  // initialize QMP/MPI, QUDA comms grid and RNG 
  initComms(argc, argv, params.procs);

  // initialize the QUDA library
  initQuda(device);
  print_info();

  // initialize PLEGMA info
  initialize(&params);
  print_status();

  // Setting the QUDA params as read from command line
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  
  double *gauge[4];
  double *gauge_APE[4];
  int *lL = params.lL;
  size_t V = lL[0]*lL[1]*lL[2]*lL[3];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = (double*) malloc(V*gaugeSiteSize*sizeof(double));
    gauge_APE[dir] = (double*) malloc(V*gaugeSiteSize*sizeof(double));
  }
  //-Read the gauge field in lime format
  readLimeGauge(gauge, latfile, &gauge_param, params.procs);

  // This gauge will be used for the inversions.
  // We need to apply the anti-periodic boundaries.
  applyBoundaryCondition(gauge, V/2 ,&gauge_param);

  // Load the gauge field into QUDA
  initGaugeQuda((void*)gauge, gauge_param);

  //-Read the smeared gauge field in lime format
  // TODO: create locally the smeared gauge
  readLimeGauge(gauge_APE, latfile_smeared, &gauge_param, params.procs);
  mapEvenOddToNormalGauge(gauge_APE,gauge_param,lL[0],lL[1],lL[2],lL[3]);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> smearedGauge(BOTH);
  smearedGauge.packGauge(gauge_APE);
  smearedGauge.loadGauge();
  printfQuda("Plaquette of smeared config:\n");
  smearedGauge.calculatePlaq();

  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver solverUP(mu);

  // ensuring mu negative
  if(mu>0) mu*=-1.;
  QUDA_solver solverDN(mu);

  PLEGMA_Vector<double> vectorIn(BOTH);
  PLEGMA_Vector<double> vectorOut(BOTH);
  PLEGMA_Vector<double> vectorAuxD(BOTH);
  PLEGMA_Vector<float> vectorAuxF(BOTH);
  PLEGMA_Propagator<float> propUP(BOTH);
  PLEGMA_Propagator<float> propDN(BOTH);
  PLEGMA_Correlator<float> corrMesons;

  for(int isource = 0 ; isource < params.Nsources ; isource++){
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
	       isource,
	       params.sourcePosition[isource][0],
	       params.sourcePosition[isource][1],
	       params.sourcePosition[isource][2],
	       params.sourcePosition[isource][3]);

    for(int isc = 0 ; isc < 12 ; isc++){
      vectorAuxD.pointSource(params.sourcePosition[isource], isc/3, isc%3);
      vectorAuxD.loadVector();
      vectorIn.gaussianSmearing(vectorAuxD,smearedGauge);
      
      printfQuda("Going to invert UP for component %d\n", isc);
      solverUP.solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut,smearedGauge);
      vectorAuxF.copy(vectorAuxD);
      propUP.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);

      printfQuda("Going to invert DN for component %d\n", isc);
      solverDN.solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut,smearedGauge);
      vectorAuxF.copy(vectorAuxD);
      propDN.absorbVectorToDevice(vectorAuxF, isc/3, isc%3);
    }

    propUP.rotateToPhysicalBase_device(+1);
    propDN.rotateToPhysicalBase_device(-1);

    corrMesons.contractMesons(propUP, propDN, isource, params.CorrSpace);

    char* filename, *str;
    if(params.CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",params.Q_sq);
    asprintf(&filename,"%s_mesons_%s_SS.%02d.%02d.%02d.%02d" ,
	    twop_filename, str,
	    params.sourcePosition[isource][0],
	    params.sourcePosition[isource][1],
	    params.sourcePosition[isource][2],
	    params.sourcePosition[isource][3]);
    free(str);
    corrMesons.writeFile(filename, &params, params.CorrFileFormat);
    free(filename);
  }
  
  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
    free(gauge_APE[i]);
  }
  
  // finalize the QUDA library
  finalizeGaugeQuda();
  endQuda();
  
  // finalize the communications layer
  finalizeComms();

  return 0;
}

