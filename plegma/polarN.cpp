#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <cuda_profiler_api.h>
#define getsign(x) (abs(x)/x)
using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE","nsmear-gauss", "alpha-gauss","nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

int main(int argc, char **argv)
{
  initializeOptions(argc, argv); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ========//
  bool isDirichlet=false;
  HGC_options->set("isDirichlet", "Enable the Dirichlet boundary conditions", verbosity, isDirichlet);
  double qdn_E_Dirichlet=0.0001;
  HGC_options->set("qdn_E_Dirichlet", "Enable the Dirichlet boundary conditions", verbosity, qdn_E_Dirichlet);
  
  int nE = 1;
  HGC_options->set("nE", "The integer index of the electric field only if isDirichlet=false. Both signs will be computed", verbosity, nE);
  int dirEfield=2;
  HGC_options->set("dirEfield", "The direction of the Electric Field (0,1,2) -> (x,y,z)", verbosity, dirEfield);
  //=========================================================================================================//
  initializePLEGMA();
  std::vector<int> nVals;
  if(nE == 0) nVals.push_back(nE);
  else{
    nVals.push_back(abs(nE));
    nVals.push_back(-abs(nE));
  }
  // Allocation done on BOTH, DEVICE and HOST

  PLEGMA_Gauge<double> gauge(BOTH), smearedGauge(BOTH);
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();
  TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  if(isDirichlet){
    PLEGMA_U1Gauge<double> u1gauge(BOTH);
    PLEGMA_Gauge<double> smearedGaugeTmp(BOTH);
    u1gauge.constField(dirEfield,3,0);
    u1gauge.modifyBoundaries(dirEfield,-1,0); // we do not need to put the temporal to zero since it does not appear in the Gaussian smearing
    smearedGaugeTmp.U3xU1(smearedGauge,u1gauge);
    smearedGauge.copy(smearedGaugeTmp);
  }

  // if(isDirichlet){

  //   PLEGMA_Gauge<double> gaugetmp(BOTH);
  //   u1gauge.constField(dirEfield,3,0);
  //   u1gauge.modifyBoundaries(3,-1,0);
  //   u1gauge.modifyBoundaries(dirEfield,-1,0);
  //   gaugetmp.U3xU1(gauge,u1gauge);
  //   TIME(initGaugeQuda(gaugetmp, true));
  // }
  // else{
  TIME(initGaugeQuda(gauge, true));
    //  }
  TIME(QUDA_solver solver(mu));
  
  for(int nn : nVals){
    std::string twop_f = twop_filename + std::string("_") + std::to_string(nn);
    PLEGMA_U1Gauge<double> u1gauge_dn(BOTH), u1gauge_up(BOTH);

    double qdn_E;
    if(isDirichlet) qdn_E = nn==0? 0:  getsign(nn)*qdn_E_Dirichlet;
    else qdn_E = - (2. * nn * PI) / (HGC_totalL[3]*HGC_totalL[dirEfield]);
    u1gauge_dn.constField(dirEfield,3,qdn_E,sourcePositions[0][3]);
    if(isDirichlet){
      u1gauge_dn.modifyBoundaries(3,-1,0);
      u1gauge_dn.modifyBoundaries(dirEfield,-1,0);
    }
    else u1gauge_dn.modifyBoundaries(3,dirEfield,-qdn_E);
    u1gauge_dn.calculatePlaq();

    double qup_E=-2*qdn_E;
    u1gauge_up.constField(dirEfield,3,qup_E,sourcePositions[0][3]);
    if(isDirichlet){
      u1gauge_up.modifyBoundaries(3,-1,0);
      u1gauge_up.modifyBoundaries(dirEfield,-1,0);
    }
    else u1gauge_up.modifyBoundaries(3,dirEfield,-qup_E);
    u1gauge_up.calculatePlaq();

    PLEGMA_Gauge<double> gauge_u3u1_dn(BOTH), gauge_u3u1_up(BOTH);
    gauge_u3u1_dn.U3xU1(gauge,u1gauge_dn);
    gauge_u3u1_dn.calculatePlaq();
    gauge_u3u1_up.U3xU1(gauge,u1gauge_up);
    gauge_u3u1_up.calculatePlaq();
    PLEGMA_printf("--------------------\n");


    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      site& source = sourcePositions[isource];
      PLEGMA_Propagator<double> propUP(BOTH);
      PLEGMA_Propagator<double> propDN(BOTH);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);
      //----------------------------------------------------------
      auto computePropagator = [&](PLEGMA_Propagator<double>& prop,
				   const double run_mu, PLEGMA_Gauge<double> &gg, int nSmear, int sign) {
				 TIME(updateGaugeQuda(gg,true));
				 mu = run_mu;
				 solver.UpdateSolver();
				 for(int isc = 0 ; isc < 12 ; isc++){
				   PLEGMA_Vector<double> vectorInOut;
				   { // Smearing the source
				     PLEGMA_Vector3D<double> vector1, vector2;
				     vector1.pointSource(source, isc/3, isc%3, DEVICE);
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
				     vectorInOut.absorb(vector2,source[DIM_T]);
				   }
				   // Inverting
				   PLEGMA_printf("Going to invert for component %d\n", isc);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   { // Smearing the solution
				     PLEGMA_Vector<double> vectorAuxD;
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
				     prop.absorb(vectorAuxD, isc/3, isc%3);
				   }
				 }  
				 prop.rotateToPhysicalBase_device(sign);
				 prop.applyBoundaries_device(source[DIM_T]);
			       };
      PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
      TIME(computePropagator(propUP, abs(mu), gauge_u3u1_up, nsmearGauss,+1));
      TIME(computePropagator(propDN, -abs(mu), gauge_u3u1_dn, nsmearGauss,-1));
      corr.contractBaryons(propUP, propDN);
      corr.writeFile(twop_f, corr_file_format);
    }
  }
  finalize();
 
  return 0;
}
