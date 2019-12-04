#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  std::string outfile_V="";
  std::string outfile_S="";
  std::string outfile_V3;
  std::string path_V="";
  std::string path_P="";
  
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outProp", "Path for saving the propagator used", verbosity, outfile_S);
  HGC_options->set("outV3", "Path for saving the result of V3_reduction", verbosity, outfile_V3);
  HGC_options->set("loadVector", "Path for loading V", verbosity, path_V);
  HGC_options->set("loadProp", "Path for loading P", verbosity, path_P);

  //=========================================================================================================//
  initializePLEGMA();
  {
    //int nsmearAPE=4;
    //float alphaAPE=0.1;
    //int nsmearGauss=4;
    //float alphaGauss=0.1;
    int source[4]={4,3,12,1};

    //Create Propagator
    PLEGMA_Propagator<float> propUP(BOTH);
    if (path_P==""){
      PLEGMA_printf("Build propagator from scratch\n");     
      // Allocation done on BOTH, DEVICE and HOST
      //PLEGMA_Gauge<double> smearedGauge(BOTH);
      //{
      PLEGMA_Gauge<double> gauge(BOTH);
      // Reading from Lime file and loading to device
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();

      // Smearing
      //smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
      //PLEGMA_printf("Plaquette after smearing:\n");
      //smearedGauge.calculatePlaq();

      //}
  
      QUDA_solver solver(mu);
    
      /*if(mu != mu_ud) {
	for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_factor[i] = mu_ud_factor[i];
	mu = mu_ud;
	solver.UpdateSolver();
      }*/
      
      //create propagator
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	PLEGMA_Vector<float> vectorAuxF;
      
	vectorAuxD.pointSource( source, isc/3, isc%3, DEVICE);
      
	//vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
	//solver.solve(vectorInOut, vectorInOut);
	solver.solve(vectorAuxD,vectorAuxD);
	//vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
	vectorAuxF.copy(vectorAuxD);
	propUP.absorb(vectorAuxF, isc/3, isc%3);
      }
      //propUP.unload();
      //propUP.writeLIME(outfile_S);

      if(outfile_S!="")
	{
	  PLEGMA_printf("Save propagator\n");
	  PLEGMA_Vector<float> vectorAuxPrint(BOTH);
	  for(int isc = 0 ; isc < 12 ; isc++){
	    std::string spin=std::to_string(isc/3);
	    std::string col=std::to_string(isc%3);
	    
	    vectorAuxPrint.absorb(propUP,isc/3,isc%3);
	    vectorAuxPrint.unload();
	    vectorAuxPrint.writeLIME(outfile_S+"_s"+spin+"_c"+col);
	  }
	}
    }
    else {
      PLEGMA_printf("Read propagator from: %s\n",path_P.c_str());     
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<float> vectorRead(BOTH);
	std::string spin=std::to_string(isc/3);
	std::string col=std::to_string(isc%3);
	vectorRead.readFile(path_P+"_s"+spin+"_c"+col,LIME_FORMAT);
	vectorRead.load();
	propUP.absorb(vectorRead, isc/3, isc%3);
      }
    }
    
    //create vector field
    PLEGMA_Vector<float> vectorStoc(BOTH);
    if(path_V==""){
      PLEGMA_printf("Build vector from scratch\n");     
      int nroots=4;
      QUDA_solver solver(mu);
      vectorStoc.randInit(1234);
      vectorStoc.stochastic_Z(nroots);
      solver.solve(vectorStoc, vectorStoc);

      if(outfile_V!=""){
	PLEGMA_printf("Print Vector\n");     
	vectorStoc.unload();
	vectorStoc.writeLIME(outfile_V);
      }
    }
    else{
      PLEGMA_printf("Read Vector from: %s\n",path_V.c_str());     
      vectorStoc.readFile(path_V, LIME_FORMAT);
      vectorStoc.load();
    }
    
    //do V3 reduction
    std::vector<int> mom={0,0,1};
    std::vector<GAMMAS> glist2={G1,G2,G3,G4};
    std::vector<GAMMAS> glist1={G4};
    PLEGMA_ScattCorrelator<float> V3reduction(MOMENTUM_SPACE, mom);
    
    V3reduction.V3( vectorStoc, glist1, propUP);
    //V3reduction.writeHDF5(outfile_V3);
    V3reduction.writeHDF5(outfile_V3+"_1");

    V3reduction.V3( vectorStoc, glist2, propUP);
    V3reduction.writeHDF5(outfile_V3+"_2");

    
    

  }
  finalize();

  return 0;
}


