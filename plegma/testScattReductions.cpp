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
  std::string outfile_V2;
  std::string outfile_V4;
  std::string path_V="";
  std::string path_P="";
  
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outProp", "Path for saving the propagator used", verbosity, outfile_S);
  HGC_options->set("outV3", "Path for saving the result of V3_reduction", verbosity, outfile_V3);
  HGC_options->set("outV2", "Path for saving the result of V3_reduction", verbosity, outfile_V2);
  HGC_options->set("outV4", "Path for saving the result of V3_reduction", verbosity, outfile_V4);
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
    
    //do reductions
    std::vector<int> mom={0,0,1};
    int Qmax=1;
    //std::vector<GAMMAS> glist2={G1,G2,G3,G4};
    //std::vector<GAMMAS> glist1={G4,G5};
    std::vector<GAMMAS> glist={G1,G2,G3,G4};
    PLEGMA_ScattCorrelator<float> reductions(MOMENTUM_SPACE, mom);
    double t0 = MPI_Wtime();

    for(int j=0; j<10;j++)
      reductions.V2( vectorStoc, glist, propUP, propUP);

    reductions.writeHDF5(outfile_V2);
    PLEGMA_printf("Time elapsed for V2 kernel = %g s\n",(MPI_Wtime()-t0)/10);

    t0 = MPI_Wtime();
    for(int j=0; j<10;j++)
      reductions.V3( vectorStoc, glist, propUP);

    reductions.writeHDF5(outfile_V3);
    PLEGMA_printf("Time elapsed for V3 kernel = %g s\n",(MPI_Wtime()-t0)/10);

    t0 = MPI_Wtime();
    for(int j=0; j<10;j++)
      reductions.V4( vectorStoc, glist, propUP, propUP);

    reductions.writeHDF5(outfile_V4);
    PLEGMA_printf("Time elapsed for V4 kernel = %g s\n",(MPI_Wtime()-t0)/10);
    //PLEGMA_ScattCorrelator<float> reductionV2(MOMENTUM_SPACE, mom);
    //reductionV2.V2( vectorStoc, glist, propUP, propUP);
    //reductionV2.writeHDF5(outfile_V2);
    
    /*{
      PLEGMA_ScattCorrelator<float> reductionV2(MOMENTUM_SPACE, mom);
      reductionV2.V2( vectorStoc, glist1, propUP, propUP);
      reductionV2.writeHDF5(outfile_V2+"_1mom_gl1_c0");
      reductionV2.V2( vectorStoc, glist2, propUP, propUP);
      reductionV2.writeHDF5(outfile_V2+"_1mom_gl2_c0");
    }
    {
      PLEGMA_ScattCorrelator<float> reductionV3(MOMENTUM_SPACE, mom);
      reductionV3.V3( vectorStoc, glist1, propUP);
      reductionV3.writeHDF5(outfile_V3+"_1mom_gl1_c0");
      reductionV3.V3( vectorStoc, glist2, propUP);
      reductionV3.writeHDF5(outfile_V3+"_1mom_gl2_c0");   
    }
    {
      PLEGMA_ScattCorrelator<float> reductionV4(MOMENTUM_SPACE, mom);
      reductionV4.V4( vectorStoc, glist1, propUP ,propUP);
      reductionV4.writeHDF5(outfile_V4+"_1mom_gl1_c0");
      reductionV4.V4( vectorStoc, glist2, propUP,propUP);
      reductionV4.writeHDF5(outfile_V4+"_1mom_gl2_c0");   
    }
    {
      PLEGMA_ScattCorrelator<float> reductions(MOMENTUM_SPACE, mom);
      reductions.V2( vectorStoc, glist1, propUP, propUP);
      reductions.writeHDF5(outfile_V2+"_1mom_gl1_c1");
      reductions.V3( vectorStoc, glist1, propUP);
      reductions.writeHDF5(outfile_V3+"_1mom_gl1_c1");
      reductions.V4( vectorStoc, glist1, propUP, propUP);
      reductions.writeHDF5(outfile_V4+"_1mom_gl1_c1");
    }
    {
      PLEGMA_ScattCorrelator<float> reductions(MOMENTUM_SPACE, Qmax);
      reductions.V2( vectorStoc, glist1, propUP, propUP);
      reductions.writeHDF5(outfile_V2+"_Qmax_gl1_c1");
      reductions.V3( vectorStoc, glist1, propUP);
      reductions.writeHDF5(outfile_V3+"_Qmax_gl1_c1");
      reductions.V4( vectorStoc, glist1, propUP, propUP);
      reductions.writeHDF5(outfile_V4+"_Qmax_gl1_c1");
    }
    */
  }
  finalize();
  
  return 0;
}


