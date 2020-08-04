#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","Projs","xiMomSm","gammas","sinkMom"};

  initializeOptions(argc, argv, true, listOpt);

  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  size_t l = 1;
  HGC_options->set("l-wave-function", "Wave function limit parameter l", verbosity, l);
  
  std::vector<int> b = {0,0,0};
  HGC_options->set("b-wave-function", "Wave function displacement parameter b", verbosity, b);

  std::vector<int> z = {0,0,0};
  HGC_options->set("z-wave-function", "Wave function parameter z", verbosity, z);

  
  //=========================================================================================================//
  initializePLEGMA();

  if(l<0) PLEGMA_error("The wave function limit parameter l has to be positive\n");
  for(size_t i=0; i < b.size() ; i ++)
    if(b[i]*sinkMom[i]!=0)
      PLEGMA_error("The displacement parameter b and the momentum have to be perpendicular\n");

  double z_mod = sqrt(pow(z[0],2)+pow(z[1],2)+pow(z[2],2));
  double P_mod = sqrt(pow(sinkMom[0],2)+pow(sinkMom[1],2)+pow(sinkMom[2],2));

  
  for(size_t i=0; i < z.size() ; i ++)
    if(z[i]/z_mod-sinkMom[i]/P_mod!=0 && z_mod!=0)
      PLEGMA_error("The parameter z and the momentum have to be parallel\n");

  ///This first implementation requires z and sinkMom to have just one non-zero component
  auto IsNonZero = [](int i) { return i!=0;}; 

  int ZsNonZero = std::count_if (z.begin(), z.end(), IsNonZero);

  if(ZsNonZero>1) PLEGMA_error("At the moment just one non-zero component of the vector z is allowed\n");

  int PNonZero = std::count_if (sinkMom.begin(), sinkMom.end(), IsNonZero);

  if(PNonZero>1) PLEGMA_error("At the moment just one non-zero component of the momentum vector is allowed\n");

  signed short int z_dir = 0;
  signed short int b_dir = 0;
  for(size_t i=0; i < z.size() ; i ++){
    if(z[i]!=0)
      z_dir = i;
    if(b[i]!=0)
      b_dir = i;
  }
  
  
  
  
  {
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
    plaqQuda();
 
  
    //Smearing
    PLEGMA_Gauge<double> smearedGauge;
    smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
    PLEGMA_printf("Plaquette after smearing:\n");
    smearedGauge.calculatePlaq();


    
    float SourcePhase;
   

    
    PLEGMA_printf("Sink Momentum px %d, py %d, pz %d, pt %d\n",
		  sinkMom[0],sinkMom[1],sinkMom[2],sinkMom[3]);
    if(maxQsq < sinkMom[0]*sinkMom[0]+sinkMom[1]*sinkMom[1]+sinkMom[2]*sinkMom[2])
      PLEGMA_error("maxQsq does not include the sink momentum\n");

    
  
    //Momentum smearing: put the momentum phase to smeared gauge field
    std::complex<double> momSmScale[N_DIMS];
    std::complex<double> I(0,1);
    for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
    TIME(smearedGauge.scaleDirWise(momSmScale));
  
  
    // ensuring mu positive
    if(mu<0)  mu*=-1.;
    TIME(QUDA_solver solver(mu));

    PLEGMA_Propagator<float> propUP(BOTH);
    PLEGMA_Propagator<float> propDN(BOTH);
    PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
    PLEGMA_Propagator<float> propDN_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
  
    
    PLEGMA_Gauge<double> *AuxSinkGauge;
    AuxSinkGauge = &smearedGauge;

    for(int isource=0;isource<numSourcePositions;isource++) {
      site& source = sourcePositions[isource];
      
      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu) {
				 // ensuring mu value
				 if(mu != run_mu) {
				   mu = run_mu;
				   solver.UpdateSolver();
				 }
				 for(int isc = 0 ; isc < 12 ; isc++){
				   PLEGMA_Vector<double> vectorInOut;
				   { // Smearing the source
				     PLEGMA_Vector3D<double> vector1, vector2;
				     vector1.pointSource(source, isc/3, isc%3, DEVICE);
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
				     vectorInOut.absorb(vector2,source[DIM_T]);
				   }
				   // Inverting
				   PLEGMA_printf("Going to invert %s for component %d\n",
						 run_mu>0 ? "UP" : "DN", isc);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   if(prop_SL.getAllocation() != NONE) {
				     PLEGMA_Vector<float> vectorAuxF;
				     vectorAuxF.copy(vectorInOut);
				     prop_SL.absorb(vectorAuxF, isc/3, isc%3);
				   }
				   { // Smearing the solution
				     PLEGMA_Vector<double> vectorAuxD;
				     PLEGMA_Vector<float> vectorAuxF;
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
				     vectorAuxF.copy(vectorAuxD);
				     prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				   }
				 }
			       };
    
      TIME(computePropagator(propUP, propUP_SL, mu>0 ? mu : -mu));
      TIME(computePropagator(propDN, propDN_SL, mu<0 ? mu : -mu));


      
      

      auto computeStaple = [&](PLEGMA_Su3field<float>& staple,PLEGMA_Gauge<float>& gaugeF){

			     staple.setUnit( (std::vector<int>) {0,4,8});

			     {
			       PLEGMA_Su3field<float> su3;
			       PLEGMA_Su3field<float> tmp;
			       su3.absorbDir_device(gaugeF, z_dir);
			       for(int j=0;j<l;j++)
				 staple.wilsonLineUpdate(su3, tmp, z_dir);
			     }

			     {
			       PLEGMA_Su3field<float> su3;
			       PLEGMA_Su3field<float> tmp;
			       su3.absorbDir_device(gaugeF, b_dir);
			       for(int j=0;j<b[b_dir];j++)
				 staple.wilsonLineUpdate(su3, tmp, 4+b_dir);
			     }

			     {
			       PLEGMA_Su3field<float> su3;
			       PLEGMA_Su3field<float> tmp;
			       su3.absorbDir_device(gaugeF, z_dir);
			       for(int j=0;j<l+z[z_dir];j++)
				 staple.wilsonLineUpdate(su3, tmp, 4+z_dir);
			     }
			   };

      //If stout smearing is needed we have to allocate a new gauge field
      PLEGMA_Gauge<float> gaugeWL;
      gaugeWL.copy(gauge);
      //Apply here stout smearing if needed
      PLEGMA_Su3field<float> WL;
      TIME(computeStaple(WL,gaugeWL));
      

      //Apply shift to propagator
      auto shiftPropagator = [&](PLEGMA_Propagator<float>* propF){

			       PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
			       PLEGMA_Propagator<float> *propExchange = nullptr;
			       propF->unload();

			       for(int j=0;j<z[z_dir];j++){
				 propExchange = propIn; propIn = propF; propF = propExchange;
				 TIME(propF->shift(*propIn, 4+z_dir));
			       }

			       for(int j=0;j<b[b_dir];j++){
				 propExchange = propIn; propIn = propF; propF = propExchange;
				 TIME(propF->shift(*propIn, 4+b_dir));
			       }
			       
			   };


      PLEGMA_Propagator<float> shifted_propUP(BOTH);
      shifted_propUP.copy(propUP);
      TIME(shiftPropagator(&shifted_propUP));
      shifted_propUP.rotateToPhysicalBase_device(+1);
      shifted_propUP.applyBoundaries_device(source[3]);
			     
      propUP.rotateToPhysicalBase_device(+1);
      propDN.rotateToPhysicalBase_device(-1);
      propUP.applyBoundaries_device(source[3]);
      propDN.applyBoundaries_device(source[3]);
    
      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      corr.setFixMomVec(sinkMom);
      TIME(corr.contractTMDWFMesons(propUP,shifted_propUP,WL));
      THREAD(corr.writeFile(twop_filename, corr_file_format));
    
    }
  
  }
  finalize();
  return 0;
}

