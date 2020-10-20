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

  std::vector<size_t> ls = {0,};
  HGC_options->set("ls-wave-function", "Wave function limit parameter l", verbosity, ls);
  
  std::vector<int> b = {0,0,0};
  HGC_options->set("b-wave-function", "Wave function displacement parameter b", verbosity, b);

  std::vector<int> z = {0,0,0};
  HGC_options->set("z-wave-function", "Wave function parameter z", verbosity, z);


  
  //=========================================================================================================//
  initializePLEGMA();

  for(auto l:ls)
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


      
      

      auto computeStaple = [&](PLEGMA_Su3field<float>& staple,PLEGMA_Gauge<float> &gaugeF,int l_d){

      			     staple.setUnit( (std::vector<int>) {0,4,8});
			     {
      			       PLEGMA_Su3field<float> *su3 = new PLEGMA_Su3field<float>(BOTH);
      			       PLEGMA_Su3field<float> *tmp = new PLEGMA_Su3field<float>(BOTH);
      			       su3->absorbDir_device(gaugeF, z_dir);
			       if(l_d>0){
				 staple.Udag(*su3);
				 for(int j=1;j<l_d;j++){
				   tmp->shift(staple,4+z_dir);
				   staple.UxUdag(*tmp,*su3);
				 }      				 
      			       }

			       delete su3;
			       delete tmp;
			     }

			     {
      			       PLEGMA_Su3field<float> *su3 = new PLEGMA_Su3field<float>(BOTH);
      			       PLEGMA_Su3field<float> *tmp = new PLEGMA_Su3field<float>(BOTH);
      			       su3->absorbDir_device(gaugeF, b_dir);
			       for(int j=0;j<b[b_dir];j++){
				 tmp->UxU(staple,*su3);
				 staple.shift(*tmp,b_dir);
			       }      				 
			     
			       delete su3;
			       delete tmp;
			     }


			     {
      			       PLEGMA_Su3field<float> *su3 = new PLEGMA_Su3field<float>(BOTH);
      			       PLEGMA_Su3field<float> *tmp = new PLEGMA_Su3field<float>(BOTH);
      			       su3->absorbDir_device(gaugeF, z_dir);
			       for(int j=0;j<l_d+z[z_dir];j++){
				 tmp->UxU(staple,*su3);
				 staple.shift(*tmp,z_dir);
			       }      				 
			       
			       delete su3;
			       delete tmp;
			     }
			     
			     
			     
      			   };


      auto computeStaplePath = [&](PLEGMA_Su3field<float>& staple,PLEGMA_Gauge<float> &gIn,int l_d){

				 staple.setUnit( (std::vector<int>) {0,4,8});
				 PLEGMA_Su3field<float> tmp(BOTH);
				 PLEGMA_Su3field<float> *u_s[4];
				 for(int idir = 0; idir < 4 ; idir++){
				   u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
				   u_s[idir]->absorbDir_device(gIn,idir);
				 }

				 int len_path = 2*l_d+z[z_dir]+b[b_dir];
				 if(len_path>0){
				   int spath[len_path];
				   for(int i=0;i<l_d;i++)
				     spath[i]=4+z_dir;
				   for(int i=l_d;i<l_d+b[b_dir];i++)
				     spath[i]=b_dir;
				   for(int i=l_d+b[b_dir];i<len_path;i++)
				     spath[i]=z_dir;
				   std::vector<int> vspath(spath,spath+len_path);
				   staple.path(vspath, u_s, tmp);
				 }
				 for(int idir = 0; idir < 4 ; idir++){
				   delete u_s[idir];
				 }

			       };

      
      //Apply shift to propagator
      auto shiftPropagator = [&](PLEGMA_Propagator<float>* propF,PLEGMA_Propagator<float>* propIn,PLEGMA_Propagator<float>* propExchange){

			       propF->unload();

			       for(int j=0;j<z[z_dir];j++){
				 propExchange = propIn; propIn = propF; propF = propExchange;
				 TIME(propF->shift(*propIn, z_dir));
			       }

			       for(int j=0;j<b[b_dir];j++){
				 propExchange = propIn; propIn = propF; propF = propExchange;
				 TIME(propF->shift(*propIn, b_dir));
			       }
			       propF->load();
			     };


      PLEGMA_Propagator<float> shifted_propUP(BOTH);
      shifted_propUP.copy(propUP);

      PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
      PLEGMA_Propagator<float> *propExchange = nullptr;

      TIME(shiftPropagator(&shifted_propUP,propIn,propExchange));
      delete propIn;

      shifted_propUP.rotateToPhysicalBase_device(+1);
      propUP.rotateToPhysicalBase_device(+1);
      shifted_propUP.applyBoundaries_device(source[3]);
      propUP.applyBoundaries_device(source[3]);
      
      for(auto l:ls){
	//If stout smearing is needed we have to allocate a new gauge field
	PLEGMA_Gauge<float> gaugeWL;
	gaugeWL.copy(gauge);
	//Apply here stout smearing if needed
	PLEGMA_Su3field<float> WL;
	//TIME(computeStaple(WL,gaugeWL,l));
	TIME(computeStaplePath(WL,gaugeWL,l));
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	corr.setFixMomVec(sinkMom);
	TIME(corr.contractTMDWFMesons(propUP,shifted_propUP,WL,l));
	THREAD(corr.writeFile(twop_filename, corr_file_format));
      }

      
      // propUP.rotateToPhysicalBase_device(+1);
      // propDN.rotateToPhysicalBase_device(-1);
      // propUP.applyBoundaries_device(source[3]);
      // propDN.applyBoundaries_device(source[3]);
    
      // PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      // corr.setFixMomVec(sinkMom);
      // TIME(corr.contractTMDWFMesons(propUP,shifted_propUP,WL));
      // THREAD(corr.writeFile(twop_filename, corr_file_format));
    
    }
  
  }
  finalize();
  return 0;
}

