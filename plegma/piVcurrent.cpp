
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

using namespace plegma;
using namespace quda;

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "readStochSamples","time-dilution","nstochSamples","confnumber"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  bool timedilution;
  bool readstochastic;
  int n_stochastic_samples;
  int nroots=4;
  int confnumber_int;
  int rand_seed1=1234;
  int rand_seed2=1234;
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
  std::string outdiagramPrefix="";
  std::string outfile_V3;
  std::string outfile_V2;
  std::string outfile_V4;
  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("readStochSamples", "Flag for switching read/building stochastic propagators", verbosity, readstochastic);
  HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outPropUP", "Path for saving the up propagator used", verbosity, outfile_upS);
  HGC_options->set("outPropDN", "Path for saving the dn propagator used", verbosity, outfile_dnS);
  HGC_options->set("outPropSeq", "Path for saving the sequential propagator used", verbosity, outfile_SEQ);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);

  //=========================================================================================================//
  initializePLEGMA();
  {

    //Storing only the smeared gauge

    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));


    //Get the confnumber for latfile
    char *ssource;
    asprintf(&ssource,"%04d", confnumber_int);
    std::string confnumber= ssource;
    free(ssource);

    /*
    //Reading the momentum lists
    PLEGMA_printf("###Momentum list read from : %s", pathListMomenta.c_str());
    momList sourcemomentumList(1,pathListMomenta,{0,});
    PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList.size());

    if(sourcemomentumList.empty())
      PLEGMA_error("momentumList empty");
    */

    std::vector<int> tmp_mom0={0,0,0};
    momList mom0List(1,tmp_mom0,{0,});
    
    //List of gammas
    std::vector<GAMMAS_SCATT> glist_sink_meson={G_1,G_2,G_3};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_1,G_2,G_3};
    


    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);

    PLEGMA_printf("Start producing stochastic vectors and propagators\n");
    //Note that we replace the f1<-f2 DN propagator with a stochastic one
    //in two steps actually
    //DN(x_f1 <- x_f2 ) = \phihat(x_f2)(x_f1)\xi^{dagger}(x_f2)(x_f2)
    //where x_f2 is the source
    //      x_f1 is the sink
    //so \phihat(x_f2)(x_f1) is the x_f1 coordinate of the stochastic 
    //propagator created at x_f2 for the down quark
    //=gamma_5*U(x_f2 <- x_f1)^dagger*gamma_5
    //=gamma_5*\xi(x_f1)(x_f1)*\phi(x_f2)(x_f1)^dagger*gamma_5
    //Here we compute phi and xi
    //Producing the stochastic source
   

    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      vectorStoc_source_oet.stochastic_Z(nroots);

      int sequential_time_source=sourcePositions[isource][DIM_T];

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource; 
      free(ssource);

      site source=site({0,0,0,sourcePositions[isource][3]});
      std::string outfilename;


      //Store zero momentum oet propagators
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
      {
         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;          
 
         vectortmp1.absorbTimeslice(vectorStoc_source_oet, sequential_time_source);

	 //Dilution     
         vectortmp2.dilutespin(vectortmp1,0);
 
         //Transforming to physical base
         vectortmp1.rotateToPhysicalBasis(vectortmp2,+1); 

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorStoc_source_oet.copy(vectortmp1);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.copy(vectorStoc_source_oet);
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
           //Gaussian smearing of the propagator
           //TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
	   stochastic_propagator_momzero[spinindex].copy(vectortmp1);
           //stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+confnumber+"propagator_"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){
             vectortmp1.diluteSpinDisplace(vectorStoc_source_oet,spinindex+1,spinindex);
             vectorStoc_source_oet.copy(vectortmp1);
           }
         }
         
      }
      
      //P diagram
      PLEGMA_ScattCorrelator<float> corrP(sourcePositions[isource], mom0List);
      corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");

      TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momzero, 0));

      //write P
      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrP.apply_sign("P"));
      TIME(corrP.writeHDF5( outfilename ));

    } //loop over source position


  } 
  finalize();
  
  return 0;
}
