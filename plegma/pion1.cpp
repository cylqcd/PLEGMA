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
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsrc","src-filename","tSinks","twop-filename", "maxQsq"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int n_stochastic_samples;
  int nroots=4;
  int rand_seed1=1234;
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);

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


    std::vector<int> tmp_mom0={0,0,0};
    momList mom0List(1,tmp_mom0,{0,});

    //List of gammas
    std::vector<GAMMAS_SCATT> glist_sink_meson={G_1,G_2,G_3};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_1,G_2,G_3};


    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);
  
    PLEGMA_printf("Start producing stochastic vectors and propagators\n");

    for(size_t its = 0; its < tSinks.size(); its++){
      int tsink = tSinks[its];
      PLEGMA_printf("\n ### Calculations for stochastic source %d - %02d begin now ###\n\n",
                    its, tsink);

      char * src_string;
      asprintf(&src_string, "_id%02d_st%03d", its, tsink);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);

      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
	continue;
      }
      
      //We draw a different random vector for every source position
      vectorStoc_source_oet.stochastic_Z(nroots);

      //Store zero momentum oet propagators
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
      {
         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;  

         vectortmp1.absorbTimeslice(vectorStoc_source_oet, tsink);

	 //Dilution
         vectortmp2.dilutespin(vectortmp1,0);

	 //Transforming to physical base
         //vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorStoc_source_oet.copy(vectortmp2);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.copy(vectorStoc_source_oet);
           TIME(solver.solve(vectortmp2, vectortmp2));
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
	   stochastic_propagator_momzero[spinindex].copy(vectortmp1);
           if (spinindex<3){
             vectortmp1.diluteSpinDisplace(vectorStoc_source_oet,spinindex+1,spinindex);
             vectorStoc_source_oet.copy(vectortmp1);
           }
         }
      }

      //P diagram
      site source=site({0,0,0,tsink});
      PLEGMA_ScattCorrelator<float> corrP(source, mom0List);
      corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");

      TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momzero, 0));

      //write P
      TIME(corrP.apply_sign("P"));
      TIME(corrP.writeHDF5( outfilename ));
    }

    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);

      site source=sourcePositions[isource];
      PLEGMA_Propagator<float> propUP;

      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);

      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
	continue;
      }

      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut;
	PLEGMA_Vector<float> vectorAuxF;
	vectorInOut.pointSource(source, isc/3, isc%3, DEVICE);
	// Inverting
	PLEGMA_printf("Going to invert for component %d\n", isc);
	TIME(solver.solve(vectorInOut, vectorInOut));
	vectorAuxF.copy(vectorInOut);
	propUP.absorb(vectorAuxF, isc/3, isc%3);
      }
      
      propUP.rotateToPhysicalBase_device(+1);
      propUP.applyBoundaries_device(source[DIM_T]);

      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      TIME(corr.contractMesonsNew(propUP, propUP));

      //write P
      TIME(corr.writeHDF5( outfilename ));
    } //loop over source position
  } 
  finalize();

  return 0;
}
