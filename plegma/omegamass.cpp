
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
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename","confnumber"};
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
  std::string outdiagramPrefix="";
  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  PLEGMA_printf("Initialization");

  //=========================================================================================================//
  initializePLEGMA();
  {

    //Storing only the smeared gauge
    PLEGMA_Gauge<double> smearedGauge;

    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.calculatePlaq();

      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();

      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));


    //Get the confnumber for latfile
    char *ssource;
    asprintf(&ssource,"%04d", confnumber_int);
    std::string confnumber= ssource;
    free(ssource);

    //Reading the momentum lists
    PLEGMA_printf("###Momentum list read from : %s", pathListMomenta.c_str());
    momList sourcemomentumList(3,pathListMomenta,{1,2});
    PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList.size());

    if(sourcemomentumList.empty())
      PLEGMA_error("momentumList empty");


    //List of gammas
    std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4};
    std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);



    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, sourcePositions[isource][DIM_T]);
      int sequential_time_source=sourcePositions[isource][DIM_T];

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource; 
      free(ssource);
  
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH);
      PLEGMA_Propagator<float> propDN(BOTH);

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }
    
      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float>  vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;
        { // Smearing the source
          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
        }

        //Rotation to the physical basis
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));       

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        //Rotation to the physical basis
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1));

        //Smearing at the sink
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

        vectorAuxF.copy(vectorAuxD);
        propUP.absorb(vectorAuxF, isc/3, isc%3);
      }

      if(mu>0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float>  vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;
        { // Smearing the source
          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
        }

        //Rotation to the physical basis for the DN quark
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1));

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        //Rotation to the physical basis for the DN quark
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1));

        //Smearing at the sink
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

        vectorAuxF.copy(vectorAuxD);
        propDN.absorb(vectorAuxF, isc/3, isc%3);
      }



      std::vector<int> mom={0,0,0};
      
      site source=site({0,0,0,sourcePositions[isource][3]});
      std::string outfilename;

      //D diagram
      {
	std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
	momList list_mtot(1,{mtot,},{0,});
	PLEGMA_ScattCorrelator<float> corrD(sourcePositions[isource], list_mtot);

	//initialize diagram
	corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,"D");
      
	PLEGMA_ScattCorrelator<float> reductionsT1(source, mtot);
	PLEGMA_ScattCorrelator<float> reductionsT2(source, mtot);

	TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));

	TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));

	//write UP,UP,UP
	outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPUP";
	
	TIME( corrD.D_diagramms( reductionsT1, reductionsT2 ));
	TIME( corrD.apply_phase() );
	TIME( corrD.applyBoundaryConditions( true ) );
	TIME( corrD.writeHDF5(outfilename) );

        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propDN, propDN));

        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propDN, propDN));

        //write DN,DN,DN
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNDNDN";

        TIME( corrD.D_diagramms( reductionsT1, reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));

        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));

        //write UP,DN,UP
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPDNUP";

        TIME( corrD.N_diagramms( reductionsT1, reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propUP, propDN));

        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propDN));

        //write DN,UP,DN
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNUPDN";

        TIME( corrD.N_diagramms( reductionsT1, reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        //write UP,DN,DN
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPDNDN";

        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propDN, propDN));
        TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        //write DN,UP,UP
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNUPUP";

        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
        TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

      }    

    } //loop over source position

  } 
  finalize();
  
  return 0;
}
