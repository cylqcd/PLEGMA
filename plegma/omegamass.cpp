
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
  std::vector<double> mu_s;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  bool timedilution;
  bool readstochastic;
  int n_stochastic_samples;
  int nroots=4;
  int confnumber_int;
  bool run_ud = true;
  int device_id;
  std::string outdiagramPrefix="";
  HGC_options->set("device-id", "which device we want to run (if you do not want to specify put -2 here", verbosity, device_id);

  HGC_options->set("run-ud", "Whether to run '+' **AND** '-' flavors or only '+' flavor", verbosity, run_ud);
  HGC_options->set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);


  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  PLEGMA_printf("Initialization");

  //=========================================================================================================//
  initializePLEGMA(device_id);
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

    updateOptions(STRANGE);
    mu = mu_s[0];
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


    int nSmaller = mu_s.size();
    //loop over all the strange quarks
    for (int ismall=0; ismall<nSmaller;++ismall){

      mu=mu_s[ismall];
      PLEGMA_printf("Strange=%e\n", mu);

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }
      else{
        solver.UpdateSolver();
      }

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

        {

           site& source_calc = sourcePositions[isource];

           PLEGMA_Propagator<float> propDN(NONE);


           PLEGMA_Correlator<float> corr(corr_space, source_calc, 1);

           TIME(corr.contractBaryonsUDSC(propDN, propDN, propUP, propDN, false, false));
           char * group;

           asprintf(&group, "baryons_udsc_omega/Oms[%+1.1e]", mu);
           corr.setGroups(group);
           free(group);
           std::string twopf= twop_filename + confnumber;
           THREAD(corr.writeFile(twopf, corr_file_format));

 
        }


        site source=site({0,0,0,sourcePositions[isource][3]});
        std::string outfilename;

        //D diagram
        {
          std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
	  momList list_mtot(1,{mtot,},{0,});
	  PLEGMA_ScattCorrelator<float> corrD(sourcePositions[isource], list_mtot);

	  //initialize diagram
	
          char *dset1;
          asprintf(&dset1, "Oms[%+1.1e]", mu);
	  corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,dset1);
      
	  PLEGMA_ScattCorrelator<float> reductionsT1(source, mtot);
	  PLEGMA_ScattCorrelator<float> reductionsT2(source, mtot);


	  //write UP,UP,UP

	  TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));
          TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));



          TIME( corrD.D_diagramms( reductionsT1, reductionsT2 ));

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_O";

          TIME( corrD.apply_phase() );
          TIME( corrD.applyBoundaryConditions( true ) );
          TIME( corrD.writeHDF5(outfilename) );
       
	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPUP_T1";
          TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));

          TIME( corrD.apply_phase() );
          TIME( corrD.applyBoundaryConditions( true ) );
          TIME( corrD.writeHDF5(outfilename) );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPUP_T2";
          TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));

          TIME( corrD.apply_phase() );
          TIME( corrD.applyBoundaryConditions( true ) );
          TIME( corrD.writeHDF5(outfilename) );


          //if we invert both + and - flavors
          //then we write out every possible factors to build
          //diagram for the delta with Cgi, Cgigt insertions
          //in particular you can build up from the factors
          //all the isospin combinations
          if (run_ud == true){
            //write UP,UP,UP
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPUP_T1";
            TIME( corrD.writeHDF5(outfilename) );

            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPUP_T2";
            TIME( corrD.writeHDF5(outfilename) );

            PLEGMA_Propagator<float> propDN(BOTH);
            //ensuring mu negative
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
              //Rotation to the physical basis
              TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));
              //Inversion
              PLEGMA_printf("Going to invert DN for component %d\n", isc);
              TIME(solver.solve(vectorAuxD, vectorAuxD));
              //Rotation to the physical basis
              TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1));
              //Smearing at the sink
              TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
              vectorAuxF.copy(vectorAuxD);
              propDN.absorb(vectorAuxF, isc/3, isc%3);
            }
            //ensuring mu positive
            if(mu<0) {
                mu*=-1.;
                solver.UpdateSolver();
            }
            //write DN,DN,DN
          
            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNDNDN_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propDN, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNDNDN_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propDN, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            //write UP,DN,UP

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPDNUP_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPDNUP_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );
 
            //write DN,UP,DN

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNUPDN_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propUP, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNUPDN_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            //write UP,UP,DN

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPDN_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPUPDN_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            //write DN,DN,UP

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNDNUP_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propDN, propUP));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNDNUP_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propDN, propUP));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            //write DN,UP,UP
          
            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNUPUP_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DNUPUP_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            //write UP,DN,DN

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPDNDN_T1";
            TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propDN, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

            outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_UPDNDN_T2";
            TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propDN, propDN));
            TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
            TIME( corrD.apply_phase() );
            TIME( corrD.applyBoundaryConditions( true ) );
            TIME( corrD.writeHDF5(outfilename) );

          
          }//if run_ud

        }//loop over D diagram

      } //loop over source position

    }//loop over strange quarks

  } 
  finalize();
  
  return 0;
}
