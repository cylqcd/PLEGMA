
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

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name, 
                    int n_stochastic_samples ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions( true ));
  TIME(source.normalize_nstoch(n_stochastic_samples));
  TIME(source.writeHDF5( outputFilename ));

}
void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name ){
  TIME(source.apply_phase());
  TIME(source.apply_sign(diagram_name));
  TIME(source.applyBoundaryConditions( true ));
  TIME(source.writeHDF5( outputFilename ));
}

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
    /*******************************************************************************
     *
     *
     *  Initialization: (1) reading the sourcepositions and the source momentum list 
     *                  (2) reading the gaugefield uploading the quda
     *                      produce the smeared gauge field
     *                      only the smeared gauge field will be stored in PLEGMA
     *                  (3) setting up the list of gammas
     *                      (1) we use three triplets for the delta Cgi, Cgigt, Cgigtg5
     *                      (2) we use C,Cg5, Cg4,Cg5g4 for the nucleon
     *                      (3) we use g5 for meson(pion)
     *
     *******************************************************************************/
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
    std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4,CG_1_G_4_G_5,CG_2_G_4_G_5,CG_3_G_4_G_5};
    std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4,CG_1_G_4_G_5,CG_2_G_4_G_5,CG_3_G_4_G_5};
    std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5,C,CG_5_G_4,CG_4};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5,C,CG_5_G_4,CG_4};
    std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

/******************************************************************************************
*
*  In the first part of the code we compute 
*      (1) the stochastic propagators for nstoch random volume sources using full-time dilution
*      (2) allocate space for zero momentum (u,d) and finite momentum (u) oet propagators
*      (3) compute loops for the pi0 using the stochastic propagators and sources produced by 
*          time dilution, note that this is needed for the 
*  Note that in both cases we store a standard vector of PLEGMA_Vectors on the host, 
*  and need to load to the device in case we need them
*
*
*******************************************************************************************/

    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_zero_mom;
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_fini_mom;
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_zero_mom;

    for(int i=0; i< 4; ++i) {
      stochastic_oet_prop_u_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_u_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_d_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
    }


    //Computing time-diluted stochastic propagators and stochastic source

    std::vector<PLEGMA_Vector<float>*> stochastic_sources;
    std::vector<PLEGMA_Vector<float>*> stochastic_propags;

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_propags.push_back(new PLEGMA_Vector<float>(HOST));
    }

    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);

    PLEGMA_printf("Start producing stochastic vectors and propagators\n");
    //Note that we replace the f1<-f2 DN propagator with a stochastic one
    //in two steps actually
    //DN(x_f1 <- x_f2 ) = \phihat(x_f2)(x_f1)\xi^{dagger}(x_f2)(x_f2)
    //where x_f2 is the source
    //      x_f1 is the sink
    //      phihat is the DN propagator obtained by acting on xi
    //      xi is the stochastic source
    //In practice however we invert for the UP type flavour and use 
    //the gamma_5 trick:
    //DN(x_f1 <- x_f2) =gamma_5*U(x_f2 <- x_f1)^dagger*gamma_5
    //-->>
    //gamma_5*\xi(x_f1)(x_f1)*\phi(x_f2)(x_f1)^dagger*gamma_5
    //In the following lines we compute phi and xi
    //Note that in the following we do not apply gamma_5 to xi and
    //phi, because we also construct U(x_f1, x_f2) for the I=1/2 case
    //Producing the stochastic source
    if (readstochastic==0){
      
      PLEGMA_Vector<double> vectorAuxD1(BOTH);//For storing the source (rotated and smeared)
      PLEGMA_Vector<double> vectorAuxD2(BOTH);//For storing the propagotor for the time-slices
      PLEGMA_Vector<double> vectorInOut; //temporary vector using in solve
      PLEGMA_Vector<double> vectorSource(BOTH);//For storing the source 
      vectorSource.randInit(rand_seed2);

      for (int i=0; i<n_stochastic_samples; ++i){
        //Step(1) Creating the time-diluted stochastic source
        vectorSource.stochastic_Z(nroots);
      
        //Step(2) Save it on the host memory
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorSource);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
        }
        vectorAuxD1.copy(vectorSource);

        vectorAuxD1.unload();
        stochastic_sources[i]->copy(vectorAuxD1,HOST);
        vectorAuxD1.load();
        vectorAuxD1.copy(vectorSource);
     
        //Step(3) Smearing all the time slice
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));
 
        //Step(4) We rotate the source to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //In vectorAuxD2 we store the results for the inversion
        vectorAuxD2.scale(0.0);
    
        if (timedilution){
          PLEGMA_printf("#piNdiagramms: Full time dilution is turned on\n");
          for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
            //Step(5) pick out a particular timeslice from the source
            vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
            //Step(6) Solve
            TIME(solver.solve(vectorInOut, vectorInOut));
            //Step(7) absorbing the particular timeslice to a 4d vector
            vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
          }
        }
        else{
          PLEGMA_printf("#piNdiagramms: No time dilution is used n stochastic propagators\n");
          vectorInOut.copy(vectorAuxD1);
          TIME(solver.solve(vectorInOut, vectorInOut));
          vectorAuxD2.copy(vectorInOut);
        } 
    

        //Step(6) We rotate back the propagator to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //Step(7) Smearing all the time slice in the propagator
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(8) Save the propagator to the disk
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorAuxD2);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber);
        }

        //Step(9) Save the propagator to the host memory
        vectorAuxD2.unload();
        stochastic_propags[i]->copy(vectorAuxD2,HOST);
        vectorAuxD2.load();
        
      } //loop over the stochastic samples

    }
    else{
      for (int i=0; i<n_stochastic_samples; ++i){
        std::string inputfilename=outfile_V+"globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read stochastic source from: %s\n",inputfilename.c_str());
        PLEGMA_Vector<float> vectorRead(BOTH);
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        stochastic_sources[i]->copy(vectorRead,HOST);
        inputfilename=outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        stochastic_propags[i]->copy(vectorRead,HOST);
      }
    }

    //Creating loops for zero momentum
    //for the I=1/2 case we consider only momentum for the nucleon
    //and not for the pion, so compute the pi0 loops for only the zero momentum case


    site source_stoch=site({0,0,0,0});
    std::vector<int> zero_mom_list_pion={0,0,0};
    momList piN12_zeropion(1,{zero_mom_list_pion,},{0,});

 
    PLEGMA_ScattCorrelator<float> Loop_UPDN(source_stoch, piN12_zeropion);

    Loop_UPDN.initialize_diagram( glist_sink_meson, "L");

    for (int i=0; i<n_stochastic_samples; ++i){
      
      TIME(Loop_UPDN.Loop_diagramms( stochastic_propags[i], stochastic_sources[i], 0, true));

    }


    std::string outfilename;

    outfilename = outdiagramPrefix+confnumber+"_LoopUPDN";
    TIME(Loop_UPDN.normalize_nstoch(n_stochastic_samples));
    TIME(Loop_UPDN.writeHDF5( outfilename ));

/********************************************************************************************
*
*
*   In the second part of the code we compute point source propagators for
*   UP and DN flavour and perform all the contractions necessary for I=3/2 and 1/2
*   that does not require sequential source propagator and allocate space for the 
*   sequential propagator
*   Producing diagrams (1) N (Nucleon 2pt) (including both N+ with U,D,U and N- with D,U,D
*                      (2) D (Delta 2pt) 
*                      (3) delta -->> pi + N (2pt)
*   Producing factors requiring only D or U for momenta pf1,pf2 or {0,0,0}
*
*********************************************************************************************/




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
      PLEGMA_Propagator<float> propTS(BOTH);

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
      /*
      if(outfile_upS!="")
        {
          PLEGMA_printf("Save propagator for the up quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propUP,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            //vectorAuxPrint.writeHDF5(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
          }
        }
       */
      // ensuring mu negative
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

        //(3 step) rotation to the physical basis
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1));

        //(4 step) doing the inversion
        PLEGMA_printf("Going to invert DN for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        //(5 step) rotating to the physical base
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1));

        //(6 step) doing the smearing on the propagator
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

        vectorAuxF.copy(vectorAuxD);

        propDN.absorb(vectorAuxF, isc/3, isc%3);
      }

      /*
      if(outfile_dnS!="")
        {
          PLEGMA_printf("Save propagator for the d quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propDN,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_dnS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            //vectorAuxPrint.writeHDF5(outfile_dnS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);

          }
        }*/

      std::vector<int> mom={0,0,0};
      
      site source=site({0,0,0,sourcePositions[isource][3]});

      //D diagram
      {

        //For I=3/2 I_3=+3/2
	std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
	momList list_mtot(1,{mtot,},{0,});
	PLEGMA_ScattCorrelator<float> corrD(sourcePositions[isource], list_mtot);

	//initialize diagram
	corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,"D");
      
	PLEGMA_ScattCorrelator<float> reductionsT1(source, mtot);
	PLEGMA_ScattCorrelator<float> reductionsT2(source, mtot);

	TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));

	TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));

	//write D
	outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_D";
	
	TIME( corrD.D_diagramms( reductionsT1, reductionsT2 ));
	TIME( corrD.apply_phase() );
	TIME( corrD.apply_sign("D") );
	TIME( corrD.applyBoundaryConditions( true ) );
	TIME( corrD.writeHDF5(outfilename) );

        //For I=1/2 I_3=+1/2
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_DNUPUP_T2";
        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
        TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_DNUPUP_T1";
        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN, propUP, propUP));
        TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_UPUPDN_T1";
        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propDN));
        TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_UPDNUP_T1";
        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propDN, propUP));
        TIME( corrD.convertTreductiontoDiagram( reductionsT1 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_UPDNUP_T2";
        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propDN));
        TIME( corrD.convertTreductiontoDiagram( reductionsT2 ));
        TIME( corrD.apply_phase() );
        TIME( corrD.applyBoundaryConditions( true ) );
        TIME( corrD.writeHDF5(outfilename) );

      }
      
      //N diagram

      outfilename=outdiagramPrefix+confnumber+sourcepositiontext+"_N";

      std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
      momList list_mpf1(1,{mpf1,},{0,});
      PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1 );
      PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1 );

      //initialize diagram
      TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP"));
      TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0"));

      
      //Computing T reductions+recombination
      { 
        PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
        PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
        //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
        TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));

        TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));

        TIME(corrNP.N_diagramms( reductionsT1N, reductionsT2N ));
 
        //Secondly compute N_0 (neutron)(we need for M diagram (N0p+))
        TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN));
        
        TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN));

        TIME(corrN0.N_diagramms( reductionsT1N, reductionsT2N ));
      }


      //P diagram
      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
      momList list_mpi2(1,{mpi2,},{0,});
      PLEGMA_ScattCorrelator<float> corrP(sourcePositions[isource], list_mpi2);
      corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }


      //We can compute V2 contractions for B and V3 contraction for W first
      //without having to compute it for all the iterations in the loop
      //over the sequential momentum
      //Provided we have the same pf1,pf2 pairs for all pi2 sequential momentum
      std::vector<int> filter={0,0,0};
      //For the saved V2 and V3 reductions we have to use all possible unique pf1 and pf2
      //In case of I=1/2 we have set pf2={0,0,0}, however consider all possible pf1
      momList filtered_sourcemomentumList_pi20 = sourcemomentumList.extract(filter, 0);
     
      //Here the prefix UU means that reduction is based phi and xi, without the gamma_5
      //We replace U(x_f2,x_f1) with phi(x_f2) xi^dagger(x_f1)
      //phi goes to V2 reduction and xi goes to V3 reduction 
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1D_U;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U_zero_mom;
     
      //Here the prefix DD means that reduction is based phi*g5 and xi*g5
      //We replace D(x_f2,x_f1) with xi(x_f2)*gamma_5* phi^dagger(x_f1) *gamma_5
      //phi goes to V3 reduction and xi goes to V2 reduction 
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2U;
      //we need these only at zero momentum pf2
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D_zero_mom;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2U_zero_mom;

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_D;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1D_U;
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V4_GAMMAF1U_D;
 
      for(int i=0; i< n_stochastic_samples; ++i) {
        try
        {
          reductions_DD_V2_GAMMAF1U_U.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));

          reductions_UU_V2_GAMMAF1D_U.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));
          reductions_UU_V4_GAMMAF1D_U.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));
          reductions_UU_V4_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));
          reductions_UU_V3_GAMMAF2U_zero_mom.push_back(new PLEGMA_ScattCorrelator<float>(source, piN12_zeropion));

          reductions_DD_V3_GAMMAF2U_zero_mom.push_back(new PLEGMA_ScattCorrelator<float>(source, piN12_zeropion));
          reductions_DD_V3_GAMMAF2D_zero_mom.push_back(new PLEGMA_ScattCorrelator<float>(source, piN12_zeropion));
          reductions_DD_V3_GAMMAF2U.push_back(         new PLEGMA_ScattCorrelator<float>(source, sourcemomentumList.uniq_p(0)));

          reductions_DD_V4_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));
          reductions_DD_V2_GAMMAF1D_U.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));
          reductions_DD_V2_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList_pi20.uniq_p(1)));


        }
        catch(std::bad_alloc&){
          PLEGMA_printf("Memory allocation fails to store factors");
          exit(1); 
        }

      }

      for (int i=0; i<n_stochastic_samples; ++i){

        PLEGMA_Vector<float> stochastic_propagator;
        PLEGMA_Vector<float> stochastic_source;
            
        stochastic_propagator.copy(*stochastic_propags[i],HOST);
        stochastic_source.copy(*stochastic_sources[i],HOST);
            
        stochastic_propagator.load();
        stochastic_source.load();
        //For U(xf1,xf2)
        //B3,B5,B9,B11
        //D1ii1,D1ii3,D1ii5,D1ii7
        TIME(reductions_UU_V2_GAMMAF1D_U[i]->V2( stochastic_propagator,     glist_sink_nucleon, propDN, propUP, false));

        //B4,B6
        //D1ii2,D1ii4,D1ii6,D1ii8
        TIME(reductions_UU_V4_GAMMAF1D_U[i]->V4( stochastic_propagator,     glist_sink_nucleon, propDN, propUP, false));

        //B10,B12
        TIME(reductions_UU_V4_GAMMAF1U_D[i]->V4( stochastic_propagator,     glist_sink_nucleon, propUP, propDN, false));
 
        //W5,W6,W7,W8
        //D1ii1,D1ii2,D1ii3,D1ii4
        //W9,W10,W11,W12
        //D1ii5,D1ii6,D1ii7,D1ii8
        TIME(reductions_UU_V3_GAMMAF2U_zero_mom[i]->V3( stochastic_source, glist_sink_meson,   propUP, true));

        //For D(xf1,xf2)

        stochastic_propagator.apply_gamma5(); 

        //W25,W26,W27,W28
        //W29,W30,W31,W32
        //D1ii13,D1ii14
        //D1ii15,D1ii16
        //W33,W34,W35,W36
        //D1ii17,D1ii18
        //D1ii19,D1ii20
        TIME(reductions_DD_V3_GAMMAF2U_zero_mom[i]->V3( stochastic_propagator, glist_sink_meson,   propUP, true));

 

        //W13,W14,W15,16
        //W21,W22,W23,W24
        //D1ii9,D1ii10
        //D1ii11,D1ii12
        TIME(reductions_DD_V3_GAMMAF2D_zero_mom[i]->V3( stochastic_propagator, glist_sink_meson,   propDN, true));

        //W1,W2,W3,W4
        TIME(reductions_DD_V3_GAMMAF2U[i]->V3( stochastic_propagator, glist_sink_meson,   propUP, true));


        stochastic_source.apply_gamma5();
        //D1ii9,D1ii10
        //D1ii11,D1ii12
        //B7,B8
        //B1,B2
        TIME(reductions_DD_V2_GAMMAF1U_U[i]->V2( stochastic_source, glist_sink_nucleon, propUP, propUP, false));

        //B13,B14,B17,B18
        TIME(reductions_DD_V4_GAMMAF1U_D[i]->V4( stochastic_source, glist_sink_nucleon, propUP, propDN, false));

        //B15,B16,B19,B20
        TIME(reductions_DD_V2_GAMMAF1D_U[i]->V2( stochastic_source, glist_sink_nucleon, propUP, propUP, false));

        //D1ii15,D1ii19,B13,B19,B20
        TIME(reductions_DD_V2_GAMMAF1U_D[i]->V2( stochastic_source, glist_sink_nucleon, propUP, propDN, false));


      }

      //We first construct the zero-momentum oet propagators
      //In the first step we draw a new random vector (different 
      //one for every source position)
      vectorStoc_source_oet.stochastic_Z(nroots);
      
      //Store zero momentum oet propagators
      {

         //Doing for +mu for the UP propagator spin dilution oet
         if(mu<0) {
           mu*=-1.;
           solver.UpdateSolver();
         }

         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;          
         PLEGMA_Vector<double> vectorSave_diluted;
 
         vectortmp1.absorbTimeslice(vectorStoc_source_oet, sequential_time_source);

         {  // Smearing the source
            
            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.absorb(vectortmp1, sourcePositions[isource][DIM_T]);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectortmp1.absorb(vector2,sourcePositions[isource][DIM_T]);
         }

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorStoc_source_oet.copy(vectortmp1);
  
         //Transforming to physical base for the UP quark
         vectortmp2.rotateToPhysicalBasis(vectorStoc_source_oet,+1);
 
         //Dilution     
         vectortmp1.dilutespin(vectortmp2,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorSave_diluted.copy(vectortmp1);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.copy(vectorSave_diluted);
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
           vectortmp2.unload();
           stochastic_oet_prop_u_zero_mom[spinindex]->copy(vectortmp2,HOST);
           vectortmp2.load();

           //stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+confnumber+"propagator_"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){
             vectortmp1.dilutespindisplace(vectorSave_diluted,spinindex+1,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           }
         }
         
         //Doing for -mu for the DN propagator spin dilution oet
         if(mu>0) {
           mu*=-1.;
           solver.UpdateSolver();
         }

         vectortmp2.rotateToPhysicalBasis(vectorStoc_source_oet,-1);

         //Dilution     
         vectortmp1.dilutespin(vectortmp2,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorSave_diluted.copy(vectortmp1);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.copy(vectorSave_diluted);
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
           vectortmp2.unload();
           stochastic_oet_prop_d_zero_mom[spinindex]->copy(vectortmp2,HOST);
           vectortmp2.load();
           if (spinindex<3){
             vectortmp1.dilutespindisplace(vectorSave_diluted,spinindex+1,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           }
         }
      }

      //T diagram piN sink
      //Details eq 51-56 in Marcus's notes
      {

        PLEGMA_printf("Start calculating T piN sink\n");
        //extract all the momenta that corresponds to pi2==(0,0,0)
	momList list_pf1pf2comb = sourcemomentumList.extract({0,0,0}, 0);
        //for the I=1/2 case we compute only at zero pion momentum
        momList list_pf        = list_pf1pf2comb.extract({0,0,0}, 2);

        //udu- dbaru - ubarubarubar: N+pi+ <- Delta++
        PLEGMA_ScattCorrelator<float> corrT_piNsink_1(sourcePositions[isource], list_pf1pf2comb);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_3(sourcePositions[isource], list_pf1pf2comb);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_5(sourcePositions[isource], list_pf1pf2comb);

        //udu- ubaru - dbarubarubar: N_+pi0 <- Delta_1/2 1
        PLEGMA_ScattCorrelator<float> corrT_piNsink_7(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_8(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_9(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_10(sourcePositions[isource], list_pf);

        //udu- dbard - dbarubarubar: N_+pi0 <- Delta_1/2 1
        PLEGMA_ScattCorrelator<float> corrT_piNsink_11(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_12(sourcePositions[isource], list_pf);

        //udu- ubaru - dbarubarubar: N_+pi0 <- Delta_1/2 2
        PLEGMA_ScattCorrelator<float> corrT_piNsink_13(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_15(sourcePositions[isource], list_pf);
        //udu- dbard - dbarubarubar: N_+pi0 <- Delta_1/2 2
        PLEGMA_ScattCorrelator<float> corrT_piNsink_17(sourcePositions[isource], list_pf);

        //dud- dbaru - dbarubarubar: N_0pi+ <- Delta_1/2 1
        PLEGMA_ScattCorrelator<float> corrT_piNsink_19(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_20(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_21(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_22(sourcePositions[isource], list_pf);

        //dud- dbaru - dbarubarubar: N_0pi+ <- Delta_1/2 2
        PLEGMA_ScattCorrelator<float> corrT_piNsink_23(sourcePositions[isource], list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_25(sourcePositions[isource], list_pf);



        corrT_piNsink_1.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "32", "T1"); 
        corrT_piNsink_3.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "32", "T3");
        corrT_piNsink_5.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "32", "T5");
        corrT_piNsink_7.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "12", "T7");
        corrT_piNsink_8.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "12", "T8");
        corrT_piNsink_9.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "12", "T9");
        corrT_piNsink_10.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T10");
        corrT_piNsink_11.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T11");
        corrT_piNsink_12.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T12");
        corrT_piNsink_13.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T13");
        corrT_piNsink_15.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T15");
        corrT_piNsink_17.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T17");
        corrT_piNsink_19.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T19");
        corrT_piNsink_20.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T20");
        corrT_piNsink_21.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T21");
        corrT_piNsink_22.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T22");
        corrT_piNsink_23.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T23");
        corrT_piNsink_25.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T25");


        PLEGMA_printf("Initialization done\n");

        //we have already all the factors computed
  
        for (int i=0; i<n_stochastic_samples; ++i){

          TIME(corrT_piNsink_1.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U[i], *reductions_DD_V2_GAMMAF1U_U[i], 1, true));
          TIME(corrT_piNsink_3.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U[i], *reductions_DD_V2_GAMMAF1U_U[i], 3, true));
          TIME(corrT_piNsink_5.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U[i], *reductions_DD_V2_GAMMAF1U_U[i], 5, true));
          TIME(corrT_piNsink_7.T_diagramms_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1D_U[i], 7, true));
          TIME(corrT_piNsink_8.T_diagramms_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1D_U[i], 8, true));
          TIME(corrT_piNsink_9.T_diagramms_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], 9, true));
          TIME(corrT_piNsink_10.T_diagramms_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], 10, true));
          TIME(corrT_piNsink_11.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1D_U[i], 11, true));
          TIME(corrT_piNsink_12.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1D_U[i], 12, true));
          TIME(corrT_piNsink_13.T_diagramms_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1D_U[i], 13, true));
          TIME(corrT_piNsink_15.T_diagramms_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], 15, true));
          TIME(corrT_piNsink_17.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i], 17, true));
          TIME(corrT_piNsink_19.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i], 19, true));
          TIME(corrT_piNsink_20.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i], 20, true));
          TIME(corrT_piNsink_21.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i], 21, true));
          TIME(corrT_piNsink_22.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i], 22, true));
          TIME(corrT_piNsink_23.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i], 23, true));
          TIME(corrT_piNsink_25.T_diagramms_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i], 25, true));


        }                
        outfilename=outdiagramPrefix+confnumber+sourcepositiontext+"_TpiNsink";

        produceOutput(corrT_piNsink_1, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_3, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_5, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_7, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_8, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_9, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_10, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_11, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_12, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_13, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_15, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_17, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_19, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_20, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_21, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_22, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_23, outfilename, "T1", n_stochastic_samples);
        produceOutput(corrT_piNsink_25, outfilename, "T1", n_stochastic_samples);

      }

      //We need the pi-N 4pt functions only at zero pion momentum and non-zero nucleon momentum
      //therefore we filter further the momentumlist corresponding to pi2==0 to also pf2==0
      momList filtered_sourcemomentumList_pi20pf20 = filtered_sourcemomentumList_pi20.extract(filter,2);
      std::vector<std::string> stringarray=filtered_sourcemomentumList_pi20pf20.to_string( {0,1,2}, {"pi2=","pf1=","pf2="} );
      for (int i=0; i<stringarray.size(); ++i){
        PLEGMA_printf("%s\n", stringarray[i].c_str());
      }

      
      //Loop diagram at the source
      {

        PLEGMA_ScattCorrelator<float> corrD1ii1(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii2(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii3(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii4(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrD1ii9(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii10(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrD1ii13(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii14(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii15(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ii16(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);


        TIME(corrD1ii1.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii1"));
        TIME(corrD1ii2.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii2"));
        TIME(corrD1ii3.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii3"));
        TIME(corrD1ii4.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii4"));

        TIME(corrD1ii9.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii9"));
        TIME(corrD1ii10.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii10"));

        TIME(corrD1ii13.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii13"));
        TIME(corrD1ii14.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii14"));
        TIME(corrD1ii15.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii15"));
        TIME(corrD1ii16.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii16"));


        float *Loop_UPDN_source=(float *)malloc(2*sizeof(float)*glist_source_meson.size());


        for (int j=0; j< 2*glist_source_meson.size(); ++j){
          Loop_UPDN_source[j]=0;
        }

        PLEGMA_ScattCorrelator<float> Loop_UP_temporary(source_stoch, piN12_zeropion); 

        Loop_UP_temporary.initialize_diagram( glist_sink_meson, "L"); 

        for (int i=0; i<n_stochastic_samples; ++i){


          TIME(Loop_UP_temporary.Loop_diagramms( stochastic_sources[i], stochastic_propags[i], 0, false));

          std::shared_ptr<float> Loop_UP_sp=Loop_UP_temporary.get_source_time_slice();

          for (int j=0; j< glist_source_meson.size(); ++j){
            Loop_UPDN_source[2*j+0]+=2*Loop_UP_sp.get()[2*j+0];
            Loop_UPDN_source[2*j+1]=0.;
          }

          for (int j=0; j< 2*glist_source_meson.size(); ++j){
            Loop_UPDN_source[j]/=n_stochastic_samples;
          }

        }

        for (int i=0; i<n_stochastic_samples; ++i){

          TIME(corrD1ii1.D1ii_diagramms(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], Loop_UPDN_source, 0, 1, true));          
          TIME(corrD1ii2.D1ii_diagramms(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1D_U[i], Loop_UPDN_source, 0, 2, true));          
          TIME(corrD1ii3.D1ii_diagramms(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], Loop_UPDN_source, 0, 3, true));
          TIME(corrD1ii4.D1ii_diagramms(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1D_U[i], Loop_UPDN_source, 0, 4, true)); 

          TIME(corrD1ii9.D1ii_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i],Loop_UPDN_source, 0, 9, true));
          TIME(corrD1ii10.D1ii_diagramms(*reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i],Loop_UPDN_source, 0, 10, true));

          TIME(corrD1ii13.D1ii_diagramms(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i],Loop_UPDN_source, 0, 13, true));          
          TIME(corrD1ii14.D1ii_diagramms(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i],Loop_UPDN_source, 0, 14, true));
          TIME(corrD1ii15.D1ii_diagramms(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i],Loop_UPDN_source, 0, 15, true));
          TIME(corrD1ii16.D1ii_diagramms(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i],Loop_UPDN_source, 0, 16, true));

        } //stochastic samples

        free(Loop_UPDN_source);

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ii";

        produceOutput(corrD1ii1, outfilename, "D1ii", n_stochastic_samples);
        produceOutput(corrD1ii2, outfilename, "D1ii", n_stochastic_samples);
        produceOutput(corrD1ii3, outfilename, "D1ii", n_stochastic_samples);
        produceOutput(corrD1ii4, outfilename, "D1ii", n_stochastic_samples);

        produceOutput(corrD1ii9, outfilename, "D1ii", n_stochastic_samples);
        produceOutput(corrD1ii10, outfilename,"D1ii", n_stochastic_samples);

        produceOutput(corrD1ii13, outfilename,"D1ii", n_stochastic_samples);
        produceOutput(corrD1ii14, outfilename,"D1ii", n_stochastic_samples);
        produceOutput(corrD1ii15, outfilename,"D1ii", n_stochastic_samples);
        produceOutput(corrD1ii16, outfilename,"D1ii", n_stochastic_samples);

      } //diagrams containing loop at the source


/**********************************************************************************************
*
*
*      In the third section we compute sequential propagators (uu,dd, ud)
*      and all the diagrams needed for I=1/2 I_z=1/2 and I=3/2 I_z=3/2
*      For the I=1/2 case we only consider zero momentum for the pion, 
*      this means the sequential momentum and the momentum in V3 reduction is 
*      set to zero, however we have all the possible momenta for pf1
*      In the I=3/2 case we consider all the momentum for the pion as well
*
*
***********************************************************************************************/
      //uu case
      {

        std::vector<int> momentum_i2 = {0,0,0};
        std::vector<std::vector<int>> mpi2_filt;
        std::vector<std::vector<int>> mptot_filt = sourcemomentumList.uniq_p(3);
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);
        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

        PLEGMA_ScattCorrelator<float> corrT15(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT17(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT21(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT22(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT23(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT24(sourcePositions[isource], list_mpi2ptot);

        corrT15.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq15");
        corrT17.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq17");
        corrT21.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq21");
        corrT22.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq22");
        corrT23.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq23");
        corrT24.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq24");



        PLEGMA_ScattCorrelator<float> reductionsV3(source, piN12_zeropion);
        PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList_pi20.uniq_p(1));

        PLEGMA_ScattCorrelator<float> corrB3(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB4(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB5(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB6(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrB17(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB18(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB19(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB20(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrW5(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW6(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW7(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW8(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrW13(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW14(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW15(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW16(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrW29(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW30(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW31(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW32(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrD1ff1389(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrD1ff24710(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);


        TIME(corrD1ff24710.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "D1ff2-4-7-10"));
        TIME(corrD1ff1389.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ff1-3-8-9"));


        corrB3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B3");
        corrB4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B4");
        corrB5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B5");
        corrB6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B6");

        corrB17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B17");
        corrB18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B18");
        corrB19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B19");
        corrB20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B20");

        corrW5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W5");
        corrW6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W6");
        corrW7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W7");
        corrW8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W8");


        corrW13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W13");
        corrW14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W14");
        corrW15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W15");
        corrW16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W16");

        corrW29.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W29");
        corrW30.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W30");
        corrW31.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W31");
        corrW32.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W32");

        PLEGMA_Propagator3D<float> propTS3D;
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorAuxD;
          PLEGMA_Vector<float> vectorAuxF;
          PLEGMA_Vector<double> vectorAuxD2;
          //Performing the smearing
          // Smearing the source
          {
            PLEGMA_Vector3D<double> vector1, vector2;
            vectorAuxF.absorb(propUP, isc/3, isc%3);
            vectorAuxD.copy(vectorAuxF);
            vector1.absorb( vectorAuxD, sequential_time_source );
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectorAuxD2.absorb(vector2,sourcePositions[isource][DIM_T]);
          }
          //Perform multiplication with gamma_i2
          vectorAuxD2.apply_gamma_scatt(G_5);
          //Perform rotation to the physical basis
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1));
          vectorAuxF.copy(vectorAuxD);
          propTS3D.absorb(vectorAuxF, sequential_time_source, isc/3, isc%3);
        }
        //Computing sequential propagators UU T_fii with insertion
        //gamma_i2=gamma_5 and momentum (0,0,0)
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorInOut;
          PLEGMA_Vector<float> vectorAuxF;
          PLEGMA_Vector<double> vectorAuxD;
          vectorAuxF.absorb(propTS3D, sequential_time_source, isc/3, isc%3);
          vectorInOut.copy(vectorAuxF);
          PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
          //performing the inversion
          TIME(solver.solve(vectorInOut, vectorInOut));
          //performing rotation to physical base
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));
          //performing smearing
          TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));
          vectorAuxF.copy(vectorInOut);
	  propTS.absorb(vectorAuxF, isc/3, isc%3);
        }

        for (int i=0; i<n_stochastic_samples; ++i){
          //Computing diagrams containing loops first


 
          PLEGMA_Vector<float> stochastic_propagator;
          PLEGMA_Vector<float> stochastic_source;
          stochastic_propagator.copy(*stochastic_propags[i],HOST);
          stochastic_source.copy(*stochastic_sources[i],HOST);
          stochastic_propagator.load();
          stochastic_source.load();

          


          //For computing the B diagrams we compute the V3 factor
          //using the sequential propagator for U(xf1,xf2) type
          TIME(reductionsV3.V3( stochastic_source, glist_sink_meson,   propTS, true));

          TIME(corrB3.B_diagramms(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i], 0, 3, true));
          TIME(corrB4.B_diagramms(reductionsV3, *reductions_UU_V4_GAMMAF1D_U[i], 0, 4, true));
          TIME(corrB5.B_diagramms(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i], 0, 5, true));
          TIME(corrB6.B_diagramms(reductionsV3, *reductions_UU_V4_GAMMAF1D_U[i], 0, 6, true));

          
          TIME(reductionsV2.V4( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false));

          TIME(corrW5.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 5, true));
          TIME(corrW7.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 7, true));

          TIME(reductionsV2.V2( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false));

          TIME(corrW6.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 6, true));
          TIME(corrW8.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 8, true));


          //for D(xf1,xf2) type
          stochastic_propagator.apply_gamma5();
          TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propTS, true));

          TIME(corrB17.B_diagramms(reductionsV3, *reductions_DD_V4_GAMMAF1U_D[i], 0, 17, true));
          TIME(corrB18.B_diagramms(reductionsV3, *reductions_DD_V4_GAMMAF1U_D[i], 0, 18, true));
          TIME(corrB19.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_D[i], 0, 19, true));
          TIME(corrB20.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_D[i], 0, 20, true));

          //For computing the W diagrams we compute the V2 factor using the sequential
          stochastic_source.apply_gamma5();
          TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propTS, propUP, false));

          TIME(corrW13.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 13, true));
          TIME(corrW15.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 15, true));

          TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false));

          TIME(corrW14.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 14, true));
          TIME(corrW16.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 16, true));

          TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propTS, propDN, false));

          TIME(corrW31.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 31, true));
          TIME(corrW32.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 32, true));

          TIME(reductionsV2.V4( stochastic_source, glist_sink_nucleon, propTS, propDN, false));

          TIME(corrW29.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 29, true));
          TIME(corrW30.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 30, true));

        }//loop over stochastic samples


        //D1ff type diagrams
        PLEGMA_ScattCorrelator<float> reductionsT1(source,  list_mpi2ptot.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsT2(source,  list_mpi2ptot.uniq_p(1));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP));
        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP));

        TIME(corrD1ff24710.LT_diagramms( reductionsT1, reductionsT2, Loop_UPDN ));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS));
        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS));

        TIME(corrD1ff1389.LT_diagramms( reductionsT1, reductionsT2, Loop_UPDN ));

        //Triangle diagrams
        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propDN, propTS, propUP));
        TIME(corrT15.convertTreductiontoDiagram( reductionsT1 ));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propDN, propUP, propTS));
        TIME( corrT17.convertTreductiontoDiagram( reductionsT1 ));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propTS, propDN, propUP));
        TIME( corrT21.convertTreductiontoDiagram( reductionsT1 ));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propUP, propDN, propTS));
        TIME( corrT23.convertTreductiontoDiagram( reductionsT1 ));

        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_delta, propTS, propDN, propUP));
        TIME( corrT22.convertTreductiontoDiagram( reductionsT2 ));

        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_delta, propUP, propTS, propDN));
        TIME( corrT24.convertTreductiontoDiagram( reductionsT2 ));

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
        
        produceOutput(corrT15, outfilename, "T");
        produceOutput(corrT17, outfilename, "T");
        produceOutput(corrT21, outfilename, "T");
        produceOutput(corrT22, outfilename, "T");
        produceOutput(corrT23, outfilename, "T");
        produceOutput(corrT24, outfilename, "T");
        
        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";

        produceOutput(corrB3, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB4, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB5, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB6, outfilename, "B", n_stochastic_samples);
 
        produceOutput(corrB17, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB18, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB19, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB20, outfilename, "B", n_stochastic_samples);

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";

        produceOutput(corrW5, outfilename, "W", n_stochastic_samples);
        produceOutput(corrW6, outfilename, "W", n_stochastic_samples);
        produceOutput(corrW7, outfilename, "W", n_stochastic_samples);
        produceOutput(corrW8, outfilename, "W", n_stochastic_samples);
        produceOutput(corrW13, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW14, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW15, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW16, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW29, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW30, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW31, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW32, outfilename,"W", n_stochastic_samples);

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ff";
        produceOutput(corrD1ff1389, outfilename, "D1ff");
        produceOutput(corrD1ff24710, outfilename, "D1ff");

      }//end of loop for sequential UU

      //start for DD
      {
        std::vector<int> momentum_i2={0,0,0};
        std::vector<std::vector<int>> mpi2_filt;
        std::vector<std::vector<int>> mptot_filt = sourcemomentumList.uniq_p(3);
        mpi2_filt.assign(mptot_filt.size(),momentum_i2);
        momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

        PLEGMA_ScattCorrelator<float> corrT19(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT25(sourcePositions[isource], list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT26(sourcePositions[isource], list_mpi2ptot);

        PLEGMA_ScattCorrelator<float> reductionsV3(source, piN12_zeropion);
        PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList_pi20.uniq_p(1));

        PLEGMA_ScattCorrelator<float> corrB7(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrB8(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrW9(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW10(sourcePositions[isource],filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW11(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW12(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrW33(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW34(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW35(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
        PLEGMA_ScattCorrelator<float> corrW36(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

        PLEGMA_ScattCorrelator<float> corrD1ff561112(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);


        corrT19.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq19");
        corrT25.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq25");
        corrT26.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq26");



        TIME(corrD1ff561112.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ff5-6-11-12"));


        corrB7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B7");
        corrB8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B8");

        corrW9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W9");
        corrW10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W10");
        corrW11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W11");
        corrW12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W12");

        corrW33.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W33");
        corrW34.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W34");
        corrW35.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W35");
        corrW36.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W36");

        PLEGMA_Propagator3D<float> propTS3D;
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorAuxD;
          PLEGMA_Vector<float> vectorAuxF;
          PLEGMA_Vector<double> vectorAuxD2;
          //Performing the smearing
          // Smearing the source
          {
            PLEGMA_Vector3D<double> vector1, vector2;
            vectorAuxF.absorb(propDN, isc/3, isc%3);
            vectorAuxD.copy(vectorAuxF);
            vector1.absorb( vectorAuxD, sequential_time_source );
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectorAuxD2.absorb(vector2,sourcePositions[isource][DIM_T]);
          }
          //Perform multiplication with gamma_i2
          vectorAuxD2.apply_gamma_scatt(G_5);
          //Perform rotation to the physical basis
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,-1));
          vectorAuxF.copy(vectorAuxD);
          propTS3D.absorb(vectorAuxF, sequential_time_source, isc/3, isc%3);
        }
        //Computing sequential propagators DD T_fii with insertion
        //gamma_i2=gamma_5 and momentum (0,0,0)
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorInOut;
          PLEGMA_Vector<float> vectorAuxF;
          PLEGMA_Vector<double> vectorAuxD;
          vectorAuxF.absorb(propTS3D, sequential_time_source, isc/3, isc%3);
          vectorInOut.copy(vectorAuxF);
          PLEGMA_printf("Going to invert DN for sequential propagator DN  for component %d\n", isc);
          //performing the inversion
          TIME(solver.solve(vectorInOut, vectorInOut));
          //performing rotation to physical base
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1));
          //performing smearing
          TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));
          vectorAuxF.copy(vectorInOut);
          propTS.absorb(vectorAuxF, isc/3, isc%3);
        }

        PLEGMA_ScattCorrelator<float> reductionsT1(source,  list_mpi2ptot.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsT2(source,  list_mpi2ptot.uniq_p(1));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propUP));
        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propUP));

        TIME(corrD1ff561112.LT_diagramms( reductionsT1, reductionsT2, Loop_UPDN ));

         //Triangle diagrams
        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propTS, propUP, propUP));
        TIME( corrT19.convertTreductiontoDiagram( reductionsT1 ));

        TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propUP, propTS, propUP));
        TIME( corrT25.convertTreductiontoDiagram( reductionsT1 ));

        TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_delta, propUP, propTS, propUP));
        TIME( corrT26.convertTreductiontoDiagram( reductionsT2 ));



        for (int i=0; i<n_stochastic_samples; ++i){
          PLEGMA_Vector<float> stochastic_propagator;
          PLEGMA_Vector<float> stochastic_source;
          stochastic_propagator.copy(*stochastic_propags[i],HOST);
          stochastic_source.copy(*stochastic_sources[i],HOST);
          stochastic_propagator.load();
          stochastic_source.load();

          TIME(reductionsV2.V2( stochastic_propagator, glist_sink_nucleon, propTS, propUP, false));

          TIME( corrW9.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0,  9, true));
          TIME(corrW11.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 11, true));

          TIME(reductionsV2.V4( stochastic_propagator, glist_sink_nucleon, propTS, propUP, false));

          TIME(corrW10.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 10, true));
          TIME(corrW12.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 12, true));


          //For computing the B diagrams we compute the V3 factor
          //using the sequential propagator
          stochastic_propagator.apply_gamma5();
          TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propTS, true));

          TIME(corrB7.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i], 0, 7, true));
          TIME(corrB8.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i], 0, 8, true));

          //For computing the W diagrams we compute the V2 factor using the sequential
          stochastic_source.apply_gamma5();
          TIME(reductionsV2.V4( stochastic_source, glist_sink_nucleon, propUP, propTS, false));
          
          TIME(corrW33.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 33, true));
          TIME(corrW34.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 34, true));

          TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false));
 
          TIME(corrW35.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 35, true));
          TIME(corrW36.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 36, true));

        }//loop over stochastic samples

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

        produceOutput(corrT19, outfilename, "T");
        produceOutput(corrT25, outfilename, "T");
        produceOutput(corrT26, outfilename, "T");

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
        produceOutput(corrB7, outfilename, "B", n_stochastic_samples);
        produceOutput(corrB8, outfilename, "B", n_stochastic_samples);

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
        produceOutput(corrW33, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW34, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW35, outfilename,"W", n_stochastic_samples);
        produceOutput(corrW36, outfilename,"W", n_stochastic_samples);

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ff";
        produceOutput(corrD1ff561112, outfilename, "D1ff");


      }//end for DD

      ///We first have a loop over all unique the source meson momentum p_i2 
      for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	auto &momentum_i2 =  mpi2[i_mpi2];
	//List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);

        std::string pi2x=std::to_string(momentum_i2[0]);
        std::string pi2y=std::to_string(momentum_i2[1]);
        std::string pi2z=std::to_string(momentum_i2[2]);
 
	// 4pt diagrams
	
	PLEGMA_ScattCorrelator<float> corrB1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrB2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW3(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW4(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ3(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ4(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList);
	
	//T diagrams
	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
	std::vector<std::vector<int>> mpi2_filt;
	mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});
	PLEGMA_ScattCorrelator<float> corrT(sourcePositions[isource], list_mpi2ptot);
	corrT.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "Tseq");	

	//initialize diagrams
	
	corrB1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B1");
	corrB2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B2");
	corrW1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W1");
	corrW2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W2");
	corrW3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W3");
	corrW4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W4");
	corrZ1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z1");
	corrZ2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z2");
	corrZ3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z3");
	corrZ4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z4");
	corrM.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "MNPPP");
	
	
        PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source, filtered_sourcemomentumList.uniq_p(2));

        //Loop over the different gamma structure for the source meson
        for (int i_gamma_i2=0; i_gamma_i2<glist_source_meson.size(); ++i_gamma_i2) {
	  GAMMAS_SCATT gamma_i2 = glist_source_meson[i_gamma_i2];
          // Computing sequential propagators f1 <- i_2 <- i_1 
          // so the sequential source source time is fixed
          // and the momentum is also fixed to be momentum_i2

          //smearing the 3D propagators
          
          PLEGMA_Propagator3D<float> propDN3D;      
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            // Performing the smearing
            // Smearing the source
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source );
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorAuxD2.absorb(vector2,sourcePositions[isource][DIM_T]);
            }

            //Perform multiplication with gamma_i2
            vectorAuxD2.apply_gamma_scatt(gamma_i2);

            //Perform rotation to the physical basis
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1));

            
            vectorAuxF.copy(vectorAuxD);
            propDN3D.absorb(vectorAuxF, sequential_time_source, isc/3, isc%3);
          }
 
          propDN3D.mulMomentumPhases(momentum_i2,1);

          //Computing sequential propagators UD T_fii with insertion
          //gamma_i2 and momentum SinkMom
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorInOut;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD;
          
            vectorAuxF.absorb(propDN3D, sequential_time_source, isc/3, isc%3);
            vectorInOut.copy(vectorAuxF);
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorInOut, vectorInOut));

            //performing rotation to physical base
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));

            //performing smearing
            TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));

            vectorAuxF.copy(vectorInOut);
            propTS.absorb(vectorAuxF, isc/3, isc%3);
          }

          /*
          if(outfile_SEQ!="")
          {
             PLEGMA_printf("Save sequential propagator for the ud \n");
             PLEGMA_Vector<float> vectorAuxPrint(BOTH);
             for(int isc = 0 ; isc < 12 ; isc++){
               std::string spin=std::to_string(isc/3);
               std::string col=std::to_string(isc%3);
               vectorAuxPrint.absorb(propUPDN,isc/3,isc%3);
               vectorAuxPrint.unload();
               vectorAuxPrint.writeLIME(outfile_SEQ+confnumber+sourcepositiontext+"_pi2"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+spin+"_c"+col);
               //vectorAuxPrint.writeHDF5(outfile_SEQ+"_s"+spin+"_c"+col);
             }
          }*/

          //case sequential UD for spin1/2
 
          if ((momentum_i2[0] == 0) && (momentum_i2[1] == 0) && (momentum_i2[2] == 0)) {
            std::vector<std::vector<int>> mpi2_filt;
            std::vector<std::vector<int>> mtot_filt = sourcemomentumList.uniq_p(3);
            mpi2_filt.assign(mptot_filt.size(),momentum_i2);
            momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});
    
            PLEGMA_ScattCorrelator<float> corrT7(sourcePositions[isource], list_mpi2ptot);
            PLEGMA_ScattCorrelator<float> corrT9(sourcePositions[isource], list_mpi2ptot);
            PLEGMA_ScattCorrelator<float> corrT11(sourcePositions[isource],list_mpi2ptot);
            PLEGMA_ScattCorrelator<float> corrT12(sourcePositions[isource],list_mpi2ptot);
            PLEGMA_ScattCorrelator<float> corrT13(sourcePositions[isource],list_mpi2ptot);
            PLEGMA_ScattCorrelator<float> corrT14(sourcePositions[isource],list_mpi2ptot);

            PLEGMA_ScattCorrelator<float> reductionsV3(source, piN12_zeropion);
            PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList_pi20pf20.uniq_p(1));

            PLEGMA_ScattCorrelator<float> corrB9(sourcePositions[isource],  filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrB10(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrB11(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrB12(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

            PLEGMA_ScattCorrelator<float> corrB13(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrB14(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrB15(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrB16(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

            PLEGMA_ScattCorrelator<float> corrW17(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW18(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW19(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW20(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

            PLEGMA_ScattCorrelator<float> corrW21(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW22(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW23(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW24(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

            PLEGMA_ScattCorrelator<float> corrW25(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW26(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW27(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrW28(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

            PLEGMA_ScattCorrelator<float> corrD1ff13141718(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
            PLEGMA_ScattCorrelator<float> corrD1ff15161920(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

            corrT7.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq7");
            corrT9.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq9");
            corrT11.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta,"12", "Tseq11");
            corrT12.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta,"12", "Tseq12");
            corrT13.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta,"12", "Tseq13");
            corrT14.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta,"12", "Tseq14");

            corrB9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B9");
            corrB10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B10");
            corrB11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B11");
            corrB12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B12");

            corrB13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B13");
            corrB14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B14");
            corrB15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B15");
            corrB16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B16");

            corrW17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W17");
            corrW18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W18");
            corrW19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W19");
            corrW20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W20");

            corrW21.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W21");
            corrW22.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W22");
            corrW23.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W23");
            corrW24.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W24");

            corrW25.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W25");
            corrW26.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W26");
            corrW27.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W27");
            corrW28.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W28");

            TIME(corrD1ff13141718.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "D1ff13-14-17-18"));
            TIME(corrD1ff15161920.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "D1ff15-16-19-20"));


            std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList_pi20.uniq_p(3);

            PLEGMA_ScattCorrelator<float> reductionsT1(source,  mptot_filt);
            PLEGMA_ScattCorrelator<float> reductionsT2(source,  mptot_filt);

            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propDN));
            TIME(reductionsT2.T1(glist_source_nucleon, glist_sink_nucleon, propTS, propUP, propDN));

            TIME(corrD1ff13141718.LT_diagramms( reductionsT1, reductionsT2, Loop_UPDN ));

            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS));
            TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP));

            TIME(corrD1ff15161920.LT_diagramms( reductionsT1, reductionsT2, Loop_UPDN ));

            TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_delta, propDN, propUP, propTS));
            TIME( corrT7.convertTreductiontoDiagram( reductionsT1 ));

            TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_delta, propDN, propUP, propTS));
            TIME( corrT9.convertTreductiontoDiagram( reductionsT2 ));

            TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_delta, propTS, propUP, propDN));
            TIME( corrT11.convertTreductiontoDiagram( reductionsT1 ));
            
            TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_delta, propTS, propDN, propUP));
            TIME( corrT12.convertTreductiontoDiagram( reductionsT2 ));

            TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_delta, propUP, propTS, propDN));
            TIME( corrT13.convertTreductiontoDiagram( reductionsT1 ));

            TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_delta, propUP, propDN, propTS));
            TIME( corrT14.convertTreductiontoDiagram( reductionsT1 ));


            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ff";

            produceOutput(corrD1ff13141718, outfilename, "D1ff");
            produceOutput(corrD1ff15161920, outfilename, "D1ff");

            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

            produceOutput(corrT7, outfilename,  "T");
            produceOutput(corrT9, outfilename,  "T");
            produceOutput(corrT11, outfilename, "T");
            produceOutput(corrT12, outfilename, "T");
            produceOutput(corrT13, outfilename, "T");
            produceOutput(corrT14, outfilename, "T");

            for (int i=0; i<n_stochastic_samples; ++i){
              PLEGMA_Vector<float> stochastic_propagator;
              PLEGMA_Vector<float> stochastic_source;
              stochastic_propagator.copy(*stochastic_propags[i],HOST);
              stochastic_source.copy(*stochastic_sources[i],HOST);
              stochastic_propagator.load();
              stochastic_source.load();

              TIME(reductionsV3.V3( stochastic_source, glist_sink_meson,   propTS, true ));

              TIME( corrB9.B_diagramms(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i], 0,  9, true));
              TIME(corrB10.B_diagramms(reductionsV3, *reductions_UU_V4_GAMMAF1D_U[i], 0, 10, true));
              TIME(corrB11.B_diagramms(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i], 0, 11, true));
              TIME(corrB12.B_diagramms(reductionsV3, *reductions_UU_V4_GAMMAF1D_U[i], 0, 12, true));


              TIME(reductionsV2.V4( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false));

              TIME(corrW17.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 17, true));
              TIME(corrW19.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 19, true));

              TIME(reductionsV2.V2( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false));

              TIME(corrW18.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 18, true));
              TIME(corrW20.W_diagramms( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 20, true));

              stochastic_propagator.apply_gamma5();
              TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propTS, true));

              TIME(corrB13.B_diagramms(reductionsV3, *reductions_DD_V4_GAMMAF1U_D[i], 0,13, true));
              TIME(corrB14.B_diagramms(reductionsV3, *reductions_DD_V4_GAMMAF1U_D[i], 0,14, true));
              TIME(corrB15.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_D[i], 0,15, true));
              TIME(corrB16.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_D[i], 0,16, true));

              stochastic_source.apply_gamma5();
              TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false));

              TIME(corrW22.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 22, true));
              TIME(corrW24.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 24, true));

              TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propTS, propUP, false));

              TIME(corrW21.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 21, true));
              TIME(corrW23.W_diagramms( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, 0, 23, true));

              TIME(reductionsV2.V4( stochastic_source, glist_sink_nucleon, propTS, propDN, false));

              TIME(corrW25.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 25, true));
              TIME(corrW26.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 26, true));

              TIME(reductionsV2.V4( stochastic_source, glist_sink_nucleon, propTS, propDN, false));

              TIME(corrW27.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 27, true));
              TIME(corrW28.W_diagramms( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2, 0, 28, true));

            }

            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";

            produceOutput(corrB9,  outfilename, "B", n_stochastic_samples);
            produceOutput(corrB10, outfilename, "B", n_stochastic_samples);
            produceOutput(corrB11, outfilename, "B", n_stochastic_samples);
            produceOutput(corrB12, outfilename, "B", n_stochastic_samples);

            produceOutput(corrB13, outfilename, "B", n_stochastic_samples);
            produceOutput(corrB14, outfilename, "B", n_stochastic_samples);
            produceOutput(corrB15, outfilename, "B", n_stochastic_samples);
            produceOutput(corrB16, outfilename, "B", n_stochastic_samples);

            outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";

            produceOutput(corrW17, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW18, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW19, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW20, outfilename, "W", n_stochastic_samples);

            produceOutput(corrW21, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW22, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW23, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW24, outfilename, "W", n_stochastic_samples);

            produceOutput(corrW25, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW26, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW27, outfilename, "W", n_stochastic_samples);
            produceOutput(corrW28, outfilename, "W", n_stochastic_samples);
 
 
          }//end of loop momentum pion == (0,0,0)

          //Compute triangle diagramms          
          
          PLEGMA_ScattCorrelator<float> reductionsT1triangle(source,  mptot_filt);
         
          PLEGMA_ScattCorrelator<float> reductionsT3triangle(source,  mptot_filt);

          PLEGMA_ScattCorrelator<float> reductionsT5triangle(source,  mptot_filt);
          TIME(reductionsT1triangle.T1(glist_source_nucleon, glist_sink_delta, propTS, propUP, propUP));
          //reductionsT1triangle.writeHDF5("T1sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
          TIME(reductionsT3triangle.T1(glist_source_nucleon, glist_sink_delta, propUP, propTS, propUP));
          //reductionsT3triangle.writeHDF5("T3sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
          TIME(reductionsT5triangle.T2(glist_source_nucleon, glist_sink_delta, propUP, propUP, propTS));
          //reductionsT5triangle.writeHDF5("T5sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
	  
	  //Compute Diagram T 
          TIME(corrT.T_diagramms(reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, i_gamma_i2));

          for (int i=0; i<n_stochastic_samples; ++i){
            //Compute Diagram B1 and B2
            PLEGMA_Vector<float> stochastic_propagator;
            PLEGMA_Vector<float> stochastic_source;

            stochastic_propagator.copy(*stochastic_propags[i],HOST);
            stochastic_source.copy(*stochastic_sources[i],HOST);

            stochastic_propagator.load();
            stochastic_source.load();

            stochastic_source.apply_gamma5();
            stochastic_propagator.apply_gamma5();
 
	  
            TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propTS, true));

	    TIME(corrB1.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i], i_gamma_i2, 1, true));
	  
	    TIME(corrB2.B_diagramms(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i], i_gamma_i2, 2, true));
          
            //Compute Diagram W1,W2
          
            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false));

            TIME(corrW1.W_diagramms( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 1, true));
	    TIME(corrW2.W_diagramms( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 2, true));
          
            //Compute Diagram W3,W4
          
            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propTS, propUP, false));

            TIME(corrW3.W_diagramms( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 3, true));
	    TIME(corrW4.W_diagramms( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 4, true));

          } //loop over stochastic samples
	  
       } //loop over gamma i2
         
	  
       //Producing spin diluted stochastic propagators for diagram Z1,Z2,Z3,Z4
             
       //We need V3 reductions for all the possible pf2 for diagrams Z1,Z2,Z3,Z4 
       std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted = {
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2))
         };


       //We need V2 reductions for all the possible nucleon momenta pf1
       std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted = {
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1))
       };



       {
          PLEGMA_Vector<double> vectortmp1;
          PLEGMA_Vector<double> vectortmp2;
          PLEGMA_Vector<double> vectorSource_finite_mom;

          //Transforming to physical base for the UP quark
          vectortmp2.rotateToPhysicalBasis(vectorStoc_source_oet,+1);

          //Dilution     
          vectortmp1.dilutespin(vectortmp2,0);         

          //Multiplying by the appropriate momentum phase
          vectorSource_finite_mom.copy(vectortmp1);
          std::vector<int> tmp_4Dmom= momentum_i2 ; 
          tmp_4Dmom.push_back(0);
          vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);
          
          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
       
            for (int spinindex=0; spinindex<4; ++spinindex){
       
              vectortmp1.copy(vectorSource_finite_mom);
              //Doing the inversion
              TIME(solver.solve(vectortmp1, vectortmp1));

              //Rotate back immediately to the physical basis
              TIME(vectortmp2.rotateToPhysicalBasis(vectortmp1,+1));

              //performing smearing
              TIME(vectortmp1.gaussianSmearing(vectortmp2, smearedGauge, nsmearGauss, alphaGauss));

              //Saving the propagator
              vectortmp1.unload();
              stochastic_oet_prop_u_fini_mom[spinindex]->copy(vectortmp1,HOST);
              vectortmp1.load();

              //stochastic_propagator_momp_i2[spinindex].writeLIME(outfile_V+confnumber+"propagator_"+sourcepositiontext+"mompi2_"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(spinindex));
         
              if (spinindex<3){
                vectortmp1.dilutespindisplace(vectorSource_finite_mom,spinindex+1,spinindex);
                vectorSource_finite_mom.copy(vectortmp1);
              }
            }
          }
       }
         
       //Z diagram in case of zero momentum pi2 : we calculate I=1/2 as well
       //Diagram Z1,Z2
       std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
       
       if  ((momentum_i2[0] == 0) && (momentum_i2[1] == 0) && (momentum_i2[2] == 0)){

         //We need V3 reductions only for pf2={0,0,0} for diagrams Z5--Z20
         std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted_zero_pf2 = {
           PLEGMA_ScattCorrelator<float>(source, piN12_zeropion),
           PLEGMA_ScattCorrelator<float>(source, piN12_zeropion),
           PLEGMA_ScattCorrelator<float>(source, piN12_zeropion),
           PLEGMA_ScattCorrelator<float>(source, piN12_zeropion)
         };
         PLEGMA_ScattCorrelator<float> corrZ5(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ6(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ7(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ8(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         PLEGMA_ScattCorrelator<float> corrZ9(sourcePositions[isource],  filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ10(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         PLEGMA_ScattCorrelator<float> corrZ11(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ12(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ13(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ14(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         PLEGMA_ScattCorrelator<float> corrZ15(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ16(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         PLEGMA_ScattCorrelator<float> corrZ17(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ18(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ19(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);
         PLEGMA_ScattCorrelator<float> corrZ20(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         corrZ5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z5");
         corrZ6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z6");
         corrZ7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z7");
         corrZ8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z8");
         corrZ9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z9");
         corrZ10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z10");
         corrZ11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z11");
         corrZ12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z12");
         corrZ13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z13");
         corrZ14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z14");
         corrZ15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z15");
         corrZ16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z16");
         corrZ17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z17");
         corrZ18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z18");
         corrZ19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z19");
         corrZ20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z20");
       
         for (int i=0; i< 4; ++i){
           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);

           st_oet_u_zero.load();

           TIME(reductionsV3_diluted[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propUP, true));

           TIME(reductionsV2_diluted[i].V4( st_oet_u_zero, glist_sink_nucleon, propDN, propUP, false));
         }

         TIME(corrZ1.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 1 ));
         TIME(corrZ2.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 2 ));

         //Diagram Z6,Z8
         for (int i=0; i<4; ++i){
           PLEGMA_Vector<float> st_oet_d_zero;
           st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom[i],HOST);
           st_oet_d_zero.load();
           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_d_zero, gamma_5_t_sinkmeson, propUP, true));
         }

         TIME(corrZ6.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 6 ));
         TIME(corrZ8.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 8 ));

      
         //Diagram Z12,Z14
         for (int i=0; i<4; ++i){
           PLEGMA_Vector<float> st_oet_u_zero;
           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();
           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propDN, true));
         }

         TIME(corrZ12.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 12 ));
         TIME(corrZ14.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 14 ));

         //Diagram Z3,Z4	
         for (int i=0; i< 4; ++i){
           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);

           st_oet_u_zero.load();

           TIME(reductionsV3_diluted[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propUP, true));
  
           TIME(reductionsV2_diluted[i].V2( st_oet_u_zero, glist_sink_nucleon, propDN, propUP, false));

         }

         TIME(corrZ3.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 3 ));
         TIME(corrZ4.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 4 ));

         //Diagram Z5,Z7

         for (int i=0; i< 4; ++i){

           PLEGMA_Vector<float> st_oet_d_zero;
           st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom[i],HOST);
           st_oet_d_zero.load();
           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_d_zero, gamma_5_t_sinkmeson, propUP, true));

         }


         TIME(corrZ5.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 5 ));
         TIME(corrZ7.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 7 ));

       
         //Diagram Z11,Z13
         for (int i=0; i<4; ++i){
           PLEGMA_Vector<float> st_oet_u_zero;
           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();
           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propDN, true));
         } 

         TIME(corrZ12.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 12 ));
         TIME(corrZ14.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 14 ));

         //Diagram Z9,Z10

         for (int i=0; i< 4; ++i){

           PLEGMA_Vector<float> st_oet_d_zero;
           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();

           st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom[i],HOST);
           st_oet_d_zero.load();

           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propDN, true));

           TIME(reductionsV2_diluted[i].V2( st_oet_d_zero, glist_sink_nucleon, propUP, propUP, false));


         }


         TIME(corrZ9.Z_diagramms(  reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 9 ));
         TIME(corrZ10.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 10 ));

         //Diagram Z15,Z16
         for (int i=0; i< 4; ++i){

           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();

           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propUP, true));

           TIME(reductionsV2_diluted[i].V2( st_oet_u_zero, glist_sink_nucleon, propDN, propDN, false));

         }


         TIME(corrZ15.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 15 ));
         TIME(corrZ16.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 16 ));

         //Diagram Z17,Z18
         for (int i=0; i< 4; ++i){

           PLEGMA_Vector<float> st_oet_d_zero;
           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();

           st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom[i],HOST); 
           st_oet_d_zero.load();

           TIME(reductionsV3_diluted_zero_pf2[i].V3( st_oet_u_zero, gamma_5_t_sinkmeson, propUP, true));

           TIME(reductionsV2_diluted[i].V2( st_oet_d_zero, glist_sink_nucleon, propUP, propDN, false));

         }


         TIME(corrZ17.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 17 ));
         TIME(corrZ18.Z_diagramms( reductionsV3_diluted_zero_pf2, reductionsV2_diluted, 18 ));

         //Diagram Z19,Z20
         for (int i=0; i< 4; ++i){

           PLEGMA_Vector<float> st_oet_d_zero;

           st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom[i],HOST);
           st_oet_d_zero.load();

           TIME(reductionsV2_diluted[i].V4( st_oet_d_zero, glist_sink_nucleon, propUP, propDN, false));

         }

         TIME(corrZ19.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 19 ));
         TIME(corrZ20.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 20 ));

         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

         produceOutput(corrZ5, outfilename, "Z");
         produceOutput(corrZ6, outfilename, "Z");
         produceOutput(corrZ7, outfilename, "Z");
         produceOutput(corrZ8, outfilename, "Z");

         produceOutput(corrZ9, outfilename, "Z");
         produceOutput(corrZ10, outfilename,"Z");

         produceOutput(corrZ11, outfilename,"Z");
         produceOutput(corrZ12, outfilename,"Z");
         produceOutput(corrZ13, outfilename,"Z");
         produceOutput(corrZ14, outfilename,"Z");

         produceOutput(corrZ15, outfilename,"Z");
         produceOutput(corrZ16, outfilename,"Z");

         produceOutput(corrZ17, outfilename,"Z");
         produceOutput(corrZ18, outfilename,"Z");
         produceOutput(corrZ19, outfilename,"Z");
         produceOutput(corrZ20, outfilename,"Z");

       }
       else{
         for (int i=0; i< 4; ++i){
           PLEGMA_Vector<float> st_oet_u_fini;
           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_fini.copy(*stochastic_oet_prop_u_fini_mom[i],HOST);
           st_oet_u_fini.load();

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();

           TIME(reductionsV3_diluted[i].V3( st_oet_u_fini, gamma_5_t_sinkmeson, propUP, true));

           TIME(reductionsV2_diluted[i].V4( st_oet_u_zero, glist_sink_nucleon, propDN, propUP, false));
         }

         TIME(corrZ1.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 1 ));
         TIME(corrZ2.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 2 ));

         for (int i=0; i< 4; ++i){
           PLEGMA_Vector<float> st_oet_u_zero;

           st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
           st_oet_u_zero.load();

           TIME(reductionsV2_diluted[i].V2( st_oet_u_zero, glist_sink_nucleon, propDN, propUP, false));

         }

         TIME(corrZ3.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 3 ));
         TIME(corrZ4.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 4 ));

       }

       if  ((momentum_i2[0] == 0) && (momentum_i2[1] == 0) && (momentum_i2[2] == 0)){

         PLEGMA_ScattCorrelator<float> corrD1if12(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         PLEGMA_ScattCorrelator<float> corrD1if34(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         PLEGMA_ScattCorrelator<float> corrD1if56(sourcePositions[isource], filtered_sourcemomentumList_pi20pf20);

         TIME(corrD1if34.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP01"));

         TIME(corrD1if12.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP02"));

         TIME(corrD1if56.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MN0PP"));

 
         TIME(corrD1if56.M_diagramms( corrN0, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom));
         TIME(corrD1if34.M_diagramms( corrNP, stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_u_zero_mom));
         TIME(corrD1if12.M_diagramms( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_d_zero_mom));

         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";
         produceOutput(corrD1if12, outfilename,"M");
         produceOutput(corrD1if34, outfilename,"M");
         produceOutput(corrD1if56, outfilename,"M");

       }


       //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
         TIME(corrP.P_diagramms( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom, i_mpi2));

         TIME(corrM.M_diagramms( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom ));
       
       }
       else{
         TIME(corrP.P_diagramms( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom, i_mpi2));


         TIME(corrM.M_diagramms( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom ));
       }
	

       //write everything
       //## T
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
     
       TIME(corrT.apply_phase());
       TIME(corrT.apply_sign("T"));
       TIME(corrT.applyBoundaryConditions( true ));

       TIME(corrT.writeHDF5(outfilename));
       
       //## B
       
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";

       produceOutput(corrB1, outfilename, "B", n_stochastic_samples);
       produceOutput(corrB2, outfilename, "B", n_stochastic_samples);

       //## W
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";

       produceOutput(corrW1, outfilename, "W", n_stochastic_samples);
       produceOutput(corrW2, outfilename, "W", n_stochastic_samples);
       produceOutput(corrW3, outfilename, "W", n_stochastic_samples);
       produceOutput(corrW4, outfilename, "W", n_stochastic_samples);

       //## Z

       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

       produceOutput(corrZ1, outfilename,"Z");
       produceOutput(corrZ2, outfilename,"Z");
       produceOutput(corrZ3, outfilename,"Z");
       produceOutput(corrZ4, outfilename,"Z");

       //## M
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";

       //TIME(corrM.writeHDF5( "mdiagrammwithoutphase" ));

       produceOutput(corrM, outfilename,"M");
      
      }//loop over unique set of momenta for p_i2
       
      //write P

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrP.apply_sign("P"));
      TIME(corrP.writeHDF5( outfilename ));

      produceOutput(corrNP, outfilename,"N");
      produceOutput(corrN0, outfilename,"N");


      for(int i=0; i< n_stochastic_samples; ++i) {

        reductions_UU_V2_GAMMAF1D_U.pop_back();
        reductions_UU_V4_GAMMAF1D_U.pop_back();
        reductions_UU_V4_GAMMAF1U_D.pop_back();
        reductions_UU_V3_GAMMAF2U_zero_mom.pop_back();

        reductions_DD_V3_GAMMAF2U.pop_back();
        reductions_DD_V2_GAMMAF1U_U.pop_back();
        reductions_DD_V3_GAMMAF2D_zero_mom.pop_back();
        reductions_DD_V3_GAMMAF2U_zero_mom.pop_back();
        reductions_DD_V2_GAMMAF1U_D.pop_back();
        reductions_DD_V2_GAMMAF1D_U.pop_back();
        reductions_DD_V4_GAMMAF1U_D.pop_back();

      }

    } //loop over source position


    for(int i=0; i< 4; ++i) {
      stochastic_oet_prop_d_zero_mom.pop_back();
      stochastic_oet_prop_u_zero_mom.pop_back();
      stochastic_oet_prop_u_fini_mom.pop_back();
    }

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      stochastic_propags.pop_back();
    }

  } 
  finalize();
  
  return 0;
}
