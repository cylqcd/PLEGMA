#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <string>
#include <fstream>
#include <vector>



std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};
                        
                        
// **********************************************************************************************************//
                        
site ReadAndDeleteBottomFileName(const std::string &filename_file, bool &continuing, int &randomIndex){
    
 int rank = 0;
 site source;
 std::string sourceLine;


 MPI_Initialized(&rank);
 if(rank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0) {
     
		std::string line;
		std::ifstream fin;
		fin.open(filename_file);
		std::string tempfile_name = "tempFiles/" + filename_file.substr(filename_file.find_last_of('/') + 1)  + "_tmp";
		std::ofstream temp;
		temp.open(tempfile_name);

		if(getline(fin,sourceLine)){
			while (getline(fin,line)){
				temp << line << std::endl;
			}
			temp.close();
			fin.close();
			remove(filename_file.c_str());
			rename(tempfile_name.c_str(), filename_file.c_str());

			continuing=true;
			}
			else{
				continuing= false;
				temp.close();
				remove(tempfile_name.c_str());
			}
		}

		MPI_Bcast(&continuing, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
		int line_size = sourceLine.size();
		MPI_Bcast(&line_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (rank != 0){
			sourceLine.resize(line_size);}
		MPI_Bcast(const_cast<char*>(sourceLine.data()), line_size, MPI_CHAR, 0, MPI_COMM_WORLD);
 	 }
  else{
     PLEGMA_error("Only single mode, reading not done!");
  }
 
 if (continuing == false){
    return source;
 }
       
    size_t pos = 0;
    int i=0;
    std::string token;
    while ((pos = sourceLine.find(" ")) != std::string::npos) {
		token = sourceLine.substr(0, pos);
		 source[DIM_T] = atoi(token.c_str());
		 i++;
		sourceLine.erase(0, pos + 1);
    }
    
    //as we have stochastic sources, we set the spatial postion to 0 (just required because correlator needs a source)
    for(int i=0;i<3; i++){
        source[i]=0;
    }

    randomIndex = atoi(sourceLine.c_str());  
    MPI_Barrier(MPI_COMM_WORLD);
    
return source;
} 

// **********************************************************************************************************//

std::string ReadAndDeletePropListFileName(const std::string &filename_file, bool &continuing){
    
 int rank = 0;
 site source;
 std::string sourceLine;

 MPI_Initialized(&rank);
 if(rank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0) {
     
		std::string line;

		std::ifstream fin;
		fin.open(filename_file);
		std::string tempfile_name = "tempFiles/" + filename_file.substr(filename_file.find_last_of('/') + 1)  + "_tmp";
		std::ofstream temp;
		temp.open(tempfile_name);

		if(getline(fin,sourceLine)){
			while (getline(fin,line)){
				temp << line << std::endl;
			}

			temp.close();
			fin.close();
			remove(filename_file.c_str());
			rename(tempfile_name.c_str(), filename_file.c_str());

			continuing=true;
		}
		else{
			continuing= false;
			temp.close();
			remove(tempfile_name.c_str());
		}
    }
 
	 MPI_Bcast(&continuing, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
	 int line_size = sourceLine.size();
	 MPI_Bcast(&line_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	 if (rank != 0){
		 sourceLine.resize(line_size);}
	 	 MPI_Bcast(const_cast<char*>(sourceLine.data()), line_size, MPI_CHAR, 0, MPI_COMM_WORLD);
 	 }
 	 else{
 		 PLEGMA_error("Only single mode, reading not done!");
 	 }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    return sourceLine;
} 

// **********************************************************************************************************//

void DeleteBottomPropagatorFile(const std::string &bottom_prop_filename, const std::string &stoch_source_filename){
    
    int rank = 0;
  
    MPI_Initialized(&rank);
    if(rank) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0) { 
          remove(bottom_prop_filename.c_str()); 
          remove(stoch_source_filename.c_str()); 
        }
    }
}
                        
// **********************************************************************************************************//

  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
   double kappa_c;
//   std::vector<double> kappa_c;
  double kappa_ud = kappa;
//  double kappa_ud_factor[QUDA_MAX_MG_LEVEL];
//  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) kappa_ud_factor[i] = kappa_factor[i];
    int nsmearGauss_c = nsmearGauss;
//   int nsmearGauss_c = 0;
  int startSource = 0;
  int storedProps = 0;
  std::string stoch_source_fileprefix;
  std::string stoch_source_filename;
  std::string bottom_prop_fileprefix;
  std::string bottom_prop_filename;
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
    options.set("kappa-c", "kappa_c to run for the charm quark in tetraquarks", verbosity, kappa_c);
//     options.set("kappa-c", "List of kappa_c to run for the charm quark in baryons", verbosity, kappa_c);
//     options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
     options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
    options.set("src-input-file", "Use the file to get finished bottom propagators", verbosity, srcInputFile);
    options.set("bottom-prop-filename", "Filename of bottom propagator to be read in", verbosity, bottom_prop_fileprefix);
    options.set("stoch-source-filename", "Filename of light propagator to be read in", verbosity, stoch_source_fileprefix);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("stored-props", "The number of propagators that are kept in storage during computation (at least 2 rquired for stochastic props)", verbosity, storedProps);
		     };
  add_options(*HGC_options);
  //=========================================================================================================//
  initializePLEGMA();

  std::vector<std::vector<int>> momentumSources{ {0,0,0}, {0,0,1},  {0,0,-1}, {0,1,0}, {0,-1,0},  {1,0,0}, {-1,0,0},
                                                 {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1},
                                                 {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1},
                                                 {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0},
                                                 {1,1,1}, {1,1,-1}, {1,-1,1}, {-1,1,1}, {1,-1,-1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1}, };

  
  
  if( storedProps < 2){
      PLEGMA_error("Number of stored props must be larger or equal 2!");
  }
  
  std::string openContr_filename = twop_filename + std::string("_openContractions_")+ ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);
  
  
//   twop_filename += std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
//     "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
//     "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
 //   PLEGMA_Gauge<float> contractGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();
      
      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();

      // Gauge for contractions
   //   contractGauge.copy(gauge);
      // apply boundary conditions since is needed for the covariant derivative
  //    applyBoundaryConditions(contractGauge,true);
   }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(0));

//     std::string given_twop_filename = twop_filename;
    
    
    PLEGMA_Propagator<float> propLT[storedProps];   
    PLEGMA_Propagator<float> propCH[storedProps];
    PLEGMA_Propagator<float> propBT[storedProps];
    PLEGMA_Propagator<float> stochSource[storedProps];
    PLEGMA_Propagator<float> none(NONE);
    
    int itime=0;
    bool runningTime = true;
    while (runningTime){
    
        std::string SourceFiles = ReadAndDeletePropListFileName(srcInputFile, runningTime);
    
        if (runningTime == false){
            break;
        }
        
        PLEGMA_printf("\n ### Treating Timeslice %02d \n\n",itime);
        itime++;
        
        


        bool running = true;
        int isource=-1;
        std::vector<site> source;
        while (running){
            isource++;
            int indNew = isource%storedProps;

            int randomIndex;
            site source_tmp = ReadAndDeleteBottomFileName(SourceFiles, running, randomIndex);
            source.push_back(source_tmp);

            if (running == false){
                break;
            }
            PLEGMA_printf("\n ### Calculations for source-position %d - timeslice %03d, randomSource %02d begin now ###\n\n",
            isource, source[isource][DIM_T], randomIndex);



            char * prop_string;
            asprintf(&prop_string, "t%03d.r%02d",source[isource][DIM_T],randomIndex);
            bottom_prop_filename = bottom_prop_fileprefix + prop_string +"_lime";
            stoch_source_filename = stoch_source_fileprefix + prop_string +"_lime";
            free(prop_string);
        
        

                
            TIME(propBT[indNew].readFile(bottom_prop_filename, LIME_FORMAT));
            TIME(stochSource[indNew].readFile(stoch_source_filename, LIME_FORMAT));
            MPI_Barrier(MPI_COMM_WORLD);
            DeleteBottomPropagatorFile(bottom_prop_filename, stoch_source_filename);

        
            PLEGMA_Gauge3D<double> smearedGauge3D;
            smearedGauge3D.absorb(smearedGauge, source[isource][DIM_T]);
                
                
            auto computeStochasticPropagator = [&](PLEGMA_Propagator<float>& prop,double run_kappa, WHICHFLAVOR fl, int nSmear) {
                    // ensuring kappa value
                    if(kappa != run_kappa) {
                    updateOptions(fl);
                    kappa = run_kappa;
                    solver.UpdateSolver();
                    }
                    for(int isc = 0 ; isc < 12 ; isc++){
                        PLEGMA_Vector3D<float> vectorSource;
                        PLEGMA_Vector<double> vectorInOut;
                        vectorSource.absorb(stochSource[indNew], source[isource][DIM_T], isc/3,isc%3);
                        { // Smearing the source
                            PLEGMA_Vector3D<double> vector1, vector2;
                            vector1.copy(vectorSource);
                            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                            vectorInOut.absorb(vector2,source[isource][DIM_T]);
                        }
                        // Inverting
                        PLEGMA_printf("Going to invert %s for component %d\n",
                                    fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"),isc);
                        TIME(solver.solve(vectorInOut, vectorInOut));
                        { // Smearing the solution
                            PLEGMA_Vector<double> vectorAuxD;
                            PLEGMA_Vector<float> vectorAuxF;
                            TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                            vectorAuxF.copy(vectorAuxD);
                            prop.absorb(vectorAuxF, isc/3, isc%3);
                        }
                    }
    //                 prop.applyBoundaries_device(source[DIM_T]);
            };

            TIME(computeStochasticPropagator(propCH[indNew], kappa_c, CHARM, nsmearGauss_c));





        

        {
    
//         TIME(computeStochasticPropagator(propLT[indNew], kappa_ud, LIGHT, nsmearGauss));
//         TIME(computeStochasticPropagator_momentum(propLT[indNew], kappa_ud, LIGHT, nsmearGauss, momentumSources[imom]));


        int currentLowestN= isource-(storedProps-1);
        if (isource == (storedProps-1)){

            for (int imom=0; imom< momentumSources.size(); imom++){
                std::string mom_string = "p"+ std::to_string(momentumSources[imom][0]) + std::to_string(momentumSources[imom][1]) +std::to_string(momentumSources[imom][2]);

                for (int iprop=0; iprop< storedProps; iprop++){

                    PLEGMA_Gauge3D<double> smearedGauge3D;
                    smearedGauge3D.absorb(smearedGauge, source[iprop][DIM_T]);

                    auto computeStochasticPropagator_momentum = [&](PLEGMA_Propagator<float>& prop,
                                   double run_kappa, WHICHFLAVOR fl, int nSmear,  std::vector<int> sourceMom) {
                            // ensuring kappa value
                            if(kappa != run_kappa) {
                            updateOptions(fl);
                            kappa = run_kappa;
                            solver.UpdateSolver();
                            }
                            for(int isc = 0 ; isc < 12 ; isc++){
                                PLEGMA_Vector3D<float> vectorSource;
                                PLEGMA_Vector<double> vectorInOut;
                                vectorSource.absorb(stochSource[iprop], source[iprop][DIM_T], isc/3,isc%3);
                                { // Smearing the source
                                    PLEGMA_Vector3D<double> vector1, vector2;
                                    vector1.copy(vectorSource);
                                    vector1.mulMomentumPhases(sourceMom,-1);
                                    TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                                    vectorInOut.absorb(vector2,source[iprop][DIM_T]);
                                }
                                // Inverting
                                PLEGMA_printf("Going to invert %s for component %d\n",
                                            fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"),isc);
                                TIME(solver.solve(vectorInOut, vectorInOut));
                                { // Smearing the solution
                                    PLEGMA_Vector<double> vectorAuxD;
                                    PLEGMA_Vector<float> vectorAuxF;
                                    TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                                    vectorAuxF.copy(vectorAuxD);
                                    prop.absorb(vectorAuxF, isc/3, isc%3);
                                }
                            }
                    };

                    TIME(computeStochasticPropagator_momentum(propLT[iprop], kappa_ud, LIGHT, nsmearGauss, momentumSources[imom]));
                }



//                         int currentLowestN= isource-(storedProps-1);
                PLEGMA_Correlator<double> corr(corr_space, source[0], maxQsq);
                char * group;

                for (int randInd1=0; randInd1< storedProps; randInd1++){
                    for (int randInd2=0; randInd2< storedProps; randInd2++){


                        int totalInd1 = randInd1; // + currentLowestN;
                        int totalInd2 = randInd2; // + currentLowestN;

//                         std::string group_name = "tetraquarks/"+ mom_string +"/r"  + std::to_string(totalInd1) +"r"  + std::to_string(totalInd2);


//                         if (randInd1 != randInd2){
//
//                             TIME(corr.contractTetraquarksStochasticBCUD(propLT[randInd1], propLT[randInd2], none, none, propCH[randInd1], propCH[randInd2], propBT[randInd1], propBT[randInd2]));
//                             asprintf(&group, group_name.c_str());
//                             corr.setGroups(group);
//                             free(group);
//                             THREAD(corr.writeFile(twop_filename, corr_file_format));
//                         }

                        std::vector<GAMMAS> gammas = {G1,G2,G3,G5};

                        for (int i=0; i<gammas.size(); i++){
                            std::vector<GAMMAS> gamma = {gammas[i]};
                            for (int s1=0; s1< N_SPINS; s1++){  // this sum is required as the GPU memory is not large enough to cover all 4*4*3*3 elements in double precision
                                TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propBT[randInd1], propLT[randInd2], gamma, totalInd1, totalInd2, s1, mom_string +"/BU"));
                                THREAD(corr.writeFile(openContr_filename, corr_file_format));

                                TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propCH[randInd1], propLT[randInd2], gamma, totalInd1, totalInd2, s1, mom_string +"/DU"));
                                THREAD(corr.writeFile(openContr_filename, corr_file_format));
                            }
                        }

                    }
                }
            }
        }



//             int currentLowestN= isource-(storedProps-1);
//             PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
//             char * group;
            
            
//                 PLEGMA_printf("corr.getVolSize(): %d\n", corr.getVolSize());
//     PLEGMA_printf("corr.getTotalSize(): %d\n", corr.getTotalSize());
//     PLEGMA_printf("corr.getSiteSize(): %d\n", corr.getSiteSize());
        
//             if (isource == (storedProps-1)){
//
//                 for (int randInd1=0; randInd1< storedProps; randInd1++){
//                     for (int randInd2=0; randInd2< storedProps; randInd2++){
//
//
//                         int totalInd1 = randInd1 + currentLowestN;
//                         int totalInd2 = randInd2 + currentLowestN;
//
//                         std::string group_name = "tetraquarks/"+ mom_string +"/r"  + std::to_string(totalInd1) +"r"  + std::to_string(totalInd2);
//
//                         if (randInd1 != randInd2){
//
//                             TIME(corr.contractTetraquarksStochasticBCUD(propLT[randInd1], propLT[randInd2], none, none, propCH[randInd1], propCH[randInd2], propBT[randInd1], propBT[randInd2]));
//                             asprintf(&group, group_name.c_str());
//                             corr.setGroups(group);
//                             free(group);
//                             THREAD(corr.writeFile(twop_filename, corr_file_format));
//                         }
//
//                         std::vector<GAMMAS> gammas = {G1,G2,G3,G5};
//
//                         for (int i=0; i<gammas.size(); i++){
//                             std::vector<GAMMAS> gamma = {gammas[i]};
//                             for (int s1=0; s1< N_SPINS; s1++){  // this sum is required as the GPU memory is not large enough to cover all 4*4*3*3 elements in double precision
//                                 TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propBT[randInd1], propLT[randInd2], gamma, totalInd1, totalInd2, s1, mom_string +"/BU"));
//                                 THREAD(corr.writeFile(openContr_filename, corr_file_format));
//
//                                 TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propCH[randInd1], propLT[randInd2], gamma, totalInd1, totalInd2, s1, mom_string +"/DU"));
//                                 THREAD(corr.writeFile(openContr_filename, corr_file_format));
//                             }
//                         }
//
//                     }
//                 }
//             }
            
            
            if (isource >= (storedProps)){

                int totalIndNew = isource;


                for (int imom=0; imom< momentumSources.size(); imom++){
                std::string mom_string = "p"+ std::to_string(momentumSources[imom][0]) + std::to_string(momentumSources[imom][1]) +std::to_string(momentumSources[imom][2]);

                PLEGMA_Gauge3D<double> smearedGauge3D;
                smearedGauge3D.absorb(smearedGauge, source[isource][DIM_T]);
                auto computeStochasticPropagator_momentum = [&](PLEGMA_Propagator<float>& prop,
                            double run_kappa, WHICHFLAVOR fl, int nSmear,  std::vector<int> sourceMom) {
                            // ensuring kappa value
                            if(kappa != run_kappa) {
                            updateOptions(fl);
                            kappa = run_kappa;
                            solver.UpdateSolver();
                            }
                            for(int isc = 0 ; isc < 12 ; isc++){
                                PLEGMA_Vector3D<float> vectorSource;
                                PLEGMA_Vector<double> vectorInOut;
                                vectorSource.absorb(stochSource[indNew], source[isource][DIM_T], isc/3,isc%3);
                                { // Smearing the source
                                    PLEGMA_Vector3D<double> vector1, vector2;
                                    vector1.copy(vectorSource);
                                    vector1.mulMomentumPhases(sourceMom,-1);
                                    TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                                    vectorInOut.absorb(vector2,source[isource][DIM_T]);
                                }
                                // Inverting
                                PLEGMA_printf("Going to invert %s for component %d\n",
                                            fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"),isc);
                                TIME(solver.solve(vectorInOut, vectorInOut));
                                { // Smearing the solution
                                    PLEGMA_Vector<double> vectorAuxD;
                                    PLEGMA_Vector<float> vectorAuxF;
                                    TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                                    vectorAuxF.copy(vectorAuxD);
                                    prop.absorb(vectorAuxF, isc/3, isc%3);
                                }
                            }
                    };

                    TIME(computeStochasticPropagator_momentum(propLT[indNew], kappa_ud, LIGHT, nsmearGauss, momentumSources[imom]));



                    PLEGMA_Correlator<double> corr(corr_space, source[0], maxQsq);
                    char * group;


                    for (int randInd=0; randInd< storedProps; randInd++){

                            int totalInd = (randInd + storedProps -(currentLowestN%storedProps) )%storedProps + currentLowestN;


                            if (randInd != indNew){
//                                 std::string group_name = "tetraquarks/"+ mom_string +"/r"  + std::to_string(totalInd) +"r"  + std::to_string(totalIndNew);
//                                 PLEGMA_printf("GroupName:  %s\n", group_name);
//                                 TIME(corr.contractTetraquarksStochasticBCUD(propLT[randInd], propLT[indNew],none, none, propCH[randInd], propCH[indNew], propBT[randInd], propBT[indNew]));
//                                 asprintf(&group, group_name.c_str());
//                                 corr.setGroups(group);
//                                 free(group);
//                                 THREAD(corr.writeFile(twop_filename, corr_file_format));
//
//                                 group_name = "tetraquarks/"+ mom_string +"/r"  + std::to_string(totalIndNew) +"r"  + std::to_string(totalInd);
//                                 TIME(corr.contractTetraquarksStochasticBCUD(propLT[indNew], propLT[randInd],none, none, propCH[indNew], propCH[randInd], propBT[indNew], propBT[randInd]));
//                                 asprintf(&group, group_name.c_str());
//                                 corr.setGroups(group);
//                                 free(group);
//                                 THREAD(corr.writeFile(twop_filename, corr_file_format));

                                std::vector<GAMMAS> gammas = {G1,G2,G3,G5};

                                for (int i=0; i<gammas.size(); i++){
                                    std::vector<GAMMAS> gamma = {gammas[i]};
                                    for (int s1=0; s1< N_SPINS; s1++){
                                        TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propBT[randInd], propLT[indNew], gamma, totalInd, totalIndNew, s1, mom_string +"/BU"));
                                        THREAD(corr.writeFile(openContr_filename, corr_file_format));

                                        TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propBT[indNew], propLT[randInd], gamma, totalIndNew, totalInd ,s1, mom_string +"/BU"));
                                        THREAD(corr.writeFile(openContr_filename, corr_file_format));


                                        TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propCH[randInd], propLT[indNew], gamma, totalInd, totalIndNew, s1, mom_string +"/DU"));
                                        THREAD(corr.writeFile(openContr_filename, corr_file_format));

                                        TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propCH[indNew], propLT[randInd], gamma, totalIndNew, totalInd ,s1, mom_string +"/DU"));
                                        THREAD(corr.writeFile(openContr_filename, corr_file_format));
                                    }
                                }
                            }

                        }

                        std::vector<GAMMAS> gammas = {G1,G2,G3,G5};

                            for (int i=0; i<gammas.size(); i++){
                                std::vector<GAMMAS> gamma = {gammas[i]};
                                for (int s1=0; s1< N_SPINS; s1++){
                                    TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propBT[indNew], propLT[indNew], gamma, totalIndNew, totalIndNew, s1, mom_string +"/BU"));
                                    THREAD(corr.writeFile(openContr_filename, corr_file_format));

                                    TIME(corr.contractTetraquarkScatteringOpenIndexStochastic(propCH[indNew], propLT[indNew], gamma, totalIndNew, totalIndNew, s1, mom_string +"/DU"));
                                    THREAD(corr.writeFile(openContr_filename, corr_file_format));
                                }

                            }
                }
            
            }
        
//         std::string group_name;
        
//         TIME(corr.contractMesonsNew(propLT[indNew], propBT[indNew] ));
//         group_name = "B_mesons/r"  + std::to_string(isource) +"r"  + std::to_string(isource);
//         asprintf(&group, group_name.c_str());
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
//
//         TIME(corr.contractMesonsNew(propST[indNew], propBT[indNew] ));
//         group_name = "Bs_mesons/r"  + std::to_string(isource) +"r"  + std::to_string(isource);
//         asprintf(&group, group_name.c_str());
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
//
//         TIME(corr.contractMesonsNew(propLT[indNew], propLT[indNew] ));
//         group_name = "Light_mesons/r"  + std::to_string(isource) +"r"  + std::to_string(isource);
//         asprintf(&group, group_name.c_str());
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
//
//         TIME(corr.contractMesonsNew(propBT[indNew], propBT[indNew] ));
//         group_name = "Heavy_mesons/r"  + std::to_string(isource) +"r"  + std::to_string(isource);
//         asprintf(&group, group_name.c_str());
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
	
        }
        }
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();
  return 0;
}

