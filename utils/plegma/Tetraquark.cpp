#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <string>
#include <fstream>



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
                        
site ReadAndDeleteBottomFileName(const std::string &filename_file, bool &continuing){
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
     source[i] = atoi(token.c_str());
     i++;
    sourceLine.erase(0, pos + 1);
    }

    source[i] = atoi(sourceLine.c_str());  
    MPI_Barrier(MPI_COMM_WORLD);
    
return source;
}                         
  
// **********************************************************************************************************//

void DeleteBottomPropagatorFile(const std::string &bottom_prop_filename){
    
    int rank = 0;
  
    MPI_Initialized(&rank);
    if(rank) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0) { 
          remove(bottom_prop_filename.c_str());  
        }
    }
}
                        
// **********************************************************************************************************//
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
   double kappa_s;
//   std::vector<double> mu_c;
  double kappa_ud = kappa;
//  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
//  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss;
//   int nsmearGauss_c = 0;
  int startSource = 0;
  std::string light_prop_fileprefix;
//   std::string light_prop_filename;
  std::string bottom_prop_fileprefix;
  std::string bottom_prop_filename;
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
    options.set("kappa-s", "kappa_s to run for the strange quark in tetraquarks", verbosity, kappa_s);
//     options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
//     options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
    options.set("src-input-file", "Use the file to get finished bottom propagators", verbosity, srcInputFile);
    options.set("bottom-prop-filename", "Filename of bottom propagator to be read in", verbosity, bottom_prop_fileprefix);
//     options.set("light-prop-filename", "Filename of light propagator to be read in", verbosity, light_prop_fileprefix);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
		     };
  add_options(*HGC_options);
  //=========================================================================================================//
  initializePLEGMA();
  
  
  std::string openContr_filename = twop_filename + std::string("_openContractions_")+ ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);
  
  
  twop_filename += std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

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

    std::string given_twop_filename = twop_filename;

    
    bool running = true;
    int isource=0;
    while (running){
        isource++;
//     for(int isource = startSource; isource < numSourcePositions; isource++){
//         site& source = sourcePositions[isource];
//         PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
//         isource, source[0], source[1], source[2], source[3]);

      // XXX  necessary???  
  //    updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);
        
    site source = ReadAndDeleteBottomFileName(srcInputFile, running);
        
    if (running == false){
        break;
    }
    PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
    isource, source[0], source[1], source[2], source[3]);
    
    
    PLEGMA_Propagator<float> propLT;
    PLEGMA_Propagator<float> propST;
    PLEGMA_Propagator<float> propBT;
    
    char * prop_string;
    asprintf(&prop_string, "x%02dy%02dz%02dt%03d", source[0], source[1], source[2], source[3]);
    bottom_prop_filename = bottom_prop_fileprefix + prop_string +"_lime";
    free(prop_string);
    
       
    TIME(propBT.readFile(bottom_prop_filename, LIME_FORMAT));
    MPI_Barrier(MPI_COMM_WORLD);
    DeleteBottomPropagatorFile(bottom_prop_filename);

    
    PLEGMA_Gauge3D<double> smearedGauge3D;
    smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

    
    auto computePropagator = [&](PLEGMA_Propagator<float>& prop,double run_kappa, WHICHFLAVOR fl, int nSmear) {
        
            // ensuring kappa value
            if(kappa != run_kappa) {
            updateOptions(fl);
            kappa = run_kappa;
            solver.UpdateSolver();
            }
            for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorInOut;
            { // Smearing the source
                PLEGMA_Vector3D<double> vector1, vector2;
                vector1.pointSource(source, isc/3, isc%3, DEVICE);
                TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                vectorInOut.absorb(vector2,source[DIM_T]);
            }
            // Inverting
            PLEGMA_printf("Going to invert %s for component %d\n",
                          fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
            TIME(solver.solve(vectorInOut, vectorInOut));
            { // Smearing the solution
                PLEGMA_Vector<double> vectorAuxD;
                PLEGMA_Vector<float> vectorAuxF;
                TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                vectorAuxF.copy(vectorAuxD);
                prop.absorb(vectorAuxF, isc/3, isc%3);
            }
            }
//              prop.applyBoundaries_device(source[DIM_T]);
            };
            
            
          
        {
   
       TIME(computePropagator(propLT, kappa_ud, LIGHT, nsmearGauss));
       TIME(computePropagator(propST, kappa_s, STRANGE, nsmearGauss_s));
       
        

        PLEGMA_Propagator<float> none(NONE);
        PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
        char * group;
        
        
        TIME(corr.contractTetraquarks(propLT, propST, none, propBT));
        asprintf(&group, "tetraquarks");
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));
        
        
        std::vector<GAMMAS> gammas = {G1,G2,G3,G5};
        
        for (int i=0; i<gammas.size(); i++){
            std::vector<GAMMAS> gamma = {gammas[i]};
            for (int s1=0; s1< N_SPINS; s1++){ // this sum is required as the GPU memory is not large enough to cover all 4*4*3*3 elements in double precision
                TIME(corr.contractTetraquarkScatteringOpenIndex(propBT, propLT, gamma, s1,"BU"));
                THREAD(corr.writeFile(openContr_filename, corr_file_format));
                
                TIME(corr.contractTetraquarkScatteringOpenIndex(propBT, propST, gamma, s1,"BS"));
                THREAD(corr.writeFile(openContr_filename, corr_file_format));
            }
        }
        
	
	
//         TIME(corr.contractMesonsNew(propLT, propBT ));
//         asprintf(&group, "B_mesons");
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
//         
//         TIME(corr.contractMesonsNew(propST, propBT ));
//         asprintf(&group, "Bs_mesons");
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
//         
//         TIME(corr.contractMesonsNew(propLT, propLT ));
//         asprintf(&group, "Light_mesons");
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
//         
//         TIME(corr.contractMesonsNew(propBT, propBT ));
//         asprintf(&group, "Heavy_mesons");
//         corr.setGroups(group);
//         free(group);
//         THREAD(corr.writeFile(twop_filename, corr_file_format));
        
        
        PLEGMA_Correlator<double> corr_meson(corr_space, source, 4);
        
        
        TIME(corr_meson.contractMesonsNew(propLT, propBT ));
        asprintf(&group, "B_mesons");
        corr_meson.setGroups(group);
        free(group);
        THREAD(corr_meson.writeFile(twop_filename, corr_file_format));
        
        TIME(corr_meson.contractMesonsNew(propST, propBT ));
        asprintf(&group, "Bs_mesons");
        corr_meson.setGroups(group);
        free(group);
        THREAD(corr_meson.writeFile(twop_filename, corr_file_format));
        
        TIME(corr_meson.contractMesonsNew(propLT, propLT ));
        asprintf(&group, "Light_mesons");
        corr_meson.setGroups(group);
        free(group);
        THREAD(corr_meson.writeFile(twop_filename, corr_file_format));
        
        TIME(corr_meson.contractMesonsNew(propBT, propBT ));
        asprintf(&group, "Heavy_mesons");
        corr_meson.setGroups(group);
        free(group);
        THREAD(corr_meson.writeFile(twop_filename, corr_file_format));
	

      }
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();
  return 0;
}

