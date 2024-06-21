#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc) runtime.push_back(MPI_Wtime()); fnc;                     \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back());  \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;

int main(int argc, char **argv) {

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "twop-filename"};

  initializeOptions(argc, argv, false, listOpt);

  size_t Zdir = 2;
  HGC_options->set("Zdir", "Direction of longitudinal part of staple", verbosity, Zdir);

  size_t Tdir = 3;
  HGC_options->set("Tdir", "Direction of transverse part of staple", verbosity, Tdir);

  size_t Zmax = 0;
  HGC_options->set("Zmax", "Maximum length of the longitudinal part of staple", verbosity, Zmax);

  size_t Tmax = 0;
  HGC_options->set("Tmax", "Maximum length of the transverse direction of staple", verbosity, Tmax);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t nStout;
  HGC_options->set("nstout", "Number of stout smearing steps", verbosity, nStout);

  initializePLEGMA();

  if(Tmax == 0) Tmax = HGC_totalL[Tdir]/2;
  if(Zmax == 0) Zmax = HGC_totalL[Zdir]/2;

  FILE *p_file1, *p_file2;
  float WLval;

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  PLEGMA_status();
  gauge.calculatePlaq();

  //copy gauge to gaugeWL for building of the staples
  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

  //apply the stout smearing steps
  if(nStout > 0) gaugeWL.stoutSmearing(gaugeWL, nStout, rhoStout, 3);

  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_3 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_temp1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_temp2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);

  //set the 4 branches as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, Tdir);
  if(getRankVerbosity()) {
    p_file1 = fopen((twop_filename+".stout"+std::to_string(nStout)+".Tdir"+std::to_string(Tdir)+".Zdir"+std::to_string(Zdir)+".TPlus.ZPlus.dat").c_str(),"w");
    p_file2 = fopen((twop_filename+".stout"+std::to_string(nStout)+".Tdir"+std::to_string(Tdir)+".Zdir"+std::to_string(Zdir)+".TPlus.ZMinus.dat").c_str(),"w");
  }

  for(int t = 0; t <= Tmax; t++) {//loop over values of L, building "1" and "3"

    if(t != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + Tdir);
    }

    WL_2->setUnit( (std::vector<int>) {0,4,8});
    WL_3->copy(*WL_1);
    su3_2->absorbDir_device(gaugeWL, Zdir);
    for(int ti = 0; ti < t; ti++){
      su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
      su3_2->shift(*su3_in, 4 + Tdir);
    }

    for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

      if(z != 0) {
        WL_2->wilsonLineUpdate(*su3_2, tmp, 4 + Zdir);
        su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
        WL_3->shift(*su3_in, 4 + Zdir);
      }
          
      WL_temp1->UxU(*WL_1, *WL_2);
      WL_temp2->UxUdag(*WL_temp1, *WL_3);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, Tdir);
      }

      WL->UxUdag(*WL_temp2, *WL_2);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, 4 + Tdir);
      }
      
      //WL->writeHDF5( ("WL.stout"+std::to_string(nStout)+".Zdir"+std::to_string(Zdir)+".TPlus.ZPlus.h5/T_"+std::to_string(t)+"Z_"+std::to_string(z)).c_str(), true );
      WLval = WL->sumRtraceU()/(HGC_totalVolume*3);
      if(getRankVerbosity()) {
        fprintf(p_file1,"%d,%d,%f\n",t,z,WLval);
      }

    }//loop over values of B, building "2"

    WL_2->setUnit( (std::vector<int>) {0,4,8});
    WL_3->copy(*WL_1);
    su3_2->absorbDir_device(gaugeWL, Zdir);
    for(int ti = 0; ti < t; ti++){
      su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
      su3_2->shift(*su3_in, 4 + Tdir);
    }

    for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

      if(z != 0) {
        WL_2->wilsonLineUpdate(*su3_2, tmp, Zdir);
        su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
        WL_3->shift(*su3_in, Zdir);
      }
          
        WL_temp1->UxU(*WL_1, *WL_2);
        WL_temp2->UxUdag(*WL_temp1, *WL_3);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, Tdir);
      }

      WL->UxUdag(*WL_temp2, *WL_2);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, 4 + Tdir);
      }

      //WL->writeHDF5( ("WL.stout"+std::to_string(nStout)+".Zdir"+std::to_string(Zdir)+".TPlus.ZMinus.h5/T_"+std::to_string(t)+"Z_"+std::to_string(z)).c_str(), true );
      WLval = WL->sumRtraceU()/(HGC_totalVolume*3);
      if(getRankVerbosity()) {
        fprintf(p_file2,"%d,%d,%f\n",t,z,WLval);
      }

    }//loop over values of B, building "2"  
        
  }//loop over values of L, building "1" and "3"

  //set the 4 branches as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, Tdir);

  if(getRankVerbosity()) {
    p_file1 = fopen((twop_filename+".stout"+std::to_string(nStout)+".Tdir"+std::to_string(Tdir)+".Zdir"+std::to_string(Zdir)+".TMinus.ZPlus.dat").c_str(),"w");
    p_file2 = fopen((twop_filename+".stout"+std::to_string(nStout)+".Tdir"+std::to_string(Tdir)+".Zdir"+std::to_string(Zdir)+".TMinus.ZMinus.dat").c_str(),"w");
  }

  for(int t = 0; t <= Tmax; t++) {//loop over values of L, building "1" and "3"

    if(t != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, Tdir);
    }

    WL_2->setUnit( (std::vector<int>) {0,4,8});
    WL_3->copy(*WL_1);
    su3_2->absorbDir_device(gaugeWL, Zdir);
    for(int ti = 0; ti < t; ti++){
      su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
      su3_2->shift(*su3_in, Tdir);
    }

    for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

      if(z != 0) {
        WL_2->wilsonLineUpdate(*su3_2, tmp, 4 + Zdir);
        su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
        WL_3->shift(*su3_in, 4 + Zdir);
      }
          
        WL_temp1->UxU(*WL_1, *WL_2);
        WL_temp2->UxUdag(*WL_temp1, *WL_3);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, 4 + Tdir);
      }

      WL->UxUdag(*WL_temp2, *WL_2);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, Tdir);
      }

      //WL->writeHDF5( ("WL.stout"+std::to_string(nStout)+".Zdir"+std::to_string(Zdir)+".TMinus.ZPlus.h5/T_"+std::to_string(t)+"Z_"+std::to_string(z)).c_str(), true );
      WLval = WL->sumRtraceU()/(HGC_totalVolume*3);
      if(getRankVerbosity()) {
        fprintf(p_file1,"%d,%d,%f\n",t,z,WLval);  
      }

    }//loop over values of B, building "2"

    WL_2->setUnit( (std::vector<int>) {0,4,8});
    WL_3->copy(*WL_1);
    su3_2->absorbDir_device(gaugeWL, Zdir);
    for(int ti = 0; ti < t; ti++){
      su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
      su3_2->shift(*su3_in, Tdir);
    }

    for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

      if(z != 0) {
        WL_2->wilsonLineUpdate(*su3_2, tmp, Zdir);
        su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
        WL_3->shift(*su3_in, Zdir);
      }
          
        WL_temp1->UxU(*WL_1, *WL_2);
        WL_temp2->UxUdag(*WL_temp1, *WL_3);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, 4 + Tdir);
      }

      WL->UxUdag(*WL_temp2, *WL_2);

      for(int ti = 0; ti < t; ti++) {
        su3_exchange = su3_in; su3_in = WL_2; WL_2 = su3_exchange;
        WL_2->shift(*su3_in, Tdir);
      }

      //WL->writeHDF5( ("WL.stout"+std::to_string(nStout)+".Zdir"+std::to_string(Zdir)+".TMinus.ZMinus.h5/T_"+std::to_string(t)+"Z_"+std::to_string(z)).c_str(), true );
      WLval = WL->sumRtraceU()/(HGC_totalVolume*3);
      if(getRankVerbosity()) {
        fprintf(p_file2,"%d,%d,%f\n",t,z,WLval);
      }

    }//loop over values of B, building "2"  
        
  }//loop over values of L, building "1" and "3"

  finalize();
  return 0;
}