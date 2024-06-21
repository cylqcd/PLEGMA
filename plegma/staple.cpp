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

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge"};

  initializeOptions(argc, argv, false, listOpt);

  size_t Ldir = 2;
  HGC_options->set("Ldir", "Direction of longitudinal part of staple", verbosity, Ldir);

  size_t Lmax = 0;
  HGC_options->set("Lmax", "Maximum length of the symmetric part of staple", verbosity, Lmax);

  size_t Bdir = 1;
  HGC_options->set("Bdir", "Direction of transverse part of staple", verbosity, Bdir);

  size_t Bmax = 0;
  HGC_options->set("Bmax", "Maximum length of the transverse part of staple", verbosity, Bmax);

  size_t Zmax = 0;
  HGC_options->set("Zmax", "Maximum length of the asymmetric part of staple", verbosity, Zmax);

  initializePLEGMA();

  if(Lmax == 0) Lmax = HGC_totalL[Ldir]/2;
  if(Bmax == 0) Bmax = HGC_totalL[Bdir]/2;
  if(Zmax == 0) Zmax = HGC_totalL[Ldir]/2;

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  PLEGMA_status();
  gauge.calculatePlaq();

#if 1
  //copy gauge to gaugeWL for building of the staples
  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

  auto computeStaple = [&](size_t ldir, size_t lmax, size_t bdir, size_t bmax, size_t zmax) {
    PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> tmp(BOTH);
    PLEGMA_Su3field<float> *u_s[4];

    for(int idir = 0; idir < 4 ; idir++){
      u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
      u_s[idir]->absorbDir_device(gaugeWL,idir);
    }

    for(int z_step=0 ; z_step <= zmax ; z_step++){
      for(int b=0 ; b <= bmax ; b++){
        for(int l=0 ; l <= lmax ; l++){

          WL->setUnit((std::vector<int>) {0,4,8});
          int spath[(2*l)+b+z_step];

          if((l!=0)||(b!=0)||(z_step!=0)){
            for(int i=0;i<l;i++){
              spath[i] = ldir;
            }
            for(int j=l;j<(l+b);j++){
              spath[j] = bdir;
            }
            for(int k=(l+b);k<((2*l)+b+z_step);k++){
              spath[k] = 4+ldir;
            }

            std::vector<int> vspath(spath,spath+(2*l)+b+z_step);

            WL->path(vspath, u_s, tmp);
          }
        }
      }
    }
  };

  auto computeStapleV1 = [&](size_t ldir, size_t lmax, size_t bdir, size_t bmax, size_t zmax) {
    PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> tmp(BOTH);

    //set the 4 branches as unit
    WL_1->setUnit( (std::vector<int>) {0,4,8});

    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, ldir);
    su3_2_1->absorbDir_device(gaugeWL, bdir);

    for(int l = 0; l <= lmax; l++) {//loop over values of L, building "1" and "3"
      if(l != 0) {
        WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + ldir);

        su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
        su3_2_1->shift(*su3_in, 4 + ldir);
      }
      WL_2_1->copy(*WL_1);
      su3_2_1->unload();
      for(int b = 0; b <= Bmax; b++) {//loop over values of B, building "2"

        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + bdir);
        }


        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + bdir);
        }

        WL_4_1->UxUdag(*WL_2_1, *WL_1);

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, bdir);
        }

        su3_4_1->absorbDir_device(gaugeWL, ldir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + bdir);
        }

        for(int z = 0; z <= zmax; z++) {//loop over values of Z, building "4"

          if(z != 0) {
            WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, ldir);
          }  

        }//loop over values of Z, building "4"

          

      }//loop over values of B, building "2"
      su3_2_1->load();
        
    }//loop over values of L, building "1" and "3"
  };

  auto computeStapleV2 = [&](size_t ldir, size_t lmax, size_t bdir1, size_t bdir2, size_t bmax, size_t zmax) {
    PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_2_2 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_4_2 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_2_2 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *WL_4_2 = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
    PLEGMA_Su3field<float> tmp(BOTH);

    //set the 4 branches as unit
    WL_1->setUnit( (std::vector<int>) {0,4,8});

    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, ldir);
    su3_2_1->absorbDir_device(gaugeWL, bdir1);
    su3_2_2->absorbDir_device(gaugeWL, bdir2);

    for(int l = 0; l <= lmax; l++) {//loop over values of L, building "1" and "3"
      if(l != 0) {
        WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + ldir);

        su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
        su3_2_1->shift(*su3_in, 4 + ldir);

        su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
        su3_2_2->shift(*su3_in, 4 + ldir);
      }
      WL_2_1->copy(*WL_1);
      su3_2_1->unload();
      WL_2_2->copy(*WL_1);
      su3_2_2->unload();
      for(int b = 0; b <= Bmax; b++) {//loop over values of B, building "2"

        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + bdir2);
        }


        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + bdir1);
        }

        WL_4_1->UxUdag(*WL_2_1, *WL_1);

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, bdir1);
        }

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + bdir2);
        }

        WL_4_2->UxUdag(*WL_2_2, *WL_1);

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, bdir2);
        }

        su3_4_1->absorbDir_device(gaugeWL, ldir);
        su3_4_2->absorbDir_device(gaugeWL, ldir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, 4 + bdir2);
        }

        for(int z = 0; z <= zmax; z++) {//loop over values of Z, building "4"

          if(z != 0) {
            WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, ldir);
            WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, ldir);
          }  

        }//loop over values of Z, building "4"

          

      }//loop over values of B, building "2"
      su3_2_1->load();
      su3_2_2->load();
        
    }//loop over values of L, building "1" and "3"
  };

  //TIME(computeStaple(Ldir, Lmax, Bdir, Bmax, Zmax));


  //TIME(computeStapleV1(Ldir, Lmax, Bdir, Bmax, Zmax));

  //size_t Bdir2 = 3 - (Ldir + Bdir);

  //TIME(computeStapleV2(Ldir, Lmax, Bdir, Bdir2, Bmax, Zmax));
#endif

  finalize();
  return 0;
}