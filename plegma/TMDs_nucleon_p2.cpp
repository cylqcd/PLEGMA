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

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge",
               "nsrc", "src-filename", "maxQsq","threep-filename",  "corr-file-format",
               "corr-space", "tSinks","gammas"};

  initializeOptions(argc, argv, false, listOpt);


  if(mu<0){
    mu *= -1.0;
  }

  size_t Ldir = 2;
  HGC_options->set("Ldir", "Direction of longitudinal part of staple", verbosity, Ldir);

  size_t Lmax = 0;
  HGC_options->set("Lmax", "Maximum length of the symmetric part of staple", verbosity, Lmax);

  size_t Bmax = 0;
  HGC_options->set("Bmax", "Maximum length of the transverse part of staple", verbosity, Bmax);

  size_t Zmax = 0;
  HGC_options->set("Zmax", "Maximum length of the asymmetric part of staple", verbosity, Zmax);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t nStout;
  HGC_options->set("nstout", "Number of stout smearing steps", verbosity, nStout);
  
  std::string proj;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::vector<int> Lvals;
  HGC_options->set("Lvals", "values of L for which to calculate the staple", verbosity, Lvals);

  std::string which_particle = "proton";
  HGC_options->set("which_particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, which_particle);
  WHICHPARTICLE nucleon = get_particle(which_particle.c_str());
  
  initializePLEGMA();



  //set the 2 transverse directions given Ldir
  size_t Bdir1, Bdir2;
  switch(Ldir) {
    case 0:
      Bdir1 = 1;
      Bdir2 = 2;
      break;
    case 1:
      Bdir1 = 0;
      Bdir2 = 2;
      break;
    case 2:
      Bdir1 = 0;
      Bdir2 = 1;
      break;
    default:
      PLEGMA_error("The direction of the WIlson line has to be smaller than 3");
  }
  
  if(Lmax == 0) Lmax = HGC_totalL[Ldir]/2;
  if(Bmax == 0) Bmax = HGC_totalL[Bdir1]/2;
  if(Zmax == 0) Zmax = HGC_totalL[Ldir]/2;

  char *fname;
  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> *gauge = new PLEGMA_Gauge<double>;
  gauge->readFile(latfile, LIME_FORMAT);
  gauge->calculatePlaq();

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(*gauge);

  //apply the stout smearing steps
  if(nStout > 0) gaugeWL.stoutSmearing(gaugeWL, nStout, rhoStout, 3);

  delete gauge;

  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sinkMom.begin(),sinkMom.begin()+3,sinkMom_3D.begin());

  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF2 = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);

  std::string suff11, suff12, suff21, suff22;

  for(int isource=0;isource<numSourcePositions;isource++){//loop over source positions
  
  asprintf(&fname, "propUP.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3]);
  propUP->readFile(fname,LIME_FORMAT);
  asprintf(&fname, "propDN.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3]);
  propDN->readFile(fname,LIME_FORMAT);

  for(int ts=0;ts<tSinks.size();ts++) {//loop over tSinks

    PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[isource], 0, (tSinks[ts]+1));

    int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
    int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
    int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
    bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

    int signProps1 = (nucleon == PROTON) ? -1: +1;

    int signProps2 = (nucleon == PROTON) ? +1: -1;

    asprintf(&fname, "seqPropOut.CP1.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d.tS%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3],tSinks[ts]);
    seqPropOut1->readFile(fname,LIME_FORMAT);
    asprintf(&fname, "seqPropOut.CP2.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d.tS%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3],tSinks[ts]);
    seqPropOut2->readFile(fname,LIME_FORMAT);
      

      /**********************************************************************
       * now we build the staples
       * idea is to build in 4 branches
       * 
       *                     _______2_______
       *                    +               +
       *                    |               |
       *                    |               |
       *                    |               |
       *                    1               3
       *                    |               |
       *                    |               |
       *                    |               |
       *                                    +
       *                                    |
       *                                    |
       *                                    4
       *                                    |
       *                                    |
       * 
       * also do the staple by taking "2" in two directions Bdir1 and Bdir2
       * this means we calculate "1" only once
       * also do it step wise in "2" which is the B direction
       * so build "1", "2" and "3", then build "4" stepwise for fixed "1", "2" and "3"
       * then shift "3" one step in B direction and update "2"
       * do this untill you reach Bmax
       * then shift "2" one step in Ldir and update both "1" and "3"
       * now follow the same procedure but in the opposite direction for B for "2", that is you start with Bmax and go to zero
       * do similar thing for "4" as well by going in opposite direction, start with Zmax and go to zero
       * keep doing this, alternating the direction of B and Z, untill you reach Lmax


      ***********************************************************************/

    //set the 4 branches as unit
    WL_1->setUnit( (std::vector<int>) {0,4,8});
    
    suff11 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Plus_";
    suff12 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Plus_";

    suff21 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Plus_";
    suff22 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Plus_";

    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, Ldir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);
    
    if(nucleon == PROTON) propF1->copy(*propDN);
    else propF1->copy(*propUP);
    
    if(nucleon == PROTON) propF2->copy(*propUP);
    else propF2->copy(*propDN);

    suff11 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Plus_";
    suff12 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Plus_";

    suff21 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Plus_";
    suff22 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Plus_";

    for(int l = 0; l <= Lmax; l++) {//loop over values of L, building "1" and "3"

      if(l != 0) {

        WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + Ldir);

        su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
        su3_2_1->shift(*su3_in, 4 + Ldir);

        su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
        su3_2_2->shift(*su3_in, 4 + Ldir);

      }

      if(std::count(Lvals.begin(), Lvals.end(), l)) {//calculate only for certain l's
        WL_2_1->copy(*WL_1);
        WL_2_2->copy(*WL_1);
        su3_2_1->unload();
        su3_2_2->unload();
        for(int b = 0; b <= Bmax; b++) {//loop over values of B, building "2"

          if(b != 0) {

            WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);

            WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);

          }


          for(int bi = 0; bi < b; bi++) {
            su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
            WL_1->shift(*su3_in, 4 + Bdir1);
          }

          WL_4_1->UxUdag(*WL_2_1, *WL_1);
          for(int bi = 0; bi < b; bi++) {
            su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
            WL_1->shift(*su3_in, Bdir1);
          }

          for(int bi = 0; bi < b; bi++) {
            su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
            WL_1->shift(*su3_in, 4 + Bdir2);
          }
 
          WL_4_2->UxUdag(*WL_2_2, *WL_1);
          for(int bi = 0; bi < b; bi++) {
            su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
            WL_1->shift(*su3_in, Bdir2);
          }

          su3_4_1->absorbDir_device(gaugeWL, Ldir);
          su3_4_2->absorbDir_device(gaugeWL, Ldir);
          for(int bi = 0; bi < b; bi++) {
            su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
            su3_4_1->shift(*su3_in, 4 + Bdir1);
            su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
            su3_4_2->shift(*su3_in, 4 + Bdir2);
          }

          if((b != 0) || (l == Lvals[0])) {//build b=0 only once
          for(int z = 0; z <= Zmax; z++) {//loop over values of Z, building "4"

            if(z != 0) {

              WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, Ldir);

              WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, Ldir);

              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, Ldir);

              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, Ldir);

            }


            //uncomment to write the staples, for testing
            //WL1->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );
            //WL2->writeHDF5( (twop_filename + suff12 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

            /* if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, 4 + Bdir1);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL_4_1, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff11 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, Bdir1);
            }

            /* if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, 4 + Bdir2);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL_4_2, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff12 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, Bdir2);
            }

            /* if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, 4 + Bdir1);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL_4_1, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff21 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, Bdir1);
            }

            /* if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, 4 + Bdir2);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL_4_2, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff22 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, Bdir2);
            }

            

          }//loop over values of Z, building "4"
          }//build b = 0 only once

          

        }//loop over values of B, building "2"
        su3_2_1->load();
        su3_2_2->load();
        for(int zi = 0; zi < Zmax; zi++) {
          propExchange = propIn; propIn = propF1; propF1 = propExchange;
          propF1->shift(*propIn, 4 + Ldir);
          propExchange = propIn; propIn = propF2; propF2 = propExchange;
          propF2->shift(*propIn, 4 + Ldir);
        }
        //WL_1->load();
      }//calculate only for certain l's
        
    }//loop over values of L, building "1" and "3"


      //now repeat the same thing only going in negative boost direction for the staple
      //set the 4 branches as unit
    WL_1->setUnit( (std::vector<int>) {0,4,8});

      //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, Ldir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);

    if(nucleon == PROTON) propF1->copy(*propDN);
    else propF1->copy(*propUP);
    
    if(nucleon == PROTON) propF2->copy(*propUP);
    else propF2->copy(*propDN);

    suff11 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Minus_";
    suff12 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Minus_";

    suff21 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Minus_";
    suff22 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Minus_";

    for(int l = 0; l <= Lmax; l++) {//loop over values of L, building "1" and "3"

      if(l != 0) {

        WL_1->wilsonLineUpdate(*su3_1, tmp, Ldir);

        su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
        su3_2_1->shift(*su3_in, Ldir);

        su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
        su3_2_2->shift(*su3_in, Ldir);

      }

      if(std::count(Lvals.begin(), Lvals.end(), l)) {//calculate only for certain l's
        WL_2_1->copy(*WL_1);
        WL_2_2->copy(*WL_1);
        su3_2_1->unload();
        su3_2_2->unload();
      for(int b = 0; b <= Bmax; b++) {//loop over values of B, building "2"

        if(b != 0) {

          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);

          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);

        }
          
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir1);
        }

        WL_4_1->UxUdag(*WL_2_1, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir1);
        }

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir2);
        }

        WL_4_2->UxUdag(*WL_2_2, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir2);
        }

        su3_4_1->absorbDir_device(gaugeWL, Ldir);
        su3_4_2->absorbDir_device(gaugeWL, Ldir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + Bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, 4 + Bdir2);
        }

        if((b != 0) || (l == Lvals[0])) {//build b=0 only once
          for(int z = 0; z <= Zmax; z++) {//loop over values of Z, building "4"

            if(z != 0) {

              WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4 + Ldir);

              WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4 + Ldir);

              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, 4 + Ldir);

              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, 4 + Ldir);

            }

            //uncomment to write the staples, for testing
            //WL1->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );
            //WL2->writeHDF5( (twop_filename + suff12 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

            /* if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, 4 + Bdir1);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL_4_1, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff11 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, Bdir1);
            }

            /* if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, 4 + Bdir2);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL_4_2, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff12 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF1; propF1 = propExchange;
              propF1->shift(*propIn, Bdir2);
            }

            /* if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, 4 + Bdir1);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL_4_1, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff21 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, Bdir1);
            }

            /* if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN); */
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, 4 + Bdir2);
            }
            /* for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Ldir);
            } */

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL_4_2, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff22 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF2; propF2 = propExchange;
              propF2->shift(*propIn, Bdir2);
            }

            

          }//loop over values of Z, building "4"
        }//build b = 0 only once
          
      }//loop over values of B, building "2"
      su3_2_1->load();
      su3_2_2->load();
      for(int zi = 0; zi < Zmax; zi++) {
        propExchange = propIn; propIn = propF1; propF1 = propExchange;
        propF1->shift(*propIn, Ldir);
        propExchange = propIn; propIn = propF2; propF2 = propExchange;
        propF2->shift(*propIn, Ldir);
      }
    }//calculate only for certain l's
        
    }//loop over values of L, building "1" and "3"

      asprintf(&fname, "seqPropOut.CP1.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d.tS%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3],tSinks[ts]);
      std::remove(fname);
      asprintf(&fname, "seqPropOut.CP2.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d.tS%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3],tSinks[ts]);
      std::remove(fname);


  }//loop over tSinks

  asprintf(&fname, "propUP.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3]);
  std::remove(fname);
  asprintf(&fname, "propDN.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3]);
  std::remove(fname);
  
  }//loop over source positions

  //delete solver;

  finalize();
  return 0;

}
