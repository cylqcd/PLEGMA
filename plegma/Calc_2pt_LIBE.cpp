#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};

#define sign(a) (((a)>=0) ? +1:-1)

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double des;
  HGC_options->set("dqed", "dqed used for LIBE", verbosity, des);
  double deltamu;
  HGC_options->set("deltamu", "deltamu used for SIB", verbosity, deltamu);
  std::string qedfile;
  HGC_options->set("qed-filename", "The path to the QED field", verbosity, qedfile);
  std::vector<double> mul;
  HGC_options->set("mul-factors", "List of factors for additional mu to run", verbosity, mul);
  double dks;
  HGC_options->set("dkappa", "dkappa used for LIBE", verbosity, dks);

  std::vector<double> mu_s;
  std::vector<double> mu_c;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int nsmearGauss_c = 0;
  bool run_ud = true;
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
    options.set("run-ud", "Whether to run or not light quark flavors", verbosity, run_ud);
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    //options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
		     };
 add_options(*HGC_options);
  //=========================================================================================================//

  initializePLEGMA();

  if (mu_s.size() > 1){
    PLEGMA_error("More than one m_s passed.\n");
  }
  if (mu_c.size() > 1){
    PLEGMA_error("More than one m_c passed.\n");
  }

  if(link_recon!=QUDA_RECONSTRUCT_NO or link_recon_sloppy!=QUDA_RECONSTRUCT_NO) {
    PLEGMA_error("QED requires QUDA_RECONSTRUCT_NO\n");
  }

  int nmus = mul.size();
  double kappa0 = kappa;
  double mass0 = mass;
  bool excludeHeavyOnly = false;

  std::vector<double> mus;
  for(int i=0; i<nmus; i++) {
    mus.push_back(mul[i]*mu);
  }

  std::string given_twop_filename = twop_filename;
  PLEGMA_Gauge<double> smearedGauge(BOTH);
  PLEGMA_GaugeU1<double> gaugeU1;

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
    
    PLEGMA_printf("\n ### Running setup for mu=%.4e ###\n\n",mu_ud);
    mu = mu_ud;
    TIME(QUDA_solver solver(mu,1));

    
    PLEGMA_Gauge3D<double> smearedGauge3D;
    std::vector<std::vector<std::shared_ptr<PLEGMA_Propagator<float>>>> propsUp(nmus);
    std::vector<std::vector<std::shared_ptr<PLEGMA_Propagator<float>>>> propsDown(nmus);
    std::vector<std::shared_ptr<PLEGMA_Propagator<float>>> propsStrange;
    std::vector<std::shared_ptr<PLEGMA_Propagator<float>>> propsCharm;   
    PLEGMA_Propagator<float> propSIB; 

    for(int nprop=0; nprop<3; nprop++) {
      for (int imu = 0; imu < nmus; imu++) {
        propsUp[imu].push_back(std::make_shared<PLEGMA_Propagator<float>>(run_ud ? BOTH : NONE));
        propsDown[imu].push_back(std::make_shared<PLEGMA_Propagator<float>>(run_ud ? BOTH : NONE));
      }
    }
    for(int nprop=0; nprop<4; nprop++) {
      if (mu_s.size() > 0) {
        propsStrange.push_back(std::make_shared<PLEGMA_Propagator<float>>(BOTH));
      }
      if (mu_c.size() > 0) {
        propsCharm.push_back(std::make_shared<PLEGMA_Propagator<float>>(BOTH));
      }
    }
    for(int isource = 0; isource < numSourcePositions; isource++){

    PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
      isource, sourcePositions[isource][0], sourcePositions[isource][1],
      sourcePositions[isource][2], sourcePositions[isource][3]);
      site& source = sourcePositions[isource];
      
    char *src_string;
    asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
    twop_filename = given_twop_filename + std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE) + src_string + ".h5";
    free(src_string);

    if(access( twop_filename.c_str(), F_OK ) != -1){
      PLEGMA_printf("\n ### Skipping source-position, %s already exists ###\n\n", twop_filename.c_str());
    continue;
    } 
    
    if(des!=0) {
      asprintf(&src_string, "%04d", isource);
      std::string U1_conf = qedfile + src_string;
      PLEGMA_printf("\n ### Going to read %s ###\n\n", U1_conf.c_str());
      gaugeU1.readFile(U1_conf, LIME_FORMAT);
      free(src_string);
          }
    
    smearedGauge3D.absorb(smearedGauge, source[DIM_T]);
    PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);

    for(int isgn=0; isgn<3; isgn++) {
      if(kappa0 == -1.0) {
        mass = mass0;
        } else {
        kappa = kappa0;
        }

      double phase = (isgn-1)*des;
      if(isgn!=1){
        PLEGMA_Gauge<double> gauge2;
        gauge2.copy(gauge);
        gaugeU1.calculatePlaq(phase);
        gauge2.qedPhase(gaugeU1, phase);
        gauge2.calculatePlaq();
        updateGaugeQuda(gauge2, true);
        plaqQuda();
        solver.UpdateSolver();
        //applyBoundaryConditions(gauge2,true);
      } else {
        updateGaugeQuda(gauge, true);
        plaqQuda();
        solver.UpdateSolver();
      }
    

    std::vector<std::thread> threads;

    // Function to compute the propagator for a given run_mu value.
    auto computePropagator = [&](PLEGMA_Propagator<float>& prop, const double run_mu, WHICHFLAVOR fl, int nSmear) {
          // Update the options and solver if the mass is changed.
          if(mu != run_mu) {
              // updateOptions(fl);
              mu = run_mu;
              solver.UpdateSolver();
          }
          for(int isc = 0; isc < 12; isc++){
              PLEGMA_Vector<double> vectorInOut;
              { // Smear the source vector.
                  PLEGMA_Vector3D<double> vector1, vector2;
                  vector1.pointSource(source, isc/3, isc%3, DEVICE);
                  TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                  vectorInOut.absorb(vector2, source[DIM_T]);
              }
              PLEGMA_printf("Going to invert %s for component %d\n",
                            fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
              TIME(solver.solve(vectorInOut, vectorInOut));
              { // Smear the solution and absorb it into the propagator.
                  PLEGMA_Vector<double> vectorAuxD;
                  PLEGMA_Vector<float> vectorAuxF;
                  TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                  vectorAuxF.copy(vectorAuxD);
                  prop.absorb(vectorAuxF, isc/3, isc%3);
              }
          }
          prop.rotateToPhysicalBase_device(run_mu / abs(run_mu));
          prop.applyBoundaries_device(source[DIM_T]);
      };

      for (int imu = 0; imu < nmus; imu++) {
        mu_ud = mus [imu];
        TIME(computePropagator(*propsDown[imu][isgn], -mu_ud, LIGHT, nsmearGauss));
      }
      TIME(computePropagator(*propsStrange[isgn], mu_s[0], STRANGE, nsmearGauss_s));

      phase = (isgn-1)*2*des;
      if(isgn!=1){
        PLEGMA_Gauge<double> gauge2;
        gauge2.copy(gauge);
        gaugeU1.calculatePlaq(phase);
        gauge2.qedPhase(gaugeU1, phase);
        gauge2.calculatePlaq();
        updateGaugeQuda(gauge2, true);
        plaqQuda();
        solver.UpdateSolver();
        //applyBoundaryConditions(gauge2,true);
      }
      for (int imu = 0; imu < nmus; imu++) {
        mu_ud = mus [imu];
        TIME(computePropagator(*propsUp[imu][isgn], mu_ud, LIGHT, nsmearGauss));
      } 
      TIME(computePropagator(*propsCharm[isgn], mu_c[0], CHARM, nsmearGauss_c));

    if(isgn==1){
      TIME(computePropagator(*propsStrange[3], mu_s[0] + deltamu*mu_s[0], STRANGE, nsmearGauss_s));
      TIME(computePropagator(*propsCharm[3], mu_c[0] + deltamu*mu_c[0], CHARM, nsmearGauss_c));
      for (int imu = 0; imu < nmus; imu++) {
        mu_ud = mus [imu];
        if (imu!=0){
          excludeHeavyOnly = true;
        }
        else if (imu==0)
        {
          excludeHeavyOnly = false;
        }

        TIME(corr.contractBaryonsUDSC(*propsUp[imu][1], *propsDown[imu][1],*propsStrange[1],  *propsCharm[1],false,false,false,false,excludeHeavyOnly ));
        char * group;
      
        asprintf(&group, "baryons/u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]", mu_ud , -1*mu_ud, mu_s[0] , mu_c[0]);
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));

        double delta = deltamu*mu_ud; 

      {
        TIME(computePropagator(propSIB, mu_ud + delta, LIGHT, nsmearGauss));
        TIME(corr.contractBaryonsUDSC(propSIB, *propsDown[imu][1],*propsStrange[1],  *propsCharm[1], true, false,false,false,excludeHeavyOnly));
        char * group;
      
        asprintf(&group, "baryons_SIB/mu_ud[%+1.3e]/upSIB_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud + deltamu*mu_ud, -1*mu_ud, mu_s[0] , mu_c[0]);
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));
      }
      {
        TIME(computePropagator(propSIB, -mu_ud - delta, LIGHT, nsmearGauss));
        TIME(corr.contractBaryonsUDSC( *propsUp[imu][1],propSIB,*propsStrange[1],  *propsCharm[1], false, true,false,false,excludeHeavyOnly));
        char * group;
      
        asprintf(&group, "baryons_SIB/mu_ud[%+1.3e]/downSIB_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud, -1*mu_ud - deltamu*mu_ud, mu_s[0] , mu_c[0]);
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));
        }
      {
        TIME(corr.contractBaryonsUDSC(*propsUp[imu][1], *propsDown[imu][1],*propsStrange[3],  *propsCharm[1], false, false,true,false,excludeHeavyOnly));
        char * group;
      
        asprintf(&group, "baryons_SIB/mu_ud[%+1.3e]/strangeSIB_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud, -1*mu_ud, mu_s[0] + deltamu*mu_s[0], mu_c[0]);
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));
      }
      {
        TIME(corr.contractBaryonsUDSC(*propsUp[imu][1], *propsDown[imu][1],  *propsStrange[1], *propsCharm[3] ,false,false, false, true,excludeHeavyOnly));
        char * group;
      
        asprintf(&group, "baryons_SIB/mu_ud[%+1.3e]/charmSIB_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud, -1*mu_ud, mu_s[0], mu_c[0] + deltamu*mu_c[0]);
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));

      }
     }
     if(kappa0 == -1.0) {
      mass = mass0+mass0*dks;
    } else {
      kappa = kappa0+kappa0*dks;
    }
    PLEGMA_printf("\n ### Updating solver ###\n\n");
    TIME(solver.UpdateSolver());  
    TIME(computePropagator(*propsStrange[3], mu_s[0], STRANGE, nsmearGauss_s));
    TIME(computePropagator(*propsCharm[3], mu_c[0] , CHARM, nsmearGauss_c));
     for (int imu = 0; imu < nmus; imu++) {
      mu_ud = mus [imu];
      if (imu!=0){
        excludeHeavyOnly = true;
      }
      else if (imu==0)
      {
        excludeHeavyOnly = false;
      }
      { // Critical mass diagrams
        { 
          TIME(computePropagator(propSIB, mu_ud , LIGHT, nsmearGauss));
          TIME(corr.contractBaryonsUDSC(propSIB, *propsDown[imu][1],*propsStrange[1],  *propsCharm[1], true, false,false,false,excludeHeavyOnly));
          char * group;
  
          asprintf(&group, "baryons_SIBc/mu_ud[%+1.3e]/upSIBc_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud , -1*mu_ud, mu_s[0] , mu_c[0]);
          corr.setGroups(group);
          free(group);
          THREAD(corr.writeFile(twop_filename, corr_file_format));
          }
          {
            TIME(computePropagator(propSIB, -mu_ud , LIGHT, nsmearGauss));
            TIME(corr.contractBaryonsUDSC( *propsUp[imu][1],propSIB,*propsStrange[1],  *propsCharm[1], false, true,false,false,excludeHeavyOnly));
            char * group;
          
            asprintf(&group, "baryons_SIBc/mu_ud[%+1.3e]/downSIBc_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud, -1*mu_ud , mu_s[0] , mu_c[0]);
            corr.setGroups(group);
            free(group);
            THREAD(corr.writeFile(twop_filename, corr_file_format));
            }
          {
          TIME(corr.contractBaryonsUDSC(*propsUp[imu][1], *propsDown[imu][1],*propsStrange[3],  *propsCharm[1], false, false,true,false,excludeHeavyOnly));
          char * group;
  
          asprintf(&group, "baryons_SIBc/mu_ud[%+1.3e]/strangeSIBc_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud, -1*mu_ud, mu_s[0], mu_c[0]);
          corr.setGroups(group);
          free(group);
          THREAD(corr.writeFile(twop_filename, corr_file_format));
          }
          {
          TIME(corr.contractBaryonsUDSC(*propsUp[imu][1], *propsDown[imu][1],  *propsStrange[1], *propsCharm[3] ,false,false, false, true,excludeHeavyOnly));
          char * group;
  
          asprintf(&group, "baryons_SIBc/mu_ud[%+1.3e]/charmSIBc_u[%+1.3e]d[%+1.3e]s[%+1.3e]c[%+1.3e]",mu_ud, mu_ud, -1*mu_ud, mu_s[0], mu_c[0]);
          corr.setGroups(group);
          free(group);
          THREAD(corr.writeFile(twop_filename, corr_file_format));
  
          }
         } 
     }
    }
   } 
   
  
  PLEGMA_printf("Computing QED diagrams for source-position %d - %02d.%02d.%02d.%02d\n",
  isource, source[0], source[1], source[2], source[3]);

  auto contractAndWrite = [&](PLEGMA_Propagator<float>& propUP,
                              PLEGMA_Propagator<float>& propDN,
                              PLEGMA_Propagator<float>& propST,
                              PLEGMA_Propagator<float>& propCH,
                              bool only_up, bool only_dn,
                              bool only_st, bool only_ch,
                              int up_q, int dn_q, int st_q, int ch_q, bool excludeHeavyOnly,
                              double mud) {

    // collect only the non‐zero charges
    std::vector<std::pair<char,int>> flav;
    if (up_q != 0) flav.emplace_back('u', up_q);
    if (dn_q != 0) flav.emplace_back('d', dn_q);
    if (st_q != 0) flav.emplace_back('s', st_q);
    if (ch_q != 0) flav.emplace_back('c', ch_q);

    // build “top” = concatenation of flavors in order
    std::string top;
    top.reserve(flav.size());
    for (auto &p : flav) top.push_back(p.first);

    // build suffix = e.g. “u+1d-1…”
    std::string suffix;
    for (auto &p : flav) {
    suffix += p.first;
    if (p.second > 0) suffix += '+';
    suffix += std::to_string(p.second);
    }

    char mud_str[64];
    snprintf(mud_str, sizeof(mud_str), "mu_ud[%+1.3e]/", mud);

    std::string group = "baryons_QED/";
    group += std::string(mud_str);
    if (top.empty()) {
    group += "none";
    } else {
    group += top + "/" + suffix;
    }

    TIME(corr.contractBaryonsUDSC(
    propUP, propDN, propST, propCH,
    only_up, only_dn, only_st, only_ch, excludeHeavyOnly));
    corr.setGroups(group.c_str());
    THREAD(corr.writeFile(twop_filename, corr_file_format));
  };


    
  int idx2[2]    = { 0, 2 };       
  int charges[2] = { -1, +1 }; 

  for (int imu = 0; imu < nmus; imu++) {
    if (imu!=0){
      excludeHeavyOnly = true;
    }
    else if (imu==0)
    {
      excludeHeavyOnly = false;
    }

    for (int i= 0; i< 2; i++) {
      contractAndWrite(*propsUp[imu][idx2[i]],*propsDown[imu][1],*propsStrange[1],*propsCharm[1],true,false,false,false,charges[i],0,0,0,excludeHeavyOnly,mus[imu]); //u 
      contractAndWrite(*propsUp[imu][1],*propsDown[imu][idx2[i]],*propsStrange[1],*propsCharm[1],false,true,false,false,0,charges[i],0,0,excludeHeavyOnly,mus[imu]); //d
      contractAndWrite(*propsUp[imu][1],*propsDown[imu][1],*propsStrange[idx2[i]],*propsCharm[1],false,false,true,false,0,0,charges[i],0,excludeHeavyOnly,mus[imu]); //s
      contractAndWrite(*propsUp[imu][1],*propsDown[imu][1],*propsStrange[1],*propsCharm[idx2[i]],false,false,false,true,0,0,0,charges[i],excludeHeavyOnly,mus[imu]); //c
    }


    for (int i=0; i<2; i++){
      for(int j=0; j<2; j++){

        contractAndWrite(*propsUp[imu][idx2[i]],*propsDown[imu][idx2[j]],*propsStrange[1],*propsCharm[1],true,true,false,false,charges[i],charges[j],0,0,excludeHeavyOnly,mus[imu]); //ud
        contractAndWrite(*propsUp[imu][idx2[i]],*propsDown[imu][1],*propsStrange[idx2[j]],*propsCharm[1],true,false,true,false,charges[i],0,charges[j],0,excludeHeavyOnly,mus[imu]); //us
        contractAndWrite(*propsUp[imu][idx2[i]],*propsDown[imu][1],*propsStrange[1],*propsCharm[idx2[j]],true,false,false,true,charges[i],0,0,charges[j],excludeHeavyOnly,mus[imu]); //uc

        contractAndWrite(*propsUp[imu][1],*propsDown[imu][idx2[i]],*propsStrange[idx2[j]],*propsCharm[1],false,true,true,false,0,charges[i],charges[j],0,excludeHeavyOnly,mus[imu]); //ds
        contractAndWrite(*propsUp[imu][1],*propsDown[imu][idx2[i]],*propsStrange[1],*propsCharm[idx2[j]],false,true,false,true,0,charges[i],0,charges[j],excludeHeavyOnly,mus[imu]); //dc  
        

        contractAndWrite(*propsUp[0][1],*propsDown[0][1],*propsStrange[idx2[i]],*propsCharm[idx2[j]],false,false,true,true,0,0,charges[i],charges[j],excludeHeavyOnly,mus[imu]); //sc
      }
    }
   } 
}
//finalize();
return 0;
}
