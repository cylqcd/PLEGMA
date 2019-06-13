#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-stout", "alpha-stout","corr-file-format","maxQsq"};

static void dumpLoops(PLEGMA_FT<double> **ft,
 		      std::string filenamePrefix, std::string confID, FILE_WRITE_FORMAT format){
  for(int idir=0; idir < 3; idir++)
    for(int i =0; i < HGC_totalL[0]; i++)
      ft[idir*HGC_totalL[0]+i]->writeToFile(filenamePrefix + "_dir" + std::to_string(idir) + "_z" + std::to_string(i) + "_" + confID + ".dat" ,format);
}

int main(int argc, char **argv){
  initializeOptions(argc, argv, false, listOpt);
  //===================================//
  std::string filesPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, filesPrefix);
  //===================================//
  initializePLEGMA();
  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  PLEGMA_printf("Unsmeared Plaquette is:");
  gauge.calculatePlaq();


  PLEGMA_Gauge<double> smearedGaugeWL;
  smearedGaugeWL.stoutSmearing(gauge, nsmearStout, alphaStout, 3);
  PLEGMA_printf("Smeared Plaquette with stout 3D:");
  smearedGaugeWL.calculatePlaq();

  PLEGMA_Gauge<double> smearedGaugeFmunu;
  smearedGaugeFmunu.stoutSmearing(gauge, nsmearStout, alphaStout, 4);
  PLEGMA_printf("Smeared Plaquette with stout 4D:");
  smearedGaugeFmunu.calculatePlaq();

  PLEGMA_Field<double> traceO1(BOTH,SCALAR);
  PLEGMA_Field<double> traceO2(BOTH,SCALAR);
  PLEGMA_Su3field<double> su3;
  PLEGMA_Su3field<double> WL;
  PLEGMA_Su3field<double> tmp;

  PLEGMA_Fmunu<double> fmunu_l;
  PLEGMA_Fmunu<double> fmunu_r;
  PLEGMA_Fmunu<double> *fmunuExchange = nullptr;
  PLEGMA_Fmunu<double> *fmunu_ptr = nullptr;
  PLEGMA_Fmunu<double> *fmunu_In = new PLEGMA_Fmunu<double>(DEVICE);
  
  fmunu_l.compute_leaves(smearedGaugeFmunu);
  fmunu_r.copy(fmunu_l);

  if(!(HGC_totalL[0] == HGC_totalL[1] && HGC_totalL[1] == HGC_totalL[2])) PLEGMA_error("Spatial total volume should be symmetric for this to work");
  int L=HGC_totalL[0];
  if(L%2 != 0) PLEGMA_error("If spatial extent is not multiple of 2 then it will not work");
  int Lo2 = L/2;

  int sizeFT=3*HGC_totalL[0];
  PLEGMA_FT<double> **FTs = new PLEGMA_FT<double>*[sizeFT];
  for(int i = 0 ; i< sizeFT; i++) FTs[i] = new PLEGMA_FT<double>(maxQsq,3);
  
  for(int wilsDir = 0 ; wilsDir < 3; wilsDir++){
    su3.absorbDir_device(smearedGaugeWL,wilsDir);
    WL.setUnit((std::vector<int>) {0,4,8});
    fmunu_ptr = &fmunu_r;
    for(int i = 0 ; i < Lo2;i++){
      traceO1.fmunuSu3Fmunu(fmunu_l,std::make_pair((wilsDir+1)%3,3),WL,*fmunu_ptr,std::make_pair((wilsDir+1)%3,3));
      traceO2.fmunuSu3Fmunu(fmunu_l,std::make_pair((wilsDir+2)%3,3),WL,*fmunu_ptr,std::make_pair((wilsDir+2)%3,3));
      traceO1.axpy(traceO2,(std::complex<double>) {1.,0.});
      traceO1.cscale((std::complex<double>) {0.5,0.}); //average the two contributions
      if(FTs[wilsDir*L+i]->IsAccum()) PLEGMA_error("We need accumulation off here");
      FTs[wilsDir*L+i]->apply(traceO1,FT_GEMV);
      fmunuExchange = fmunu_In; fmunu_In = fmunu_ptr; fmunu_ptr = fmunuExchange;
      WL.wilsonLineUpdate(su3,tmp,4+wilsDir);
      fmunu_ptr->shift(*fmunu_In,4+wilsDir);
    }

    su3.absorbDir_device(smearedGaugeWL,wilsDir);
    WL.setUnit((std::vector<int>) {0,4,8});
    fmunu_ptr->copy(fmunu_l);
    for(int i = 0 ; i < Lo2;i++){
      traceO1.fmunuSu3Fmunu(fmunu_l,std::make_pair((wilsDir+1)%3,3),WL,*fmunu_ptr,std::make_pair((wilsDir+1)%3,3));
      traceO2.fmunuSu3Fmunu(fmunu_l,std::make_pair((wilsDir+2)%3,3),WL,*fmunu_ptr,std::make_pair((wilsDir+2)%3,3));
      traceO1.axpy(traceO2,(std::complex<double>) {1.,0.});
      traceO1.cscale((std::complex<double>) {0.5,0.}); //average the two contributions
      if(FTs[wilsDir*L+i+Lo2]->IsAccum()) PLEGMA_error("We need accumulation off here");
      FTs[wilsDir*L+i+Lo2]->apply(traceO1,FT_GEMV);
      fmunuExchange = fmunu_In; fmunu_In = fmunu_ptr; fmunu_ptr = fmunuExchange;
      WL.wilsonLineUpdate(su3,tmp,wilsDir);
      fmunu_ptr->shift(*fmunu_In,wilsDir);
    }
    fmunu_ptr->copy(fmunu_l);
  }

  std::size_t foundPos = latfile.find("conf.");
  if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
  std::string confID = latfile.substr(foundPos+5,latfile.length());

  dumpLoops(FTs,filesPrefix + "/gLoops_OperType1",confID,corr_file_format);
  
  delete fmunu_In;
  for(int i = 0 ; i< sizeFT; i++) delete FTs[i];
  delete[] FTs;
  
  finalize();
}
