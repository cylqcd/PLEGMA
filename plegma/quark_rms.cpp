#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, false, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::vector<int> nsmearGaussList;
  for(int i = 10; i<= 200; i+=10) nsmearGaussList.push_back(i);
  std::vector<double> alphaGaussList;
  for(double a = 0.1; a<=4.; a+=0.1) alphaGaussList.push_back(a);
  int src[] = {0,0,0,0};
  std::string outPrefix = "./output";
  HGC_options->set("src-xyzt", "Source position in order (x,y,z,t)",verbosity,src[0],src[1],src[2],src[3]);
  HGC_options->set("list-nsmear-gauss", "List for the number of the Gaussian smearing steps", verbosity, nsmearGaussList);
  HGC_options->set("list-alpha-gaussian", "List of the alpha for Gaussian smearing", verbosity, alphaGaussList);
  HGC_options->set("outPrefix", "Path to the prefix output file", verbosity, outPrefix);
  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge;

  // Reading from Lime file and loading to device
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette using APE is:");
  smearedGauge.calculatePlaq();

  std::vector<int> list_R2;
  std::vector<int> counter = createR2(list_R2);
  PLEGMA_Vector<double> v1,v2;

  //  smearedGauge.communicateSideGhost();
  
  for(auto alpha : alphaGaussList){
    v1.pointSource(src,0,0,DEVICE);
    for(int n = 0; n < *std::max_element(nsmearGaussList.begin(), nsmearGaussList.end()); n++ ){
      if(n%2 == 0) v2.gaussianSmearing(v1,smearedGauge,1, alpha);
      else v1.gaussianSmearing(v2,smearedGauge,1, alpha);
      
      if(std::find(nsmearGaussList.begin(), nsmearGaussList.end(),n+1) != nsmearGaussList.end()){
	PLEGMA_printf("%d %f\n",n+1, alpha);
	std::vector<double> rms = (n%2 == 0) ? v2.rms(list_R2,src) : v1.rms(list_R2,src);	  
	std::string filename = outPrefix + "_nAPE" + std::to_string(nsmearAPE) + "_aAPE" + convNumToStr(alphaAPE) + "_nGau" + std::to_string(n+1)
	  + "aGau" + convNumToStr(alpha);
	if(comm_rank() == 0) write_std_vecs( filename,list_R2, counter,rms);
      }
    }
  }
  
  
  finalize();
}
