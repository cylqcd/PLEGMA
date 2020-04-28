#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE","load-list-vectors",
					   "nsrc","src-filename"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::string invertConv = "QUDA";
  HGC_options->set("invert-convention", "With what package gammas convention to invert, options (QUDA, tmLQCD)", verbosity, invertConv);
  std::string boundaryCond = "antiperiodic";
  HGC_options->set("boundary-condition", "If we want periodic or antiperiodic in the temporal direction", verbosity, boundaryCond);
  std::string sourceType = "custom";
  HGC_options->set("source-type", "Source types allowed: i) a vector full of unities called (unitSource). ii) point sources read from src-list called (pointSource).iii) momentum sources read from src-list called (momSource). iv) custom made sources read from list of vectors (customSource)", verbosity, sourceType);
  std::string pathOut = "./";
  HGC_options->set("output-path","Path to the directory to dump inverted files",verbosity,pathOut);
  //==========================//
  initializePLEGMA();
  if(invertConv != "QUDA" && invertConv != "tmLQCD") PLEGMA_error("Unknown invert conventions: %s\n",invertConv.c_str());
  if(boundaryCond != "periodic" && boundaryCond != "antiperiodic") PLEGMA_error("Unknown boundary condition: %s\n",invertConv.c_str());
  if(sourceType != "unitSource" && sourceType != "pointSource" && sourceType != "momSource" && sourceType != "customSource") PLEGMA_error("Unknown source type: %s", sourceType.c_str());

  int Nsc=12;
  if(sourceType == "unitSource") {numSourcePositions = 1; Nsc=1;}
  if(sourceType == "customSource") {numSourcePositions = listVecs.size(); Nsc=1;}

  {
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();
    initGaugeQuda(gauge, boundaryCond == "antiperiodic");
    plaqQuda();
    std::string confStr=splitStrFwd(latfile,'.');
  
    if(invertConv == "tmLQCD") mu = -mu;
    QUDA_solver solver(mu);
    PLEGMA_Vector<double> vIn,vOut;

    std::string outName;
    for(int i = 0; i < numSourcePositions; i++){
      for(int isc =0; isc < Nsc; isc++){
	if(sourceType == "unitSource"){
	  vIn.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
	  outName = pathOut + sourceType + "." + confStr + ".inverted";
	}
	else if(sourceType == "pointSource"){
	  site& sS = sourcePositions[i];
	  vIn.pointSource(sS,isc/3,isc%3,DEVICE);
	  std::string xyzt = "xx" + std::to_string(sS[0]) + "yy" + std::to_string(sS[1]) +
	    "zz" + std::to_string(sS[2]) + "tt" + std::to_string(sS[3]);
	  outName = pathOut + sourceType + "_" + xyzt + "." + confStr + "." + std::to_string(isc) + ".inverted"; 
	}
	else if(sourceType == "momSource") {
	  std::vector<int> scInd = {isc};
	  vIn.setUnit(scInd);
	  site& sS = sourcePositions[i];
	  vIn.mulMomentumPhases((std::vector<int>) {sS[0],sS[1],sS[2],sS[3]},+1);
	  std::string pxpypzpt = "px" + std::to_string(sS[0]) + "py" + std::to_string(sS[1]) +
	    "pz" + std::to_string(sS[2]) + "pt" + std::to_string(sS[3]);
	  outName = pathOut + sourceType + "_" + pxpypzpt + "." + confStr + "." + std::to_string(isc) + ".inverted"; 	
	}
	else{
	  vIn.readLIME(listVecs[i]);
	  outName = listVecs[i] + ".inverted";
	}

	if(invertConv == "tmLQCD"){
	  vIn.rotate_uk_ch();
	  if(boundaryCond == "antiperiodic") vIn.mulThetaPhase(1.,true);
	}
	solver.solve(vOut,vIn);
      
	if(invertConv == "tmLQCD"){
	  if(boundaryCond == "antiperiodic") vOut.mulThetaPhase(1.,false);
	  vOut.rotate_uk_ch();
	}

	vOut.writeLIME(outName);
      }
    }
  }
  
  finalize();
  return 0;
}
