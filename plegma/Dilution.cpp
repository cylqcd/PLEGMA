#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, false); // Add list of options
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();
  

  PLEGMA_Vector<double> vectorAuxD(DEVICE);
  PLEGMA_Vector<double> vectorSOutD(HOST);
  PLEGMA_Vector<double> vectorCOutD(HOST);
  PLEGMA_Vector<double> vectorSCOutD(HOST);
  PLEGMA_Vector<double> vectorSDilD(BOTH);
  PLEGMA_Vector<double> vectorSDilD1(BOTH);
  PLEGMA_Vector<double> vectorSDilD2(BOTH);
  PLEGMA_Vector<double> vectorCDilD(BOTH);
  PLEGMA_Vector<double> vectorSCDilD(BOTH);
  std::complex<double> res;

  int nroots=2;
  vectorAuxD.randInit(1234);
  vectorAuxD.stochastic_Z(nroots);
  vectorSDilD.dilutespin(vectorAuxD, 1);
  vectorSDilD1.dilutespin(vectorAuxD, 1);
  vectorSDilD2.dilutespin(vectorAuxD, 2);
  vectorCDilD.dilutecolor(vectorAuxD, 2);
  vectorSCDilD.dilutespincolor(vectorAuxD, 2, 1);
  res = vectorSDilD1.dot(vectorSDilD2);
  vectorSDilD.unload();
  vectorCDilD.unload();
  vectorSCDilD.unload();
  vectorSDilD.unpack(vectorSOutD.H_elem());
  vectorSOutD.norm2Host();
  vectorCDilD.unpack(vectorCOutD.H_elem());
  vectorCOutD.norm2Host();
  vectorSCDilD.unpack(vectorSCOutD.H_elem());
  vectorSCOutD.norm2Host();
  vectorAuxD.zero_device();
  
  
  finalize();

  return 0;
}
