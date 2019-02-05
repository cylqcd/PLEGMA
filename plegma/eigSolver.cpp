#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_BLAS.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  PLEGMA_params params;
  initialize(argc, argv, &params);
  PLEGMA_Vector<double> source(DEVICE);
  source.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  std::complex<double> a(2.,0.);
  std::complex<double> b(3.,0.);
  std::complex<double> c(2.,0.);
  plegma::axpbypcz(4*3*GK_localVolume,reinterpret_cast<double(&)[2]>(a), source.D_elem(), reinterpret_cast<double(&)[2]>(b), source.D_elem(), reinterpret_cast<double(&)[2]>(c), source.D_elem());
  std::complex<double> aka = source.dot(source);
  finalize();

  return 0;
}
