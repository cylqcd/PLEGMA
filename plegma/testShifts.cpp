#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, false); // Add list of options
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  PLEGMA_Vector<double> vec1(BOTH, FIRST_VERTEX);
  PLEGMA_Vector<double> vec2;
  PLEGMA_Vector<double> vec3;
  PLEGMA_Vector<double> vec4;

  vec1.random();
  for(int dir1=0; dir1<8; dir1++)
    for(int dir2=0; dir2<8; dir2++) {
      if(dir1%4==dir2%4) continue;
      vec2.shift(vec1, dir1, dir2);
      vec3.shift(vec1, dir1);
      vec4.shift(vec3, dir2);
      vec4.add(vec2,-1);
      PLEGMA_printf("dir1=%d dir2=%d diff=%e\n", dir1, dir2, vec4.norm());
    }

  PLEGMA_Vector<double> vec5;

  for(int dir1=0; dir1<8; dir1++)
    for(int dir2=0; dir2<8; dir2++)
      for(int dir3=0; dir3<8; dir3++) {
	if(dir1%4==dir2%4 || dir1%4==dir3%4 || dir3%4==dir2%4) continue;
	vec2.shift(vec1, dir1, dir2, dir3);
	vec3.shift(vec1, dir1);
	vec4.shift(vec3, dir2);
	vec5.shift(vec4, dir3);
	vec5.add(vec2,-1);
	PLEGMA_printf("dir1=%d dir2=%d dir3=%d diff=%e\n", dir1, dir2, dir3, vec5.norm());
      }
  
  finalize();
 
  return 0;
}
