#include<stdio.h>
#include<stdlib.h>
/*************** 
* This code extracts all the momentum combinations of 
* p_i1, p_i2, p_f1, p_f2
* for which |p_ij| ^2 <= individual_mom^2
*           |p_i1+p_i2| <= total_momentum^2 
* Input parameters p_imax^2
*                  p_totmax^2
* Output: all the possible combinations
****************/
int main(int argc, char *argv[]){

  if (argc!=3){
    printf("The usage of the code is ./a.out pi_max2 ptot_max2\n");
    exit(1);
  }
  int pimax2=atoi(argv[1]);
  int ptotmax2=atoi(argv[2]);
  
  int pf2x,pf2y,pf2z;
  int pf1x,pf1y,pf1z;
  int pi2x,pi2y,pi2z;
  for (pi2x=-pimax2; pi2x<= pimax2; ++pi2x){
    for (pi2y=-pimax2; pi2y<= pimax2; ++pi2y){
      for (pi2z=-pimax2; pi2z<= pimax2; ++pi2z){
        for (pf2x=-pimax2; pf2x <= pimax2; ++pf2x){
          for (pf2y=-pimax2; pf2y <= pimax2; ++pf2y){
            for (pf2z=-pimax2; pf2z <= pimax2; ++pf2z){
              for (pf1x=-pimax2; pf1x<= pimax2; ++pf1x){
                for (pf1y=-pimax2; pf1y<= pimax2; ++pf1y){
                  for (pf1z=-pimax2; pf1z<= pimax2; ++pf1z){
                    if ( ((pi2x*pi2x+pi2y*pi2y+pi2z*pi2z)>pimax2) || (pf2x*pf2x+pf2y*pf2y+pf2z*pf2z >pimax2) || (pf1x*pf1x+pf1y*pf1y+pf1z*pf1z >pimax2) || (((pf1x+pf2x)*(pf1x+pf2x)+(pf1y+pf2y)*(pf1y+pf2y)+(pf1z+pf2z)*(pf1z+pf2z)) >ptotmax2) )
                    continue;
                    printf("%d %d %d %d %d %d %d %d %d\n",pi2x,pi2y,pi2z,pf1x,pf1y,pf1z,pf2x,pf2y,pf2z);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

}
