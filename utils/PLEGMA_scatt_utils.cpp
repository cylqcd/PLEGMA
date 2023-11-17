#include <PLEGMA.h>
//vectorOut <- vectorOut + c * vectorIn 
template<typename Float>
void x_pe_cy( Float *dest, Float *floatcomplex, Float *temporary, int size ){
  for (int i=0; i<size; ++i){
    dest[2*i+0]+= floatcomplex[0]*temporary[2*i+0]-floatcomplex[1]*temporary[2*i+1];
    dest[2*i+1]+= floatcomplex[1]*temporary[2*i+0]+floatcomplex[0]*temporary[2*i+1];
  }
}
template void x_pe_cy<float>(  float *dest,  float  *floatcomplex, float  *temporary, int size) ;

template void x_pe_cy<double>( double *dest, double *floatcomplex, double *temporary, int size) ;

//vectorOut <- vectorOut + s * vectorIn 
template<typename Float>
void x_pe_sy( Float *dest, Float floatnumber, Float *temporary, int size ){
  for (int i=0; i<size; ++i){
    dest[2*i+0]+= floatnumber*temporary[2*i+0];
    dest[2*i+1]+= floatnumber*temporary[2*i+1];
  }
}
template void x_pe_sy<float>(  float *dest,  float  floatnumber, float  *temporary, int size) ;

template void x_pe_sy<double>( double *dest, double floatnumber, double *temporary, int size) ;


//vectorOut <- vectorOut + vectorIn 
template<typename Float>
void x_pe_y( Float *dest, Float *temporary, int size ){
  for (int i=0; i<size; ++i){
    dest[2*i+0]+= temporary[2*i+0];
    dest[2*i+1]+= temporary[2*i+1];
  }
}
template void x_pe_y<float>(  float *dest,  float  *temporary, int size) ;

template void x_pe_y<double>( double *dest, double *temporary, int size) ;

//vectorOut <- c * vectorOut
template<typename Float>
void x_e_cx( Float *dest, const Float floatcomplex[2],  int size ){
  for (int i=0; i<size; ++i){
    Float tmpre=floatcomplex[0]*dest[2*i+0]-floatcomplex[1]*dest[2*i+1];
    Float tmpim=floatcomplex[1]*dest[2*i+0]+floatcomplex[0]*dest[2*i+1];
    dest[2*i+0]= tmpre;
    dest[2*i+1]= tmpim;
  }
}
template void x_e_cx<float>(  float *dest,  const float  floatcomplex[2], int size) ;

template void x_e_cx<double>( double *dest, const double floatcomplex[2], int size) ;
//vectorOut <- s * vectorOut
template<typename Float>
void x_e_sx( Float *dest, const Float floatreal,  int size ){
  for (int i=0; i<size; ++i){
    Float tmpim=floatreal*dest[2*i+1];
    Float tmpre=floatreal*dest[2*i+0];
    dest[2*i+1]= tmpim;
    dest[2*i+0]= tmpre;
  }
}
template void x_e_sx<float>(  float *dest,  const float  floatreal, int size) ;

template void x_e_sx<double>( double *dest, const double floatreal, int size) ;

 

GAMMAS_SCATT apply_g5(GAMMAS_SCATT source, LEFTRIGHT LR)
{
  switch(LR){
  case(RIGHT):
    switch (source){
    case(C):
      return CG_5;
      break;
    case(CG_5): 
      return C;
      break;
    case(CG_4):
      return(CG_4_G_5);
      break;
    case(CG_5_G_4):
      return (CG_5_G_4_G_5);
      break;
    case(ID): 
      return G_5;
      break;
    case(G_1):
      return G_1_G_5;
      break;
    case(G_2):
      return G_2_G_5;
      break;
    case(G_3):
      return G_3_G_5;
      break;
    case(G_4):
      return G_4_G_5;
      break;
    case(G_5): 
      return ID;
      break;
    case(CG_1):
      return CG_1_G_5;
      break;
    case(CG_2):
      return CG_2_G_5;
      break;
    case(CG_3):
      return CG_3_G_5;
      break;
    case(CG_1_G_4):
      return CG_1_G_4_G_5;
      break;
    case(CG_2_G_4):
      return CG_2_G_4_G_5;
      break;
    case(CG_3_G_4):
      return CG_3_G_4_G_5;
      break;
    case(CG_1_G_4_G_5):
      return CG_1_G_4;
      break;
    case(CG_2_G_4_G_5):
      return CG_2_G_4;
      break;
    case(CG_3_G_4_G_5):
      return CG_3_G_4;
      break;    
    case(S_12):
      return S_43;
      break;
    case(S_13):
      return MC;
      break;
    case(S_23):
      return S_41;
      break;
    case(S_41):
      return S_23;
      break;
    case(S_42):
      return CG_5; 
      break;
    case(S_43):
      return S_12;
      break;
    }
    break;
  case(LEFT):
    switch (source){
    case(C):
      return CG_5;
      break;
    case(CG_5): 
      return C;
      break;
    case(CG_4):
      return CG_5_G_4;
      break;
    case(CG_5_G_4):
      return CG_4;
      break;
    case(ID):
      return G_5;
      break;
    case(G_1):
      return G_5_G_1;
      break;
    case(G_2):
      return G_5_G_2;
      break;
    case(G_3):
      return G_5_G_3;
      break;
    case(G_4):
      return G_5_G_4;
      break;
    case(G_5):
      return ID;
      break;
    case(CG_1):
      return G_5_CG_1;
      break;
    case(CG_2):
      return G_5_CG_2;
      break;
    case(CG_3):
      return G_5_CG_3;
      break;
    case(CG_1_G_4):
      return CG_1_G_4_G_5;
      break;
    case(CG_2_G_4):
      return CG_2_G_4_G_5;
      break;
    case(CG_3_G_4):
      return CG_3_G_4_G_5;
      break;
    case(CG_1_G_4_G_5):
      return CG_1_G_4;
      break;
    case(CG_2_G_4_G_5):
      return CG_2_G_4;
      break;
    case(CG_3_G_4_G_5):
      return CG_3_G_4;
      break;
    case(S_12):
      return S_43;
      break;
    case(S_13):
      return MC;
      break;
    case(S_23):
      return S_41;
      break;
    case(S_41):
      return S_23;
      break;
    case(S_42):
      return CG_5;
      break;
    case(S_43):
      return S_12;
      break;
    }
    default: 
      PLEGMA_error("Gamma matrix multiplication with gamma is not implemented for the particular source gamma %d\n",source);
  }
  return ID;
}
std::vector<GAMMAS_SCATT> apply_gamma5_scatt_gamma( std::vector<GAMMAS_SCATT> &source, LEFTRIGHT LR){
  std::vector<GAMMAS_SCATT> output;
  if (source.empty()){
   PLEGMA_error("Trying to apply g5 on an empty list of gammas\n");
  }
  const int d_gamma=source.size();
  for (int i=0; i<d_gamma; ++i){
    output.push_back(apply_g5( source[i], LR));
  }
  return(output);
}   
