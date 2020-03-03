#include <PLEGMA.h>
/**
 *
 *  @brief vector(spin x color)  matrix(spin x spin)  vector(spin x color) 
 *          multiplication for piN scattering project resulting in complex
 *          number: V1*gamma*V2
 *  @params Float * V1 pointer to a float array of size 2*N_COLS*N_SPINS
 *  @params Float * V2 pointer to a float array of size 2*N_COLS*N_SPINS
 *  @params GAMMAS_SCATT gamma enumerator specifies the gamma matrix
 *  @params bool transp transp==false then V1(a)Gamma(a,b)V2(b) is returned
 *                      transp==true  then V1(a)Gamma(b,a)V2(b) is returned 
 *  @params Float * Dest pointer to 2 Float number (complex)
 **/
template<typename Float>
void V_M_V( Float * V1, Float * V2, GAMMAS_SCATT gamma, bool transp, Float *Dest ){
   *(Dest+0)=0.;
   *(Dest+1)=0.;
   #pragma unroll
   for(int nz_e = 0 ; nz_e < 4 ; nz_e++){  
     int beta0= (!transp) ? gammaInd_scatt_host[gamma][nz_e][0] : gammaInd_scatt_host[gamma][nz_e][1];
     int beta1= (!transp) ? gammaInd_scatt_host[gamma][nz_e][1] : gammaInd_scatt_host[gamma][nz_e][0];
     #pragma unroll
     for (int nz_c = 0; nz_c < 3; nz_c++) {
       *(Dest+0)+= +V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt_host[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+0]
                   -V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt_host[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+1]
                   -V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt_host[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+1]
                   -V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt_host[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+0];
       *(Dest+1)+= -V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt_host[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+1]
                   +V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt_host[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+0]
                   +V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt_host[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+0]
                   +V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt_host[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+1];
     }
   }
}
template void V_M_V<float>( float * V1, float * V2, GAMMAS_SCATT gamma, bool transp, float *Dest );

template void V_M_V<double>( double * V1, double * V2, GAMMAS_SCATT gamma, bool transp, double *Dest );

/**
 *  @brief tensor*matrix multiplication  
 *         for piN scattering project returns a color vector
 *  @params Float * V1 pointer to a Float array of size 2*N_COLS*N_SPINS*N_SPINS
 *  @params GAMMAS_SCATT gamma enumerator specifies the gamma matrix
 *  @params bool transp if transp==false Gamma(a,b)*V1(b,a) is returned
 *                      if transp==true  Gamma(a,b)*V1(a,b) is returned
 *  @params Float *Dest pointer to array of Float with size 2*N_COLS
 **/
template<typename Float>
void V_TR_MM( Float * V1, GAMMAS_SCATT gamma,bool transp, Float *Dest ){
  #pragma unroll
  for (int nz_c = 0 ; nz_c < 3 ; nz_c++){
    *(Dest+2*nz_c+0) = 0;
    *(Dest+2*nz_c+1) = 0;
  }
  #pragma unroll
  for (int nz_c=0; nz_c < 3 ; nz_c++){
    #pragma unroll
    for(int nz_e_inner = 0 ; nz_e_inner < 4 ; nz_e_inner++){
      int beta0=(!transp) ? gammaInd_scatt_host[gamma][nz_e_inner][0] : gammaInd_scatt_host[gamma][nz_e_inner][1];
      int beta1=(!transp) ? gammaInd_scatt_host[gamma][nz_e_inner][1] : gammaInd_scatt_host[gamma][nz_e_inner][0];
      *(Dest+2*nz_c+0)+=+gamma_scatt_host[gamma][nz_e_inner][0]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+0]
                        -gamma_scatt_host[gamma][nz_e_inner][1]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+1];
      *(Dest+2*nz_c+1)+=+gamma_scatt_host[gamma][nz_e_inner][1]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+0]
                        +gamma_scatt_host[gamma][nz_e_inner][0]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+1];
    }
  }
  
}
template void V_TR_MM<float>( float * V1, GAMMAS_SCATT gamma, bool transp, float *Dest );

template void V_TR_MM<double>( double * V1, GAMMAS_SCATT gamma, bool transp, double *Dest );


template<typename Float>
void x_pe_cy( Float *dest, Float *floatcomplex, Float *temporary, int size ){
  for (int i=0; i<size; ++i){
    dest[2*i+0]+= floatcomplex[0]*temporary[2*i+0]-floatcomplex[1]*temporary[2*i+1];
    dest[2*i+1]+= floatcomplex[1]*temporary[2*i+0]+floatcomplex[0]*temporary[2*i+1];
  }
}
template void x_pe_cy<float>(  float *dest,  float  *floatcomplex, float  *temporary, int size) ;

template void x_pe_cy<double>( double *dest, double *floatcomplex, double *temporary, int size) ;

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

template<typename Float>
void M_e_GNG( Float *dest, const GAMMAS_SCATT Gamma_f, const GAMMAS_SCATT Gamma_i, const Float *source ){
  const int N2=N_SPINS*N_SPINS*2;
  for (int i=0; i < N2; ++i)
    dest[i]=0.;
  for (int n_gamma_f=0; n_gamma_f<4; ++n_gamma_f) {    
    const int alfa =   gammaInd_scatt_host[Gamma_f][n_gamma_f][0];
    const int alfa0=   gammaInd_scatt_host[Gamma_f][n_gamma_f][1];
    Float gf[2];
    gf[1]=gamma_scatt_host[Gamma_f][n_gamma_f][1];
    gf[0]=gamma_scatt_host[Gamma_f][n_gamma_f][0];
    for (int n_gamma_i=0; n_gamma_i<4; ++n_gamma_i){
      const int beta=    gammaInd_scatt_host[Gamma_i][n_gamma_i][1];
      const int beta0=   gammaInd_scatt_host[Gamma_i][n_gamma_i][0];
      Float gi[2];
      gi[1]=gamma_scatt_host[Gamma_i][n_gamma_i][1];
      gi[0]=gamma_scatt_host[Gamma_i][n_gamma_i][0];
      dest[(alfa*N_SPINS+beta)*2+0]+=
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[0]
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[0]
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[1]
                -gi[0]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[1];
      dest[(alfa*N_SPINS+beta)*2+1]+=
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[1]
                +gi[1]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[0]
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[0]
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[1];
    }
  }
}

template void M_e_GNG<float>( float *dest, const GAMMAS_SCATT Gamma_f, const GAMMAS_SCATT Gamma_i, const float *source) ;

template void M_e_GNG<double>( double *dest, const GAMMAS_SCATT Gamma_f, const GAMMAS_SCATT Gamma_i, const double *source) ;

static inline GAMMAS_SCATT apply_g5(GAMMAS_SCATT source, LEFTRIGHT LR)
{
  switch(LR){
  case(LEFT):
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
    }
    break;
  case(RIGHT):
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
