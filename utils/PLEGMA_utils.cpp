#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace quda;
#define ALLOCATE
#include "utils/PLEGMA_params.h"
#include "utils/QUDA_params.h"
#undef ALLOCATE

static bool isInitOpt = false;

void initializeOptions(int argc, char **argv, bool withQuda, std::vector<std::string> listOptPLEGMA){
  HGC_options = new Options(argc,argv);

  // initialize QMP/MPI, QUDA comms grid and RNG
  // we need to do it first for enabling the printing
  HGC_options->setForced("procs","Set number of processors (X Y Z T), e.g. 1 1 1 1", 0,
			 procs[0], procs[1], procs[2], procs[3]);
    
  for(int i=0; i<4; i++) if( procs[i] <= 0 )
			   PLEGMA_error("Error with dim %d: Negative proc or not divisor of dim\n", i);
  initComms(argc, argv, procs);

  // Reading plegma options
  plegmaOptions(*HGC_options, listOptPLEGMA);

  if(withQuda) {
    qudaOptions(*HGC_options);
    // initialize the QUDA library
    if(verbosity>0) infoQuda();
  }
  isInitOpt=true;
}

void initializePLEGMA() {
  if(!isInitOpt){fprintf(stderr,"initializeOptions should be called before initializePLEGMA");exit(EXIT_FAILURE);}
  HGC_options->close();
  initQuda(device);
  qudaInitialized=true;
  // initialize PLEGMA params
  PLEGMA_init(dims, procs, verbosity);
  PLEGMA_status();
}

void finalize() {
  delete HGC_options;
  PLEGMA_end();
  
  saveTuneCache(false);
  // finalize the QUDA library
  if(qudaInitialized) {
    finalizeGaugeQuda();
    endQuda();
  }
  
  // finalize the communications layer
  finalizeComms();
}

void createMom(int *Nmom, int momElem[][3], int Q_sq){
  int counter = 0;

  for(int iQ = 0 ; iQ <= Q_sq ; iQ++){
    for(int nx = iQ ; nx >= -iQ ; nx--)
      for(int ny = iQ ; ny >= -iQ ; ny--)
        for(int nz = iQ ; nz >= -iQ ; nz--){
          if( nx*nx + ny*ny + nz*nz == iQ ){
            momElem[counter][0] = nx;
            momElem[counter][1] = ny;
            momElem[counter][2] = nz;
            counter++;
          }
        }
  }
  *Nmom = counter;
}

template<typename FloatOut, typename FloatIn>
void unpackGaugeToEvenOdd(FloatOut *buf[4], PLEGMA_Gauge<FloatIn> &gauge)
{
  int VOLUME=gauge.Total_length();
  int VOLUMEh = VOLUME / 2;
  int gSize = N_COLS*N_COLS;
 
  for(int even = 0; even < VOLUMEh; even++) {
    int odd = even+VOLUMEh;
    int norm_coord = 2 * even;
    
    int evenSiteBit = 0;
    int tmp = norm_coord/dims[0];
    for(int i=1; i<N_DIMS; i++) {
      evenSiteBit += tmp%dims[i];
      tmp /= dims[i];
    }
    evenSiteBit = evenSiteBit % 2;
    int oddSiteBit  = evenSiteBit ^ 1;
    
    for(int dir = 0 ; dir < N_DIMS; dir++)
      for(int c = 0; c < gSize; c++) {
	buf[dir][(even*gSize + c)*2 + 0] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + evenSiteBit)*2  +0];
	buf[dir][(even*gSize + c)*2 + 1] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + evenSiteBit)*2  +1];
	buf[dir][(odd*gSize + c)*2 + 0] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + oddSiteBit)*2  +0];
	buf[dir][(odd*gSize + c)*2 + 1] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + oddSiteBit)*2  +1];
      }
  }
}

template void unpackGaugeToEvenOdd<double,double>(double *buf[4], PLEGMA_Gauge<double> &gauge);
template void unpackGaugeToEvenOdd<double,float>(double *buf[4], PLEGMA_Gauge<float> &gauge);
template void unpackGaugeToEvenOdd<float,double>(float *buf[4], PLEGMA_Gauge<double> &gauge);
template void unpackGaugeToEvenOdd<float,float>(float *buf[4], PLEGMA_Gauge<float> &gauge);

template<typename FloatOut, typename FloatIn>
void packGaugeToNormal(PLEGMA_Gauge<FloatOut> &gauge, FloatIn *buf[4])
{
  int VOLUME=gauge.Total_length();
  int VOLUMEh = VOLUME / 2;
  int gSize = N_COLS*N_COLS;
 
  for(int even = 0; even < VOLUMEh; even++) {
    int odd = even+VOLUMEh;
    int norm_coord = 2 * even;
    
    int evenSiteBit = 0;
    int tmp = norm_coord/dims[0];
    for(int i=1; i<N_DIMS; i++) {
      evenSiteBit += tmp%dims[i];
      tmp /= dims[i];
    }
    evenSiteBit = evenSiteBit % 2;
    int oddSiteBit  = evenSiteBit ^ 1;
    
    for(int dir = 0 ; dir < N_DIMS; dir++)
      for(int c = 0; c < gSize; c++) {
	gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + evenSiteBit)*2  +0] = buf[dir][(even*gSize + c)*2 + 0];
	gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + evenSiteBit)*2  +1] = buf[dir][(even*gSize + c)*2 + 1];
	gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + oddSiteBit)*2  +0] = buf[dir][(odd*gSize + c)*2 + 0];
	gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + oddSiteBit)*2  +1] = buf[dir][(odd*gSize + c)*2 + 1];
      }
  }
}

template void packGaugeToNormal<double,double>(PLEGMA_Gauge<double> &gauge, double *buf[4]);
template void packGaugeToNormal<float,double>(PLEGMA_Gauge<float> &gauge, double *buf[4]);
template void packGaugeToNormal<double,float>(PLEGMA_Gauge<double> &gauge, float *buf[4]);
template void packGaugeToNormal<float,float>(PLEGMA_Gauge<float> &gauge, float *buf[4]);


template<typename Float>
void applyAntiperiodicBoundary(Float **buf)
{
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  bool last_node_in_t = (commCoords(3) == commDim(3)-1) ? true : false;
#else
  bool last_node_in_t = true;
#endif

  // Apply boundary conditions to temporal links
  if (last_node_in_t) {
    int gSize = N_COLS*N_COLS*2;
    size_t Vh = dims[0]*dims[1]*dims[2]*dims[3]/2;
    for (size_t j = Vh-Vh/dims[3]; j < Vh; j++) {
      for (int i = 0; i < gSize; i++) {
	buf[3][j*gSize+i] *= -1.0;
	buf[3][(Vh+j)*gSize+i] *= -1.0;
      }
    }
  }
}

template void applyAntiperiodicBoundary<double>(double **buf);
template void applyAntiperiodicBoundary<float>(float **buf);

template<typename Float>
void applyBoundaryConditions(PLEGMA_Gauge<Float> &gauge, bool antiperiodic){
  if(!antiperiodic) return;
  
  Float* buf[N_DIMS];
  for(int i=0; i<N_DIMS; i++) hostMalloc(buf[i], gauge.Bytes_total()/N_DIMS);

  gauge.unload();
  unpackGaugeToEvenOdd(buf, gauge);
  applyAntiperiodicBoundary(buf);
  packGaugeToNormal(gauge,buf);
  gauge.load();
  for(int i=0; i<N_DIMS; i++) hostFree(buf[i], gauge.Bytes_total()/N_DIMS);
}

template void applyBoundaryConditions<double>(PLEGMA_Gauge<double> &gauge, bool antiperiodic);
template void applyBoundaryConditions<float>(PLEGMA_Gauge<float> &gauge, bool antiperiodic);

std::vector<int> createR2(std::vector<int> &vec){
  if(!HGC_init_PLEGMA_flag) PLEGMA_error("Initialize PLEGMA first");
  if(vec.size() != 0) PLEGMA_error("The vector provided is not empty");
  for(int xx = -HGC_totalL[0]/2; xx < HGC_totalL[0]/2; xx++)
    for(int yy = -HGC_totalL[1]/2; yy < HGC_totalL[1]/2; yy++)
      for(int zz = -HGC_totalL[2]/2; zz < HGC_totalL[2]/2; zz++)
	vec.push_back(xx*xx + yy*yy + zz*zz);
  return clearDuplicates(vec);
}


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
void M_e_GNG( Float *dest, GAMMAS_SCATT Gamma_i, GAMMAS_SCATT Gamma_f, Float *source ){
  const int N2=N_SPINS*N_SPINS*2;
  for (int i=0; i < N2; ++i)
    dest[i]=0.;
  for (int n_gamma_i=0; n_gamma_i<4; ++n_gamma_i) {    
    const int alfa =   gammaInd_scatt_host[Gamma_i][n_gamma_i][0];
    const int alfa0=   gammaInd_scatt_host[Gamma_i][n_gamma_i][1];
    Float gi[2];
    gi[1]=gamma_scatt_host[Gamma_i][n_gamma_i][1];
    gi[0]=gamma_scatt_host[Gamma_i][n_gamma_i][0];
    for (int n_gamma_f=0; n_gamma_f<4; ++n_gamma_f){
      const int beta=    gammaInd_scatt_host[Gamma_f][n_gamma_f][1];
      const int beta0=   gammaInd_scatt_host[Gamma_f][n_gamma_f][0];
      Float gf[2];
      gf[1]=gamma_scatt_host[Gamma_f][n_gamma_f][1];
      gf[0]=gamma_scatt_host[Gamma_f][n_gamma_f][0];
      dest[(alfa*N_SPIN+beta)*2+0]+=
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+0]*gi[0]
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[0]
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[1]
                -gi[0]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[1];
      dest[(alfa*N_SPIN+beta)*2+1]+=
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+1]*gi[1]
                +gi[1]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[0]
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[0]
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[1];
    }
  }
}

template void M_e_GNG<float>( float *dest, GAMMAS_SCATT Gamma_i, GAMMAS_SCATT Gamma_f, float *source) ;

template void M_e_GNG<double>( double *dest, GAMMAS_SCATT Gamma_i, GAMMAS_SCATT Gamma_f, double *source) ;

