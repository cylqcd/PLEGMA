#include <PLEGMA.h>
#include <PLEGMA_utils.h>

/************
 *  It provides utility functions (e.g. initialization, updating, finalization, and some others)
 ***********/

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

void checkQudaError(){
   auto error = qudaGetLastError();
   if (error != QUDA_SUCCESS) { // check we don't have a sticky error
     qudaDeviceSynchronize();
     if (qudaGetLastError() != QUDA_SUCCESS)
      errorQuda("Failed to clear error state %s\n", qudaGetLastErrorString().c_str());
   }
}


void updateOptions(std::string filename, std::vector<std::string>& listOpt, std::function<void(Options&)> add_options){
  PLEGMA_printf("Reading options from file %s\n", filename.c_str());
  if(!filename.empty() and access( filename.c_str(), F_OK ) != -1){
    const char *aux_str[3];
    aux_str[0] = "random_string";
    aux_str[1] = "--inputFile";
    aux_str[2] = const_cast<char*>(filename.c_str());
    Options LocalOptions = Options(3,const_cast<char**>(aux_str));
    if(add_options) add_options(LocalOptions);
    plegmaOptions(LocalOptions, listOpt, true);
    qudaOptions(LocalOptions);
    if(verbosity>0) infoQuda();
    LocalOptions.close();
  }
}

void updateOptions(WHICHFLAVOR fl){
  std::string filename;
  switch(fl)
    {
    case LIGHT:
      filename = inputLIGHT;
      break;
    case STRANGE:
      filename = inputST;
      break;
    case CHARM:
      filename = inputCH;
      break;
    }
  std::vector<std::string> aux = {};
  return updateOptions(filename, aux);
}

void initializePLEGMA() {
  if(!isInitOpt){fprintf(stderr,"initializeOptions should be called before initializePLEGMA");exit(EXIT_FAILURE);}
  HGC_options->close();
  initQuda(device_number);
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
