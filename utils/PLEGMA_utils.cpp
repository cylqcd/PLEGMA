#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace quda;

void initPlegma(int argc, char **argv, PLEGMA_params *params){
  initComms(argc, argv, params->procs);
  initQuda(device);
  PLEGMA_init(params);
  print_status();  
}

void initialize(int argc, char **argv, PLEGMA_params *params) {
  read_command_line(argc, argv, params);
  
  // initialize QMP/MPI, QUDA comms grid and RNG 
  initComms(argc, argv, params->procs);

  // initialize the QUDA library
  initQuda(device);
  print_info();

  // initialize PLEGMA params
  PLEGMA_init(params);
  print_status();
}

void finalize() {
  PLEGMA_end();
  
  // finalize the QUDA library
  saveTuneCache(false);
  finalizeGaugeQuda();
  endQuda();
  
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

// =================== Mapping utils for spinor ===================//

static int getLatticeCoordinateParity2(int latt_coord, int nx , int ny , int nz)
{
 	int x2, x3, x4;                 //x / 2, y, z, t normal coordinates on even/odd latice  
        int z1, z2;
        z1  = (2 * latt_coord) / nx;        //latt_coord - lattice coordinate of the half lattice
        z2  = z1 / ny;
        x2  = z1 - z2 * ny;
        x4  = z2 / nz;
        x3  = z2 - x4 * nz;

        return ((x2 + x3 + x4 + 0) & 1);
}

template <typename Float>
static void map_EvenOdd2Normal_DiracOrderedSpinor_automorph(Float *spinor , int nx ,int ny ,int nz, int nt)
{
  
  int VOLUME=nx*ny*nz*nt;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;
  size_t sSize =spinorSiteSize* sizeof(Float);
  Float *tmp = (Float*)malloc(VOLUME * sSize);
  for(int even = 0; even < VOLUMEh; even++)
  {
    norm_coord = 2 * even;

    evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
    oddSiteBit  = evenSiteBit ^ 1;

    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                     
      tmp[(norm_coord + evenSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 0] = spinor[even*spinorSiteSize + s*3*2 + c*2 + 0] ;
      tmp[(norm_coord + evenSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 1] = spinor[even*spinorSiteSize + s*3*2 + c*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[(norm_coord + oddSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 0] = spinor[odd*spinorSiteSize + s*3*2 + c*2 + 0] ;
      tmp[(norm_coord + oddSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 1] = spinor[odd*spinorSiteSize + s*3*2 + c*2 + 1] ;
      }
    }
  }

  for(int i = 0; i < VOLUME; i++)///!!!                                                                                                                                      
    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                   
      spinor[i*spinorSiteSize + s*(3*2) + c *2 + 0] = tmp[i*spinorSiteSize + s*3*2 + c*2 + 0] ;
      spinor[i*spinorSiteSize + s*(3*2) + c *2 + 1] = tmp[i*spinorSiteSize + s*3*2 + c*2 + 1] ;

      }
    }

  free(tmp);
  
}

template <typename Float>
static void map_Normal2EvenOdd_DiracOrderedSpinor_automorph(Float* spinor , int nx , int ny, int nz, int nt)
{ 
  int VOLUME=nx*ny*nz*nt;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;
  size_t sSize =spinorSiteSize* sizeof(Float);
  Float *tmp = (Float*)malloc(VOLUME * sSize);
  for(int even = 0; even < VOLUMEh; even++)
  {
    norm_coord = 2 * even;

    evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
    oddSiteBit  = evenSiteBit ^ 1;

    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                     
      tmp[even*spinorSiteSize + s*3*2 + c*2 + 0] = spinor[(norm_coord + evenSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 0] ;
      tmp[even*spinorSiteSize + s*3*2 + c*2 + 1] = spinor[(norm_coord + evenSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[odd*spinorSiteSize + s*3*2 + c*2 + 0] = spinor[(norm_coord + oddSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 0] ;
      tmp[odd*spinorSiteSize + s*3*2 + c*2 + 1] = spinor[(norm_coord + oddSiteBit)*spinorSiteSize + s*(3*2) + c *2 + 1] ;
      }
    }
  }
  for(int i = 0; i < VOLUME; i++)
    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                    
      spinor[i*spinorSiteSize + s*(3*2) + c *2 + 0] = tmp[i*spinorSiteSize + s*3*2 + c*2 + 0] ;
      spinor[i*spinorSiteSize + s*(3*2) + c *2 + 1] = tmp[i*spinorSiteSize + s*3*2 + c*2 + 1] ;

      }
    }

  free(tmp);
}

template <typename Float>
static void map_EvenOdd2Normal_QDPDiracOrderedSpinor_automorph(Float* spinor ,int nx , int ny , int nz, int nt)
{
 
  int VOLUME=nx*ny*nz*nt;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;
  size_t sSize =spinorSiteSize* sizeof(Float);
  Float *tmp = (Float*)malloc(VOLUME * sSize);
  for(int even = 0; even < VOLUMEh; even++)
  {
    norm_coord = 2 * even;

    evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
    oddSiteBit  = evenSiteBit ^ 1;

    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                     
      tmp[(norm_coord + evenSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 0] = spinor[even*spinorSiteSize + c*4*2 + s*2 + 0] ;
      tmp[(norm_coord + evenSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 1] = spinor[even*spinorSiteSize + c*4*2 + s*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[(norm_coord + oddSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 0] = spinor[odd*spinorSiteSize + c*4*2 + s*2 + 0] ;
      tmp[(norm_coord + oddSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 1] = spinor[odd*spinorSiteSize + c*4*2 + s*2 + 1] ;
      }
    }
  }

  for(int i = 0; i < VOLUME; i++)///!!!                                                                                                                                       
    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                    
      spinor[i*spinorSiteSize + c*(4*2) + s *2 + 0] = tmp[i*spinorSiteSize + c*4*2 + s*2 + 0] ;
      spinor[i*spinorSiteSize + c*(4*2) + s *2 + 1] = tmp[i*spinorSiteSize + c*4*2 + s*2 + 1] ;

      }
    }

  free(tmp);
}

template <typename Float>
static void map_Normal2EvenOdd_QDPDiracOrderedSpinor_automorph(Float* spinor, int nx , int ny, int nz, int nt)
{
 
  int VOLUME=nx*ny*nz*nt;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;
  size_t sSize =spinorSiteSize* sizeof(Float);
  Float *tmp = (Float*)malloc(VOLUME * sSize);
  for(int even = 0; even < VOLUMEh; even++)
  {
    norm_coord = 2 * even;

    evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
    oddSiteBit  = evenSiteBit ^ 1;

    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                     
      tmp[even*spinorSiteSize + c*4*2 + s*2 + 0] = spinor[(norm_coord + evenSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 0] ;
      tmp[even*spinorSiteSize + c*4*2 + s*2 + 1] = spinor[(norm_coord + evenSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[odd*spinorSiteSize + c*4*2 + s*2 + 0] = spinor[(norm_coord + oddSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 0] ;
      tmp[odd*spinorSiteSize + c*4*2 + s*2 + 1] = spinor[(norm_coord + oddSiteBit)*spinorSiteSize + c*(4*2) + s *2 + 1] ;
      }
    }
  }
  for(int i = 0; i < VOLUME; i++)
    for(int s = 0; s < 4; s++)
    {
      for(int c = 0; c < 3; c++)
      {
///load even site spinor:                                                                                                                                                     
      spinor[i*spinorSiteSize + c*(4*2) + s *2 + 0] = tmp[i*spinorSiteSize + c*4*2 + s*2 + 0] ;
      spinor[i*spinorSiteSize + c*(4*2) + s *2 + 1] = tmp[i*spinorSiteSize + c*4*2 + s*2 + 1] ;

      }
    }

  free(tmp);
}

template <typename Float>
void map_EvenOdd2Normal_automorph(Float *spinor , int nx ,int ny ,int nz, int nt)
{
  
  int VOLUME=nx*ny*nz;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;

  Float *tmp = (Float*)malloc(VOLUME * 2*sizeof(Float));
  for(int even = 0; even < VOLUMEh; even++)
  {
    norm_coord = 2 * even;

    evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
    oddSiteBit  = evenSiteBit ^ 1;
    
///load even site spinor:                                                                                                                                                     
    tmp[(norm_coord + evenSiteBit)*2 + 0] = spinor[even*2 + 0] ;
    tmp[(norm_coord + evenSiteBit)*2 + 1] = spinor[even*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
    odd = even + VOLUMEh;
    tmp[(norm_coord + oddSiteBit)*2 + 0] = spinor[odd*2 + 0] ;
    tmp[(norm_coord + oddSiteBit)*2 + 1] = spinor[odd*2 + 1] ;

  }

  for(int i = 0; i < VOLUME; i++){
    
      spinor[i*2 + 0] = tmp[i*2 + 0] ;
      spinor[i*2 + 1] = tmp[i*2 + 1] ;

      
    }

  free(tmp);
  
}


void mapNormalToEvenOdd(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt)
{

  if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
    if(param.dirac_order == QUDA_DIRAC_ORDER) map_Normal2EvenOdd_DiracOrderedSpinor_automorph((double*) spinor , nx ,ny, nz, nt);
    if(param.dirac_order == QUDA_QDP_DIRAC_ORDER) map_Normal2EvenOdd_QDPDiracOrderedSpinor_automorph((double*)spinor, nx ,ny, nz, nt);
  }
  else
    {
    if(param.dirac_order == QUDA_DIRAC_ORDER) map_Normal2EvenOdd_DiracOrderedSpinor_automorph((float*) spinor , nx ,ny, nz, nt);
    if(param.dirac_order == QUDA_QDP_DIRAC_ORDER) map_Normal2EvenOdd_QDPDiracOrderedSpinor_automorph((float*)spinor, nx ,ny, nz, nt);
    }
}

void mapEvenOddToNormal(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt)
{

  if(param.cpu_prec == QUDA_DOUBLE_PRECISION){
    if(param.dirac_order == QUDA_DIRAC_ORDER) map_EvenOdd2Normal_DiracOrderedSpinor_automorph((double*) spinor , nx ,ny, nz, nt);
    if(param.dirac_order == QUDA_QDP_DIRAC_ORDER) map_EvenOdd2Normal_QDPDiracOrderedSpinor_automorph((double*) spinor , nx ,ny, nz, nt);
  }
  else
    {
    if(param.dirac_order == QUDA_DIRAC_ORDER) map_EvenOdd2Normal_DiracOrderedSpinor_automorph((float*) spinor , nx ,ny, nz, nt);
    if(param.dirac_order == QUDA_QDP_DIRAC_ORDER) map_EvenOdd2Normal_QDPDiracOrderedSpinor_automorph((float*) spinor , nx ,ny, nz, nt);
    }

}




//================ Mapping utils for Gauge field =========================//

template <typename Float>
static void map_EvenOdd2Normal_Gauge_automorph(Float **gauge , int nx ,int ny ,int nz, int nt)
{
  
  int VOLUME=nx*ny*nz*nt;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;
  //  size_t sSize = spinorSiteSize * sizeof(Float);
  size_t gSize = gaugeSiteSize * sizeof(Float);
 
  //  Float *tmp = (Float*)malloc(VOLUME * sSize);

  Float *gauge_tmp[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge_tmp[dir] = (Float*)malloc(VOLUME * gSize);
  }

  for(int dir = 0 ; dir < 4 ; dir++)
    for(int even = 0; even < VOLUMEh; even++)
      {
	norm_coord = 2 * even;
	
	evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
	oddSiteBit  = evenSiteBit ^ 1;
	
	for(int c1 = 0; c1 < 3; c1++)
	  {
	    for(int c2 = 0; c2 < 3; c2++)
	      {
		///load even site gauge:
		gauge_tmp[dir][(norm_coord + evenSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 0] = gauge[dir][even*gaugeSiteSize + c1*3*2 + c2*2 + 0] ;
		gauge_tmp[dir][(norm_coord + evenSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 1] = gauge[dir][even*gaugeSiteSize + c1*3*2 + c2*2 + 1] ;
		///load odd site gauge:               
		odd = even + VOLUMEh;
		gauge_tmp[dir][(norm_coord + oddSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 0] = gauge[dir][odd*gaugeSiteSize + c1*3*2 + c2*2 + 0] ;
		gauge_tmp[dir][(norm_coord + oddSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 1] = gauge[dir][odd*gaugeSiteSize + c1*3*2 + c2*2 + 1] ;
	      }
	  }
      }
  
  for(int dir = 0 ; dir < 4 ; dir++)
    for(int i = 0; i < VOLUME; i++)
      for(int c1 = 0; c1 < 3; c1++)    
	for(int c2 = 0; c2 < 3; c2++)
	  {
	    gauge[dir][i*gaugeSiteSize + c1*(3*2) + c2*2 + 0] = gauge_tmp[dir][i*gaugeSiteSize + c1*3*2 + c2*2 + 0] ;
	    gauge[dir][i*gaugeSiteSize + c1*(3*2) + c2*2 + 1] = gauge_tmp[dir][i*gaugeSiteSize + c1*3*2 + c2*2 + 1] ;
	  }


  for(int dir = 0 ; dir < 4 ; dir++)
    free(gauge_tmp[dir]);
  
}



template <typename Float>
static void map_Normal2EvenOdd_Gauge_automorph(Float **gauge , int nx ,int ny ,int nz, int nt)
{
  
  int VOLUME=nx*ny*nz*nt;

  int VOLUMEh = VOLUME / 2;
  int norm_coord, odd;
  int evenSiteBit, oddSiteBit;
  //  size_t sSize = spinorSiteSize * sizeof(Float);
  size_t gSize = gaugeSiteSize * sizeof(Float);
 
  //  Float *tmp = (Float*)malloc(VOLUME * sSize);

  Float *gauge_tmp[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge_tmp[dir] = (Float*)malloc(VOLUME * gSize);
  }

  for(int dir = 0 ; dir < 4 ; dir++)
    for(int even = 0; even < VOLUMEh; even++)
      {
	norm_coord = 2 * even;
	
	evenSiteBit = getLatticeCoordinateParity2(even,nx,ny,nz);
	oddSiteBit  = evenSiteBit ^ 1;
	
	for(int c1 = 0; c1 < 3; c1++)
	  {
	    for(int c2 = 0; c2 < 3; c2++)
	      {
		///load even site gauge:
		gauge_tmp[dir][even*gaugeSiteSize + c1*3*2 + c2*2 + 0] = gauge[dir][(norm_coord + evenSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 0] ;
		gauge_tmp[dir][even*gaugeSiteSize + c1*3*2 + c2*2 + 1] = gauge[dir][(norm_coord + evenSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 1] ;
		///load odd site gauge:               
		odd = even + VOLUMEh;
		gauge_tmp[dir][odd*gaugeSiteSize + c1*3*2 + c2*2 + 0] = gauge[dir][(norm_coord + oddSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 0] ;
		gauge_tmp[dir][odd*gaugeSiteSize + c1*3*2 + c2*2 + 1] = gauge[dir][(norm_coord + oddSiteBit)*gaugeSiteSize + c1*(3*2) + c2*2 + 1] ;

	      }
	  }
      }
  
  for(int dir = 0 ; dir < 4 ; dir++)
    for(int i = 0; i < VOLUME; i++)
      for(int c1 = 0; c1 < 3; c1++)    
	for(int c2 = 0; c2 < 3; c2++)
	  {
	    gauge[dir][i*gaugeSiteSize + c1*(3*2) + c2*2 + 0] = gauge_tmp[dir][i*gaugeSiteSize + c1*3*2 + c2*2 + 0] ;
	    gauge[dir][i*gaugeSiteSize + c1*(3*2) + c2*2 + 1] = gauge_tmp[dir][i*gaugeSiteSize + c1*3*2 + c2*2 + 1] ;
	  }


  for(int dir = 0 ; dir < 4 ; dir++)
    free(gauge_tmp[dir]);
  
}


void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam &param, int nx , int ny , int nz, int nt)
{

  if(param.gauge_order == QUDA_QDP_GAUGE_ORDER)
    map_Normal2EvenOdd_Gauge_automorph( gauge , nx ,ny ,nz, nt);
  else
    errorQuda("only QDP order supported for gauge");
}

void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam &param, int lL[4])
{
  mapNormalToEvenOddGauge( gauge, param, lL[0], lL[1], lL[2], lL[3]);
}

void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam &param, int nx , int ny , int nz, int nt)
{

  if(param.gauge_order == QUDA_QDP_GAUGE_ORDER)
    map_EvenOdd2Normal_Gauge_automorph(gauge, nx ,ny ,nz, nt);
  else
    errorQuda("only QDP order supported for gauge");

}

void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam &param, int lL[4])
{
  mapEvenOddToNormalGauge( gauge, param, lL[0], lL[1], lL[2], lL[3]);
}

// ============= Apply boundary to gauge field =================//

template <typename Float>
void applyGaugeFieldScaling(Float **gauge, int Vh, QudaGaugeParam *param) {
  // Apply spatial scaling factor (u0) to spatial links
  for (int d = 0; d < 3; d++) {
    for (int i = 0; i < gaugeSiteSize*Vh*2; i++) {
      gauge[d][i] /= param->anisotropy;
    }
  }
    
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  bool last_node_in_t = (commCoords(3) == commDim(3)-1) ? true : false;
#else
  bool last_node_in_t = true;
#endif

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t) {
    for (int j = (HGC_localL[0]/2)*HGC_localL[1]*HGC_localL[2]*(HGC_localL[3]-1); j < Vh; j++) {
      for (int i = 0; i < gaugeSiteSize; i++) {
	gauge[3][j*gaugeSiteSize+i] *= -1.0;
	gauge[3][(Vh+j)*gaugeSiteSize+i] *= -1.0;
      }
    }
  }
    
  if (param->gauge_fix) {
    // set all gauge links (except for the last HGC_localL[0]*HGC_localL[1]*HGC_localL[2]/2) to the identity,
    // to simulate fixing to the temporal gauge.
    int iMax = ( last_node_in_t ? (HGC_localL[0]/2)*HGC_localL[1]*HGC_localL[2]*(HGC_localL[3]-1) : Vh );
    int dir = 3; // time direction only
    Float *even = gauge[dir];
    Float *odd  = gauge[dir]+Vh*gaugeSiteSize;
    for (int i = 0; i< iMax; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }
}

void applyBoundaryCondition(double **gauge, int Vh ,QudaGaugeParam *gauge_param){

  if( gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION )
    applyGaugeFieldScaling<double>((double**)gauge, Vh, gauge_param); 
  else
    errorQuda("boundary condition application implement only for double precision");
}

void applyBoundaryCondition(double **gauge, int lL[4] ,QudaGaugeParam *gauge_param){
  int Vh = lL[0]*lL[1]*lL[2]*lL[3]/2;
  applyBoundaryCondition(gauge, Vh, gauge_param);
}
