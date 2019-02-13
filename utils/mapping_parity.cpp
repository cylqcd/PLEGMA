#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <PLEGMA.h>
#include <errno.h>
#include <mpi.h>
#include <limits>


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

//----------------------------- Vector with spin-color inside volume -----------------------//

template <typename Float>
static void map_EvenOdd2Normal_Spinor_automorph(Float *spinor , int nx ,int ny ,int nz, int nt)
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

    for(int sc = 0; sc < 12; sc++)
    {
///load even site spinor:                                                                                                                                                     
      tmp[(norm_coord + evenSiteBit)*spinorSiteSize + sc*2 + 0] = spinor[even*spinorSiteSize + sc*2 + 0] ;
      tmp[(norm_coord + evenSiteBit)*spinorSiteSize + sc*2 + 1] = spinor[even*spinorSiteSize + sc*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[(norm_coord + oddSiteBit)*spinorSiteSize + sc*2 + 0] = spinor[odd*spinorSiteSize + sc*2 + 0] ;
      tmp[(norm_coord + oddSiteBit)*spinorSiteSize + sc*2 + 1] = spinor[odd*spinorSiteSize + sc*2 + 1] ;
    }
  }

  memcpy(spinor,tmp,VOLUME*sSize);
  free(tmp);
  
}

template <typename Float>
static void map_Normal2EvenOdd_Spinor_automorph(Float* spinor , int nx , int ny, int nz, int nt)
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

    for(int sc = 0; sc < 12; sc++)
      {
///load even site spinor:                                                                                                                                                     
      tmp[even*spinorSiteSize + sc*2 + 0] = spinor[(norm_coord + evenSiteBit)*spinorSiteSize + sc*2 + 0] ;
      tmp[even*spinorSiteSize + sc*2 + 1] = spinor[(norm_coord + evenSiteBit)*spinorSiteSize + sc*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[odd*spinorSiteSize + sc*2 + 0] = spinor[(norm_coord + oddSiteBit)*spinorSiteSize + sc*2 + 0] ;
      tmp[odd*spinorSiteSize + sc*2 + 1] = spinor[(norm_coord + oddSiteBit)*spinorSiteSize + sc*2 + 1] ;
      }
  }

  memcpy(spinor,tmp,VOLUME*sSize);
  free(tmp);
}


template<typename Float>
void mapNormalToEvenOdd(Float *spinor, int lL[4], QudaDiracFieldOrder order)
{
  if(order == QUDA_DIRAC_ORDER) map_Normal2EvenOdd_Spinor_automorph(spinor , lL[0] ,lL[1], lL[2], lL[3]);
  else errorQuda("Only QUDA_DIRAC_ORDER is supported");
}

template void mapNormalToEvenOdd<float>(float *spinor, int lL[4], QudaDiracFieldOrder order);
template void mapNormalToEvenOdd<double>(double *spinor, int lL[4], QudaDiracFieldOrder order);

template<typename Float>
void mapEvenOddToNormal(Float *spinor, int lL[4], QudaDiracFieldOrder order)
{
    if(order == QUDA_DIRAC_ORDER) map_EvenOdd2Normal_Spinor_automorph(spinor , lL[0] ,lL[1], lL[2], lL[3]);
    else errorQuda("Only QUDA_DIRAC_ORDER is supported");
}

template void mapEvenOddToNormal<float>(float *spinor, int lL[4], QudaDiracFieldOrder order);
template void mapEvenOddToNormal<double>(double *spinor, int lL[4], QudaDiracFieldOrder order);


//----------------------------- Vector with volume inside spin-color -----------------------//
template <typename Float>
static void map_EvenOdd2Normal_Spinor_GPU_format_automorph(Float *spinor , int nx ,int ny ,int nz, int nt)
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

    for(int sc = 0; sc < 12; sc++)
    {
///load even site spinor:                                                                                                                                                     
      tmp[sc*VOLUME*2 + (norm_coord + evenSiteBit)*2 + 0] = spinor[sc*VOLUME*2 + even*2 + 0] ;
      tmp[sc*VOLUME*2 + (norm_coord + evenSiteBit)*2 + 1] = spinor[sc*VOLUME*2 + even*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[sc*VOLUME*2 + (norm_coord + oddSiteBit)*2 + 0] = spinor[sc*VOLUME*2 + odd*2 + sc*2 + 0] ;
      tmp[sc*VOLUME*2 + (norm_coord + oddSiteBit)*2 + 1] = spinor[sc*VOLUME*2 + odd*2 + sc*2 + 1] ;
    }
  }
  memcpy(spinor,tmp,VOLUME*sSize);
  free(tmp);
  
}

template <typename Float>
static void map_Normal2EvenOdd_Spinor_GPU_format_automorph(Float* spinor , int nx , int ny, int nz, int nt)
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

    for(int sc = 0; sc < 12; sc++)
      {
///load even site spinor:                                                                                                                                                     
      tmp[sc*VOLUME*2 + even*2 + 0] = spinor[sc*VOLUME*2 + (norm_coord + evenSiteBit)*2 + 0] ;
      tmp[sc*VOLUME*2 + even*2 + 1] = spinor[sc*VOLUME*2 + (norm_coord + evenSiteBit)*2 + 1] ;
///load odd site spinor:                                                                                                                                                      
      odd = even + VOLUMEh;
      tmp[sc*VOLUME*2 + odd*2 + 0] = spinor[sc*VOLUME*2 + (norm_coord + oddSiteBit)*2 + 0] ;
      tmp[sc*VOLUME*2 + odd*2 + 1] = spinor[sc*VOLUME*2 + (norm_coord + oddSiteBit)*2 + 1] ;
      }
  }

  memcpy(spinor,tmp,VOLUME*sSize);
  free(tmp);
}


template<typename Float>
void mapNormalToEvenOddGPUformat(Float *spinor, int lL[4], QudaDiracFieldOrder order)
{
  if(order == QUDA_DIRAC_ORDER) map_Normal2EvenOdd_Spinor_GPU_format_automorph(spinor , lL[0] ,lL[1], lL[2], lL[3]);
  else errorQuda("Only QUDA_DIRAC_ORDER is supported");
}

template void mapNormalToEvenOddGPUformat<float>(float *spinor, int lL[4], QudaDiracFieldOrder order);
template void mapNormalToEvenOddGPUformat<double>(double *spinor, int lL[4], QudaDiracFieldOrder order);

template<typename Float>
void mapEvenOddToNormalGPUformat(Float *spinor, int lL[4], QudaDiracFieldOrder order)
{
    if(order == QUDA_DIRAC_ORDER) map_EvenOdd2Normal_Spinor_GPU_format_automorph(spinor , lL[0] ,lL[1], lL[2], lL[3]);
    else errorQuda("Only QUDA_DIRAC_ORDER is supported");
}

template void mapEvenOddToNormalGPUformat<float>(float *spinor, int lL[4], QudaDiracFieldOrder order);
template void mapEvenOddToNormalGPUformat<double>(double *spinor, int lL[4], QudaDiracFieldOrder order);



///////////////////////////////////////////////////  Gauge field ////////////////////////////////////////

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

template<typename Float>
void mapNormalToEvenOddGauge(Float **gauge, int lL[4], QudaGaugeFieldOrder order)
{
  if(order == QUDA_QDP_GAUGE_ORDER)
    map_Normal2EvenOdd_Gauge_automorph( gauge , lL[0] ,lL[1] ,lL[2], lL[3]);
  else
    errorQuda("only QDP order supported for gauge");
}

template void mapNormalToEvenOddGauge<float>(float **gauge, int lL[4], QudaGaugeFieldOrder order);
template void mapNormalToEvenOddGauge<double>(double **gauge, int lL[4], QudaGaugeFieldOrder order);

template<typename Float>
void mapEvenOddToNormalGauge(Float **gauge, int lL[4], QudaGaugeFieldOrder order)
{
  if(order == QUDA_QDP_GAUGE_ORDER)
    map_EvenOdd2Normal_Gauge_automorph(gauge, lL[0] ,lL[1] ,lL[2], lL[3]);
  else
    errorQuda("only QDP order supported for gauge");
}

template void mapEvenOddToNormalGauge<float>(float **gauge, int lL[4], QudaGaugeFieldOrder order);
template void mapEvenOddToNormalGauge<double>(double **gauge, int lL[4], QudaGaugeFieldOrder order);
