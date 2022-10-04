#pragma once
#include <PLEGMA_global.h>
#include <PLEGMA_io.h>
#include <PLEGMA_Field.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_Correlator.h>
#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_QLoops.h>
#include <PLEGMA_FT.h>
#include <PLEGMA_Fmunu.h>
#include <PLEGMA_U1Gauge.h>
namespace plegma {
  
  void PLEGMA_init(int localL[4], int nProcs[4], int verbosity);
  
  void PLEGMA_status();
  
  void PLEGMA_end();
  
}
