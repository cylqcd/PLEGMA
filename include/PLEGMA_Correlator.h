#include <PLEGMA_global.h>
#include <PLEGMA_Su3field.h>
#ifndef _PLEGMA_CORRELATOR_H
#define _PLEGMA_CORRELATOR_H

namespace plegma {

  // Information for correlators
  enum CORR_TYPE{MESONS,BARYONS,THRP_LOCAL,THRP_NOETHER,THRP_ONED,
		 // add here
                 N_CORR}; // N_CORR must be last
  
  enum BARYONS_TYPE{NtoN,
#ifdef ALL_BARYONS
		    NtoR, RtoN, RtoR, DELTA_1O2_1, DELTA_1O2_2, DELTA_1O2_3,
		    DELTA_3O2_1, DELTA_3O2_2, DELTA_3O2_3,
#endif
		    // add here
		    N_BARYONS}; // N_BARYONS must be last 

  const int nGroups[N_CORR] = { N_MESONS, N_BARYONS, 1, 1, 1 };
  const int nFlavors[N_CORR] = { 2, 2, 1, 1, 1 };
  const int nComp[N_CORR] = { 1, N_SPINS*N_SPINS, N_SPINS*N_SPINS, N_DIMS, N_SPINS*N_SPINS*N_DIMS };
  const int nDims[N_CORR] = { 0, 1, 1, 1, 2 };

  // Names used for writing in HDF5
  const static char *meson_groups[N_MESONS] = {"pseudoscalar", "scalar",
					       "g5g1", "g5g2", "g5g3", "g5g4",
					       "g1", "g2", "g3", "g4"};

  const static char *meson_flavors[2] = {"twop_meson_1",
					 "twop_meson_2"};

  const static char *baryons_groups[N_BARYONS] = {"nucl_nucl",
#ifdef ALL_BARYONS
						  "nucl_nucl2","nucl2_nucl","nucl2_nucl2","deltap_deltaz_11","deltap_deltaz_22","deltap_deltaz_33",
						  "deltapp_deltamm_11","deltapp_deltamm_22","deltapp_deltamm_33"
#endif
  };

  const static char *baryons_flavors[2] = {"twop_baryon_1",
					   "twop_baryon_2"};

  const static char **corr_groups_names[N_CORR] = {meson_groups, baryons_groups, NULL};
  const static char **corr_flavors_names[N_CORR] = {meson_flavors, baryons_flavors, NULL};


  
  // forward declaration
  template<typename Float>  class PLEGMA_Vector;
  template<typename Float>  class PLEGMA_Propagator;
  template<typename Float>  class PLEGMA_Propagator3D;

  //////////////////////////////
  // CLASS: PLEGMA_Correlator //
  ////////////////////////////// 

  template<typename Float>
    class PLEGMA_Correlator {
  protected:
    Float* corr;
    bool isAlloc;
    
    CORR_TYPE corr_type;
    CORR_SPACE corr_space;
    int isource;

    int n_groups;
    int n_flavors;
    int n_comp;
    size_t vol_size;
    size_t site_size;
    size_t bytes_total_length;

    void change_order_for_HDF5(Float* corrHDF5);
    void initialize(CORR_TYPE CorrType, CORR_SPACE CorrSpace);
    void finalize() {
      if (isAlloc) free(corr);
      isAlloc = 0;
    }

  public:
    PLEGMA_Correlator(){isAlloc = 0;}
    PLEGMA_Correlator(CORR_TYPE CorrType, CORR_SPACE CorrSpace){isAlloc = 0; initialize(CorrType, CorrSpace);};
    ~PLEGMA_Correlator(){finalize();}
    CORR_SPACE getCorrSpace() {
      return corr_space;
    }
    size_t getSiteSize() {
      return site_size;
    }
    size_t getVolSize() {
      return vol_size;
    }
    size_t getTotalSize() {
      return site_size*vol_size;
    }
    int getIdSource() {
      return isource;
    }
    Float* getCorr() {
      return corr;
    }
    void contractMesons(PLEGMA_Propagator<Float> &prop1,
			PLEGMA_Propagator<Float> &prop2, 
			int isource, CORR_SPACE CorrSpace);

    void contractBaryons(PLEGMA_Propagator<Float> &prop1,
			 PLEGMA_Propagator<Float> &prop2, 
			 int isource, CORR_SPACE CorrSpace);

    void contractNucleonThrp(PLEGMA_Propagator<Float> &bwdProp,
			     PLEGMA_Propagator<Float> &fwdProp,
			     int signProps, std::vector<GAMMAS> gammas,
			     int isource, CORR_SPACE corrSpace);

    void writeFile(PLEGMA_params &params);
    void writeFile(char *filename, PLEGMA_params &params);
    void writeASCII(char *filename);
    void writeHDF5(char *filename, PLEGMA_params &params);
  };
}

#endif
