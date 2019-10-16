#pragma once
#include <PLEGMA_global.h>
#include <PLEGMA_io.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_FT.h>

namespace plegma {

  // forward declaration
  template<typename Float>  class PLEGMA_Vector;
  template<typename Float>  class PLEGMA_Propagator;
  template<typename Float>  class PLEGMA_Propagator3D;

  //////////////////////////////
  // CLASS: PLEGMA_Correlator //
  ////////////////////////////// 

  template<typename Float>
  class PLEGMA_Correlator : public IO<void> {
  protected:
    // Allocation
    bool isAlloc;
    PLEGMA_Field<Float>* corr_pos_space;
    PLEGMA_FT<Float>* corr_mom_space;
    Float* corr;

    // Correlator info
    CORR_SPACE corr_space;
    int Q2_max;
    std::vector<int> fixMomVec ;
    size_t vol_size;
    std::vector<int> shape;
    // Allocated site_size = n_datasets * n_groups * prod(shape) (slowest to fastest running index)
    int site_size;
    std::array<int,4> source_position;
    int maxT;

    // Writing informations
    std::vector<std::string> datasets;
    std::vector<std::string> groups;
    std::string description;

    void initialize();
    void finalize();
    
    // For HDF5 file writing
    std::string fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start);
    
  public:
    PLEGMA_Correlator(CORR_SPACE CorrSpace, int Q2_max):
      isAlloc(false),corr_pos_space(NULL), corr_mom_space(NULL), corr(NULL), corr_space(CorrSpace),
      Q2_max(Q2_max)
    {}
    PLEGMA_Correlator(CORR_SPACE CorrSpace, std::vector<int> fixMomVec):
      isAlloc(false),corr_pos_space(NULL), corr_mom_space(NULL), corr(NULL), corr_space(CorrSpace),
      fixMomVec(fixMomVec)
    {}

    ~PLEGMA_Correlator(){finalize();}
    CORR_SPACE getCorrSpace() {
      return corr_space;
    }
    inline size_t n_datasets() {
      return MAX(1,datasets.size());
    }
    inline size_t n_groups() {
      return MAX(1,groups.size());
    }
    size_t getSiteSize() {
      size_t size=n_datasets()*n_groups();
      std::for_each(shape.begin(), shape.end(), [&] (int n) {size *= n;});
      return size;
    }
    size_t getVolSize() {
      return vol_size;
    }
    size_t getTotalSize() {
      return site_size*vol_size;
    }
    int getTSize() {
      // Returns the local T size accordingly to the time source and maxT
      int startT = (HGC_procPosition[DIM_T] * HGC_localL[DIM_T] - source_position[DIM_T] +
		    HGC_totalL[DIM_T]) % HGC_totalL[DIM_T];
      if(startT>=maxT) return 0;
      else return MIN(maxT-startT, HGC_localL[DIM_T]);
    }
    int3 getSource3() {
      int3 source;
      source.x = source_position[0];
      source.y = source_position[1];
      source.z = source_position[2];
      return source;
    }
    std::array<int,4> getSource() {
      return source_position;
    }
    void setSource(int source[4]) {
      for ( int i = 0; i < 4; i++ )
	source_position[i] = source[i];
    }

    tex_mom_list getTexMomList() {
      if(corr_space == MOMENTUM_SPACE) {
	return corr_mom_space->getTexMomList();
      } else {
	tex_mom_list dummy;
	dummy.Nmoms=0;
	return dummy;
      }
    }
    Float* getCorr() {
      return corr;
    }
    std::vector<std::string> getDatasets() {
      return datasets;
    }
    void setDatasets(std::vector<std::string> d) {
      if (datasets.size() == d.size() || datasets.size() == 0) {
	datasets = d;
      } else {
	PLEGMA_error("Given vector size do not match. This would change the correlator size.");
      }
    }
    void setDatasets(std::string s) {
      return setDatasets({s});
    }
    std::vector<std::string> getGroups() {
      return groups;
    }
    void setGroups(std::vector<std::string> d) {
      if (groups.size() == d.size() || groups.size() == 0) {
	groups = d;
      } else {
	PLEGMA_error("Given vector size do not match. This would change the correlator size.");
      }
    }
    void setGroups(std::string s) {
      std::vector<std::string> d = {s};
      setGroups(d);
    }
    void contractMesons(PLEGMA_Propagator<Float> &prop1,
			PLEGMA_Propagator<Float> &prop2, 
			int source[4], int max_t=HGC_totalL[DIM_T]);

    
    void contractBaryons(PLEGMA_Propagator<Float> &prop1,
			 PLEGMA_Propagator<Float> &prop2, 
			 int source[4], int max_t=HGC_totalL[DIM_T]);
    
    void contractBaryonsUDSC(PLEGMA_Propagator<Float> &propUP,
			     PLEGMA_Propagator<Float> &propDN, 
			     PLEGMA_Propagator<Float> &propST, 
			     PLEGMA_Propagator<Float> &propCH, 
			     int source[4], int max_t=HGC_totalL[DIM_T],
			     bool only_st=false, bool only_ch=false);
    
    void contractNucleonThrp_local(PLEGMA_Propagator<Float> &bwdProp,
				   PLEGMA_Propagator<Float> &fwdProp,
				   int signProps, std::vector<GAMMAS> gammas,
				   int source[4], int max_t=HGC_totalL[DIM_T]);
    
    void contractNucleonThrp_oneD(PLEGMA_Propagator<Float> &bwdProp,
				  PLEGMA_Propagator<Float> &fwdProp,
				  PLEGMA_Gauge<Float> &gauge,
				  int signProps, std::vector<GAMMAS> gammas,
				  int source[4], int max_t=HGC_totalL[DIM_T]);
    
    void contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
				 PLEGMA_Propagator<Float> &fwdProp,
				 PLEGMA_Gauge<Float> &gauge,
				 int signProps, int source[4], int max_t=HGC_totalL[DIM_T]);

    void contractNucleonThrp_wilsonLine(PLEGMA_Propagator<Float> &bwdProp,
					PLEGMA_Propagator<Float> &fwdProp,
					PLEGMA_Su3field<Float> &su3,
					int signProps, std::vector<GAMMAS> gammas,
					int source[4], int max_t=HGC_totalL[DIM_T]);

    virtual void writeASCII(std::string filename);
    virtual void writeHDF5(std::string filename);
  };
}
