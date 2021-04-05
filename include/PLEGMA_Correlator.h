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
    // Correlator info
    const site source;
    const int totalT;
    std::vector<int> shape;
    std::vector<std::string> datasets;
    std::vector<std::string> groups;
    std::string description;

    // Correlator in pos/mom space
    const CORR_SPACE corr_space;
    std::shared_ptr<PLEGMA_Field<Float>> corr_pos_space;
    std::shared_ptr<PLEGMA_FT<Float>> corr_mom_space;
    std::shared_ptr<MPI_Comm> comm;

    void initialize();
    
    // For HDF5 file writing
    std::string fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start) const;
    
  public:
    bool hasSource(int dir) const {
      // Tells if the source is included in the local lattice for the given direction
      if(dir<0 || dir>N_DIMS) return false;
      return ((HGC_procPosition[dir]*HGC_localL[dir]) <= source[dir])
	&& (source[dir] < ((HGC_procPosition[dir]+1)*HGC_localL[dir]));
    }
    bool hasSource() const {
      bool ret=true;
      // Tells if the source is included in the local lattice
      for(int dir=0; ret && dir<N_DIMS; dir++) ret &= hasSource(dir);
      return ret;
    }
    int startT() const {
      // Returns the starting point in time of the correlator wrt the source.
      // Zero is returned if the local time slice is not used.
      int start=(HGC_procPosition[DIM_T] * HGC_localL[DIM_T] + HGC_totalL[DIM_T] - source[DIM_T] )
	% HGC_totalL[DIM_T];
      return (start>=totalT) ? 0 : start;
    }
    int endT() const {
      // Returns the end point
      int start = startT();
      if(start>0)
	return std::min(totalT, HGC_localL[DIM_T]+start);
      else
	return 0;
    }
    int localT() const {
      // Returns the local T size
      if(hasSource(DIM_T)) {
	// When the source is in the local lattice we may have two pieces:
	// |-->  s   | from startT to endT
	// |     s-->| from the source to the end
	int t_source = source[DIM_T]%HGC_localL[DIM_T];
	return endT() - startT() + std::min(totalT, HGC_localL[DIM_T]-t_source); 
      } else {
	return endT() - startT();
      }
    }
    
    PLEGMA_Correlator(CORR_SPACE corr_space, site source, int Q2_max = 0, int totalT=HGC_totalL[DIM_T]):
      source(source), totalT(totalT), corr_space(corr_space), corr_pos_space(nullptr),
      corr_mom_space(corr_space==MOMENTUM_SPACE ?
		     new PLEGMA_FT<Float>(Q2_max, 3, false, localT()) : nullptr),
      comm(new MPI_Comm(HGC_fullComm)) { }

    ~PLEGMA_Correlator() {}
    
    CORR_SPACE getCorrSpace() const {
      return corr_space;
    }
    Float* H_elem() const {
      if(corr_space == MOMENTUM_SPACE) {
	assert(corr_mom_space);
	return corr_mom_space->H_elem();
      } else {
	assert(corr_pos_space);
	return corr_pos_space->H_elem();
      }
    }
    site getSource() {
      return source;
    }
    int getTotalT() {
      return totalT;
    }
    size_t nDatasets() const {
      return std::max(datasets.size(), (size_t) 1);
    }
    size_t nGroups() const {
      return std::max(groups.size(), (size_t) 1);
    }
    size_t getSiteSize() const {
      // Allocated site_size = n_datasets * n_groups * prod(shape) (slowest to fastest running index)
      size_t size=nDatasets()*nGroups();
      for(auto n: shape) size *= n;
      return size;
    }
    size_t getVolSize() const {
      if(corr_space == MOMENTUM_SPACE) {
	assert(corr_mom_space);
	return corr_mom_space->Nmoms()*corr_mom_space->DimT();
      } else {
	assert(corr_pos_space);
	return corr_pos_space->Total_length();
      }
    }
    size_t getTotalSize() const {
      return getSiteSize()*getVolSize();
    }
    void setQ2max(int Q2_max) {
      assert(corr_space == MOMENTUM_SPACE);
      corr_mom_space.reset(new PLEGMA_FT<Float>(Q2_max, 3, false, localT()));
    }
    void setFixMomVec(std::vector<int>& fixMomVec) {
      assert(corr_space == MOMENTUM_SPACE);
      corr_mom_space.reset(new PLEGMA_FT<Float>(fixMomVec, 3, false, localT()));
    }
    void setFixMomList(std::vector<std::vector<int>>& fixMomList) {
      assert(corr_space == MOMENTUM_SPACE);
      corr_mom_space.reset(new PLEGMA_FT<Float>(fixMomList, 3, false, localT()));
    }
   
    std::vector<std::vector<int>> getMomList(){
      if(corr_space == MOMENTUM_SPACE) {
        std::vector<std::vector<Float>> list =  corr_mom_space->MomList();
        std::vector<std::vector<int>> casted;
        for(auto &mom: list) {
          casted.push_back(std::vector<int>());
          for(auto &imom: mom) {
            casted.back().push_back((int) std::lround(imom));
          }
        }
        return casted;
      } else {
        return {};
      }
    }

    std::shared_ptr<tex_mom_list> getTexMomList() const {
      if(corr_space == MOMENTUM_SPACE) {
	return corr_mom_space->getTexMomList();
      } else {
	return std::shared_ptr<tex_mom_list>(new tex_mom_list());
      }
    }
    std::vector<std::string> getDatasets() const{
      return datasets;
    }
    void setDatasets(std::vector<std::string> d) {
      if (datasets.size() == d.size() || (datasets.size() == 0 && d.size() == 1)) {
	datasets = d;
      } else {
	PLEGMA_error("Given vector size do not match. This would change the correlator size.");
      }
    }
    void setDatasets(std::string s) {
      return setDatasets({s});
    }
    std::vector<std::string> getGroups() const{
      return groups;
    }
    void setGroups(std::vector<std::string> d) {
      if (groups.size() == d.size() || (groups.size() == 0 && d.size() == 1)) {
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
			PLEGMA_Propagator<Float> &prop2);
    void contractMesonsNew(PLEGMA_Propagator<Float> &prop1,
			   PLEGMA_Propagator<Float> &prop2);
    void contractMesonsAll(PLEGMA_Propagator<Float> &prop1,
			   PLEGMA_Propagator<Float> &prop2);
    
    void contractBaryons(PLEGMA_Propagator<Float> &prop1,
			 PLEGMA_Propagator<Float> &prop2);
    
    void contractBaryonsUDSC(PLEGMA_Propagator<Float> &propUP,
			     PLEGMA_Propagator<Float> &propDN, 
			     PLEGMA_Propagator<Float> &propST, 
			     PLEGMA_Propagator<Float> &propCH,
			     bool only_st=false, bool only_ch=false);
    
    void contractNucleonThrp_local(PLEGMA_Propagator<Float> &bwdProp,
				   PLEGMA_Propagator<Float> &fwdProp,
				   int signProps, std::vector<GAMMAS> gammas, bool isZfac = false);
    
    void contractNucleonThrp_oneD(PLEGMA_Propagator<Float> &bwdProp,
				  PLEGMA_Propagator<Float> &fwdProp,
				  PLEGMA_Gauge<Float> &gauge,
				  int signProps, std::vector<GAMMAS> gammas, bool isZfac = false);
    
    void contractNucleonThrp_twoD(PLEGMA_Propagator<Float> &bwdProp,
				  PLEGMA_Propagator<Float> &fwdProp,
				  PLEGMA_Gauge<Float> &gauge,
				  int signProps, std::vector<GAMMAS> gammas, bool isZfac = false);
    
    void contractNucleonThrp_threeD(PLEGMA_Propagator<Float> &bwdProp,
				  PLEGMA_Propagator<Float> &fwdProp,
				  PLEGMA_Gauge<Float> &gauge,
				  int signProps, std::vector<GAMMAS> gammas, bool isZfac = false);
    
    void contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
				 PLEGMA_Propagator<Float> &fwdProp,
				 PLEGMA_Gauge<Float> &gauge, int signProps);

    void contractNucleonThrp_wilsonLine(PLEGMA_Propagator<Float> &bwdProp,
					PLEGMA_Propagator<Float> &fwdProp,
					PLEGMA_Su3field<Float> &su3,
					int signProps, std::vector<GAMMAS> gammas,
					int z, std::string quark);

    virtual void writeASCII(std::string filename) const;
    virtual void writeHDF5(std::string filename) const;
  };
}
