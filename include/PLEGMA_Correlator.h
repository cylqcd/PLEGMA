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
  class PLEGMA_Correlator {
  protected:
    // Allocation
    bool isAlloc;
    PLEGMA_Field<Float>* corr_pos_space;
    PLEGMA_FT<Float>* corr_mom_space;
    Float* corr;

    // Correlator info
    CORR_SPACE corr_space;
    int Q2_max;
    std::vector<int> sink_mom = {0,0,0};
    size_t vol_size;
    int n_datasets;
    int n_groups;
    std::vector<int> shape;
    // Allocated site_size = n_datasets * n_groups * prod(shape) (slowest to fastest running index)
    int site_size;
    std::array<int,4> source_position;

    // Writing informations
    std::vector<std::string> datasets;
    std::vector<std::string> groups;
    std::string description;

    void initialize();
    void finalize();
    
    // For HDF5 file writing
    std::string fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start);
    
  public:
    PLEGMA_Correlator(CORR_SPACE CorrSpace = MOMENTUM_SPACE, int Q2_max = 64):
      isAlloc(false),corr_pos_space(NULL), corr_mom_space(NULL), corr(NULL), corr_space(CorrSpace),
      Q2_max(Q2_max)
    {}
    PLEGMA_Correlator(CORR_SPACE CorrSpace = MOMENTUM_SPACE, std::vector<int> sink_mom={0,0,0}):
      isAlloc(false),corr_pos_space(NULL), corr_mom_space(NULL), corr(NULL), corr_space(CorrSpace),
      sink_mom(sink_mom)
    {}

    ~PLEGMA_Correlator(){finalize();}
    CORR_SPACE getCorrSpace() {
      return corr_space;
    }
    size_t getSiteSize() {
      int size=n_datasets*n_groups;
      std::for_each(shape.begin(), shape.end(), [&] (int n) {size *= n;});
      return size;
    }
    size_t getVolSize() {
      return vol_size;
    }
    size_t getTotalSize() {
      return site_size*vol_size;
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

    void setMom(std::vector<int> mom) {this->sink_mom=mom;}
    std::vector<int> getMom() {return this->sink_mom;}
    
    Float* getCorr() {
      return corr;
    }
    void contractMesons(PLEGMA_Propagator<Float> &prop1,
			PLEGMA_Propagator<Float> &prop2, 
			int source[4]);

    
    void contractBaryons(PLEGMA_Propagator<Float> &prop1,
			 PLEGMA_Propagator<Float> &prop2, 
			 int source[4]);

    void contractNucleonThrp_local(PLEGMA_Propagator<Float> &bwdProp,
				   PLEGMA_Propagator<Float> &fwdProp,
				   int signProps, std::vector<GAMMAS> gammas,
				   int source[4]);
    
    void contractNucleonThrp_oneD(PLEGMA_Propagator<Float> &bwdProp,
				  PLEGMA_Propagator<Float> &fwdProp,
				  PLEGMA_Gauge<Float> &gauge,
				  int signProps, std::vector<GAMMAS> gammas,
				  int source[4]);
    
    void contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
				 PLEGMA_Propagator<Float> &fwdProp,
				 PLEGMA_Gauge<Float> &gauge,
				 int signProps, int source[4]);

    void contractNucleonThrp_wilsonLine(PLEGMA_Propagator<Float> &bwdProp,
					PLEGMA_Propagator<Float> &fwdProp,
					PLEGMA_Su3field<Float> &su3,
					int signProps, std::vector<GAMMAS> gammas,
					int source[4]);

    void writeFile(const char *filename, FILE_WRITE_FORMAT format);
    void writeASCII(const char *filename);
    void writeHDF5(std::string filename, std::string top = "/");
  };
}
