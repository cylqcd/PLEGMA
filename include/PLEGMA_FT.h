#include <PLEGMA_Field.h>
#pragma once

namespace plegma {
  /////////////////
  // This Class will be reponsible to momentum transfer fields
  //and write data in ASCII and HDF5 
  ////////////////

  template<typename Float>
  class PLEGMA_FT {
    using Vint = std::vector<int>;
    using VVint = std::vector<Vint>;
  private:
    int Q2_max;
    VVint momList;
    bool isAllocated;
    int dof; // degrees of freedom the field has
    Float *h_elem; // memory to hold the transformed data
    int dims; // the dimensionality of the transformation either 3 or 4
    int dimT; // if dims = 3, dimT = (dims ==3) ? GK_localL[3] : 1; 
    bool accum;
    void createMom();
    void checkAllocation(int newDof);
    void zero();
  public:
    PLEGMA_FT(int Q2_max, int D3D4 = 3, bool accum = false); // allow also for a transformation in 4D
    ~PLEGMA_FT();
    int Nmoms() const{ return momList.size();}
    VVint MomList() const{ return momList;}
    int Dims() const{return dims;}
    int DimT() const{return dimT;}
    Float* H_elem() const{return h_elem;}
    void writeToFile(std::string filename, FILE_WRITE_FORMAT outputFormat, int timeshift = 0);
    void applyNaive(const PLEGMA_Field<Float> &f, int sign=-1); // naive transformation using a simple custom kernel for reduction
    void apply(const PLEGMA_Field<Float> &f, int sign=-1);      // transformation using THRUST for the momentum field and cuBLAS for reduction
    void applyFFT(const PLEGMA_Field<Float> &f, int sign=-1);  // use FFT in case in the future is implemented
    void mulMomentumPhases(Vint src, int sign); // put momentum phases due to the point sources
  };
}
