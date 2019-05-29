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
    std::vector<int> fixMomVec;
    VVint momList;
    bool isAllocated;
    int dof; // degrees of freedom the field has
    Float *h_elem; // memory to hold the transformed data
    int sizeN; // size of the elements array include real,imag
    int dims; // the dimensionality of the transformation either 3 or 4
    int dimT; // if dims = 3, dimT = (dims ==3) ? HGC_localL[3] : 1; 
    bool accum;
    void createMom();
    void zero();
  public:
    /**
       @brief Constructor of the FT class with max momentum value
       @params int Q2_max: Up to which momentum square we want to do the transformation
       @params int D3D4 = 3: The dimensionality of the FT, either 3 or 4 dimensions are supported
       @params bool accum = false: In case we want to accumulation results from each transformation on the class buffer
     **/
    PLEGMA_FT(int Q2_max, int D3D4 = 3, bool accum = false); 
    /**
       @brief Constructor of the FT class with specific momentum vector
       @params std::vector<int> mom: Momentum vector, either 3 or 4 components based on the choice of D3D4
       @params int D3D4 = 3: The dimensionality of the FT, either 3 or 4 dimensions are supported
       @params bool accum = false: In case we want to accumulation results from each transformation on the class buffer
     **/
    PLEGMA_FT(std::vector<int> mom, int D3D4 = 3, bool accum = false);
    

    ~PLEGMA_FT();
    void checkAllocation(int newDof);

    int Nmoms() const{ return momList.size();}
    VVint MomList() const{ return momList;}
    int Dims() const{return dims;}
    int DimT() const{return dimT;}
    
    Float* H_elem() const{return h_elem;}
    tex_mom_list getTexMomList();

    
    void applyNaive(const PLEGMA_Field<Float> &f, int sign=-1); // naive transformation using a simple custom kernel for reduction
    void apply(const PLEGMA_Field<Float> &f, int sign=-1);      // transformation using THRUST for the momentum field and cuBLAS for reduction
    void applyFFT(const PLEGMA_Field<Float> &f, int sign=-1);  // use FFT in case in the future is implemented
    
    void mulConstMomentumPhases(Vint src, int sign); // put momentum phases due to the point sources
    void scale(Float a);

    void writeToFile(std::string filename, FILE_WRITE_FORMAT outputFormat, int timeshift = 0);
};
}
