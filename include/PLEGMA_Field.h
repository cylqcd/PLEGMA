#include <PLEGMA_global.h>
#include <PLEGMA_Random.h>
#include <PLEGMA_Hprobing.h>
#include <PLEGMA_io.h>
#include <vector>
#ifndef _PLEGMA_FIELD_H
#define _PLEGMA_FIELD_H

namespace plegma {
  template<typename Float>  class PLEGMA_Field3D;
  template<typename Float>  class PLEGMA_Fmunu;
  template<typename Float>  class PLEGMA_Su3field;
  template<typename Float>  class PLEGMA_Gauge;
  

  ////////////////////////
  // CLASS: PLEGMA_Field //
  ////////////////////////
  
  template<typename Float>
  class PLEGMA_Field : virtual public IO<void,bool> {
  protected:
    
    int field_length;
    size_t total_length;        
    size_t ghost_length;
    size_t ghost_corner_length;
    size_t ghost_vertex_length;
    
    Float *h_elem;
    Float *d_elem;
    Float *h_ext_ghost_r;
    Float *h_ext_ghost_s;
    Float *h_ext_ghost_corner_r;
    Float *h_ext_ghost_corner_s;
    Float *h_ext_ghost_vertex_r;
    Float *h_ext_ghost_vertex_s;
    PLEGMA_RNG *randstate_ptr;
    
    GHOST_FLAG ghost_flag;
    ALLOCATION_FLAG allocation;
    bool isPinnedHost;
    bool isAllocHost;
    bool isAllocDevice;
    bool checkErr;
    std::vector<MsgHandle*> messages;

    CLASS_ENUM field_type;
    std::string field_name;
    std::vector<int> site_shape;
    
    void create_host();
    void destroy_host();
    void create_device();
    void destroy_device();
    void initialize(ALLOCATION_FLAG alloc_flag, int field_l, size_t vol_l);
  public:
    PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true);
    PLEGMA_Field(ALLOCATION_FLAG alloc_flag, int site_size, size_t localV = HGC_localVolume, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true);
    ~PLEGMA_Field();
    // Deleting copy contructor at the moment. This would cause seg fault due to the fields allocated
    PLEGMA_Field(const PLEGMA_Field<Float>&) = delete;
    void zero_host();
    void zero_host_backup();
    void zero_device();
    void zero_where(ALLOCATION_FLAG alloc_flag);
    cudaTextureObject_t createTexObject() const;
    void destroyTexObject(cudaTextureObject_t tex) const;
    
    Float* H_elem() const { return h_elem; }
    Float* D_elem() const { return d_elem; }

    bool IsAllocHost() const { return isAllocHost;}
    bool IsAllocDevice() const { return isAllocDevice;}

    ALLOCATION_FLAG getAllocation() const { return allocation; }

    int Field_length() const { return field_length;} // degrees of freedom per lattice point
    size_t Total_length() const { return total_length;} // the length of the field (local)
    size_t Ghost_length() const { return ghost_length;} // the length of the ghost
    size_t GhostCorner_length() const { return ghost_corner_length;} // the length of the ghost for corners
    size_t GhostVertex_length() const { return ghost_vertex_length;} // the length of the ghost for vertex
    size_t TotalPlusGhost_length() const { return Total_length()+Ghost_length()+GhostCorner_length()+GhostVertex_length();} // total + ghost

    size_t Bytes_total() const { return this->Total_length()*this->Field_length()*2*sizeof(Float); }
    size_t Bytes_ghost() const { return this->Ghost_length()*this->Field_length()*2*sizeof(Float); }
    size_t Bytes_ghostCorner() const { return this->GhostCorner_length()*this->Field_length()*2*sizeof(Float); }
    size_t Bytes_ghostVertex() const { return this->GhostVertex_length()*this->Field_length()*2*sizeof(Float); }
    size_t Bytes_total_plus_ghost() const { return this->TotalPlusGhost_length()*this->Field_length()*2*sizeof(Float); }

    template<class... Args>
    bool checkVolume(const Args&... fields) {
      std::vector<bool> checks = {this->Total_length() == fields.Total_length() ...};
      return std::all_of(checks.begin(), checks.end(), [](bool i){return i;});
    }
    
    std::string Field_name() const {return field_name;}
    GHOST_FLAG Ghost_flag() const {return ghost_flag;}
    int Precision() const{
      if( typeid(Float) == typeid(float) )
	return 4;
      else if( typeid(Float) == typeid(double) )
	return 8;
      else
	return 0;
    } 
    void printInfo();
    void communicateSideGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, ACTION action=DO_ALL);
    void communicateCornerGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, ACTION action=DO_ALL);
    void communicateVertexGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, ACTION action=DO_ALL);
    void communicateGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, GHOST_FLAG which_ghost=ALL_GHOSTS, ACTION action=DO_ALL);

    std::vector<int> getSiteShape() const {return site_shape;}
    void setSiteShape(std::vector<int> new_shape) {
      int current_size=1, new_size = 1;
      std::for_each(site_shape.begin(), site_shape.end(), [&] (int n) {current_size *= n;});
      std::for_each(new_shape.begin(), new_shape.end(), [&] (int n) {new_size *= n;});
      assert(current_size==new_size);
      site_shape = new_shape;
    }
    std::string fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start) const;
    
    void pack(Float *topack);
    void unpack(Float *out);

    void load();
    void unload() const;
    
    void shift(PLEGMA_Field &Fin, short dirOr);
    void shift(PLEGMA_Field &Fin, short dirOr1, short dirOr2);
    void shift(PLEGMA_Field &Fin, short dirOr1, short dirOr2, short dirOr3);
    void randInit(int seed);
    void destroy_randstate();
    void stochastic_Z(int n=2);
    void random(DIST sampling=Uniform);
    void setUnit(std::vector<int> indDiag);

    void conjugate();

    template<typename FloatIn>
    void copy(PLEGMA_Field<FloatIn> &f, ALLOCATION_FLAG where=DEVICE);

    template<typename T>
    void mulMomentumPhases(std::vector<T> mom, int sign=-1);

    // F += a*Fin
    void add(PLEGMA_Field &Fin, std::complex<Float> alpha = 1.);
    std::complex<Float> dot(PLEGMA_Field<Float> &FieldIn);    
    Float norm();
    void scale(Float val);
    void cscale(std::complex<Float> val);
    
    void applyHpropColoring4D(PLEGMA_Field<Float> &fin,PLEGMA_Hprobing &hprob, int ih, std::vector<int> indDof);

    void absorbTimeslice(PLEGMA_Field<Float> &srcfield, int global_it, bool forcetozero=true);

    void TrFmunuSu3FmunuSu3(PLEGMA_Fmunu<Float> &Fl, std::pair<int,int> munu_l, PLEGMA_Su3field<Float> &Wl,
			    PLEGMA_Fmunu<Float> &Fr, std::pair<int,int> munu_r, PLEGMA_Su3field<Float> &Wr);

    void trPmunu(PLEGMA_Gauge<Float> &gauge, std::pair<int,int> munu);

    /**
       @brief Absorbs all  elements from a 3D field and puts it at a specific global time of the 4D field
       @param PLEGMA_Field3D<Float> prop, The 3D field
       @param int global_it, The global time slice where data which will be inserted, the rest of the time-slices will become zero in the 4D field
       @return void
     **/    
    void absorb(const PLEGMA_Field3D<Float> &field, int global_it);
    /**
       @brief Multiplies a field with theta twists in temporal direction, namely e^{i \theta \pi t/T}
       @param double theta: the parameter \theta as used above
       @param bool dagger: If true flips the sign in the exponential
     **/
    void mulThetaPhase(Float theta, bool dagger=false);

    virtual void readLIME(std::string filename, bool loadToDev=true);
    virtual void writeLIME(std::string filename, bool unloadFromDev=true) const;
    virtual void writeHDF5(std::string filename, bool unloadFromDev=true) const;
    void readFile(std::string filename, FILE_FORMAT format, bool loadToDev=true) {
      IO<void,bool>::readFile(filename, format, loadToDev);
    }
    void readFile(std::string filename, bool loadToDev=true) {
      IO<void,bool>::readFile(filename, loadToDev);
    }
    void writeFile(std::string filename, FILE_FORMAT format, bool unloadFromDev=true) const {
      IO<void,bool>::writeFile(filename, format, unloadFromDev);
    }
    void writeFile(std::string filename, bool unloadFromDev=true) const{
      IO<void,bool>::writeFile(filename, unloadFromDev);
    }

    virtual bool includesActiveTimeSlice() const{return true;}
    virtual bool is4D() const{assert(Total_length()==HGC_localVolume); return true;}
  };

  template<typename Float>
  class PLEGMA_Field3D : virtual public PLEGMA_Field<Float> {
  public:
    bool activeTimeSlice;
    
    PLEGMA_Field3D(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true) :
      PLEGMA_Field<Float>(alloc_flag, classT, ghost_flag, isPinnedHost, checkErr), activeTimeSlice(false) { }
    PLEGMA_Field3D(ALLOCATION_FLAG alloc_flag, int site_size, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true) :
      PLEGMA_Field<Float>(alloc_flag, site_size, HGC_localVolume3D, ghost_flag, isPinnedHost, checkErr), activeTimeSlice(false) { }
    PLEGMA_Field3D() : PLEGMA_Field<Float>(NONE, 0, HGC_localVolume3D), activeTimeSlice(false) { }

    virtual bool includesActiveTimeSlice() const{return activeTimeSlice;}
    virtual bool is4D() const{assert(this->Total_length()==HGC_localVolume3D); return false;}

    template<typename FloatIn>
    void copy(PLEGMA_Field3D<FloatIn> &f, ALLOCATION_FLAG where=DEVICE) {
      this->activeTimeSlice = f.activeTimeSlice;
      return ((PLEGMA_Field<Float>*) this)->copy(f,where);
    }
    
    /**
       @brief Absorbs a time-slice from a 4D field to a 3D field
       @param PLEGMA_Field<Float> field, The 4D field
       @param int global_it, The global time slice from where data will be extracted from the the 4D field
       @return void
     **/    
    void absorb(const PLEGMA_Field<Float> &field, int global_it);
  };
}
#endif
