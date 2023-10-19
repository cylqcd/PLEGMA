#pragma once
#include <io/PLEGMA_lime.h>
#include <io/PLEGMA_hdf5.h>

namespace plegma {
  template<class returnT, class ...argsT>
  class IO {
    FILE_FORMAT deduce_type(std::string filename) const{
      std::vector<std::string> hdf5{".h5", ".hdf5"};
      for( auto ext: hdf5 ) {
	size_t pos = filename.rfind(ext);
	if(pos + ext.length() == filename.length()) {
	  return HDF5_FORMAT;
	} else if(pos != std::string::npos && filename[pos + ext.length()] == '/') {
	  return HDF5_FORMAT;
	}
      }
      std::vector<std::string> ascii{".txt", ".dat", ".ascii", ".raw"};
      for( auto ext: ascii ) {
	size_t pos = filename.rfind(ext);
	if(pos + ext.length() == filename.length()) {
	  return ASCII_FORMAT;
	}
      }
      std::vector<std::string> lime{".lime", ".bin"};
      for( auto ext: lime ) {
	size_t pos = filename.rfind(ext);
	if(pos + ext.length() == filename.length()) {
	  return LIME_FORMAT;
	}
      }
      PLEGMA_warning("Deduce type of %s failed.", filename.c_str());
      return DEFAULT_FORMAT;
    }
  public:
    
    virtual returnT writeDEFAULT(std::string filename, argsT ... args) const {
      PLEGMA_error("No default writing defined\n");
      return returnT();
    }
    virtual returnT writeASCII(std::string filename, argsT ... args) const {
      PLEGMA_error("Writing in ASCII not supported. Trying default writing\n");
      return writeDEFAULT(filename, args...);
    }
    virtual returnT writeHDF5(std::string filename, argsT ... args) const {
      PLEGMA_error("Writing in ASCII not supported. Trying default writing\n");
      return writeDEFAULT(filename, args...);
    }
    virtual returnT writeLIME(std::string filename, argsT ... args) const {
      PLEGMA_error("Writing in ASCII not supported. Trying default writing\n");
      return writeDEFAULT(filename, args...);
    }
    returnT writeFile(std::string filename, FILE_FORMAT format, argsT ... args) const {
      switch (format) {
      case ASCII_FORMAT:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to write file %s in ASCII format\n",filename.c_str());
	return writeASCII(filename, args...);
      case HDF5_FORMAT:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to write file %s in HDF5 format\n",filename.c_str());
	return writeHDF5(filename, args...);
      case LIME_FORMAT:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to write file %s in LIME format\n",filename.c_str());
	return writeLIME(filename, args...);
      case DEFAULT_FORMAT:
      default:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to write file %s in DEFAULT format\n",filename.c_str());
	return writeDEFAULT(filename, args...);
      }
    }
    returnT writeFile(std::string filename, argsT ... args) const {
      return writeFile(filename, deduce_type(filename), args...);
    }


    virtual returnT readDEFAULT(std::string filename, argsT ... args) {
      PLEGMA_error("No default reading defined\n");
      return returnT();
    }
    virtual returnT readASCII(std::string filename, argsT ... args) {
      PLEGMA_error("Reading in ASCII not supported. Trying default reading\n");
      return readDEFAULT(filename, args...);
    }
    virtual returnT readHDF5(std::string filename, argsT ... args) {
      PLEGMA_error("Reading in ASCII not supported. Trying default reading\n");
      return readDEFAULT(filename, args...);
    }
    virtual returnT readLIME(std::string filename, argsT ... args) {
      PLEGMA_error("Reading in ASCII not supported. Trying default reading\n");
      return readDEFAULT(filename, args...);
    }
    returnT readFile(std::string filename, FILE_FORMAT format, argsT ... args) {
      switch (format) {
      case ASCII_FORMAT:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to read file %s in ASCII format\n",filename.c_str());
	return readASCII(filename, args...);
      case HDF5_FORMAT:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to read file %s in HDF5 format\n",filename.c_str());
	return readHDF5(filename, args...);
      case LIME_FORMAT:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to read file %s in LIME format\n",filename.c_str());
	return readLIME(filename, args...);
      case DEFAULT_FORMAT:
      default:
	if(HGC.verbosity > 1) PLEGMA_printf("Going to read file %s in DEFAULT format\n",filename.c_str());
	return readDEFAULT(filename, args...);
      }
    }
    returnT readFile(std::string filename, argsT ... args) {
      return readFile(filename, deduce_type(filename), args...);
    }

  };
}
