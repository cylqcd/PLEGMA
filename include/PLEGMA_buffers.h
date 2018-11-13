#ifndef _PLEGMA_BUFFER_H
#define _PLEGMA_BUFFER_H

#include <PLEGMA_global.h>
#include <stdlib.h>
#include <malloc.h>

#include <new> // for availability of std::bad_alloc exception

namespace plegma {

  static inline void * BUFFER_MALLOC(size_t alignment, size_t size){
#ifdef PLEGMA_HAVE_MEMALIGN
    return( memalign(alignment, size) );
#else
    return( malloc(size) );
#endif
  }

  // depending on the architecture it might be necessary to free
  // aligned memory with something other than free, so let's 
  // foresee the possibility
  static inline void BUFFER_FREE(void* memptr){
    free(memptr);
  }

template<typename T>
class GaugeBuffer {
  public:
    GaugeBuffer(PLEGMA_params & params) {
      int *lL = params.lL;
      size_t V = lL[0];
      for(int i = 1; i < N_DIMS; ++i){
        V *= lL[i];
      }
      for(int i = 0; i < N_DIMS; ++i){
        buffer[i] = static_cast<T*>(NULL);
        buffer[i] = static_cast<T*>(BUFFER_MALLOC(PLEGMA_ALIGNMENT, V*gaugeSiteSize*sizeof(T))); 
        if(buffer[N_DIMS] == static_cast<T*>(NULL) ){
          throw( std::bad_alloc() );
        }
      }
    }

    T** get_ptr(void){
      return(buffer);
    }

    // no copy constructor to avoid implementing deep copy stuff
    GaugeBuffer(const GaugeBuffer & other) = delete;
    // no default constructor
    GaugeBuffer(void) = delete;

    ~GaugeBuffer() {
      for(int i = 0; i < N_DIMS; ++i){
        BUFFER_FREE(buffer[i]);
      }
    }
  
  private:
    T* buffer[N_DIMS];
};

}

#endif
