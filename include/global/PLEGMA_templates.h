//======== Templated types and functions =========//

// MPI_Type(): return the MPI type to use in MPI reductions
template<typename Float> inline MPI_Datatype MPI_Type();
template<typename Float> inline MPI_Datatype MPI_Type(Float a){ return MPI_Type<Float>(); }
template<> inline MPI_Datatype MPI_Type<float>() { return MPI_FLOAT; }
template<> inline MPI_Datatype MPI_Type<float*>() { return MPI_FLOAT; }
template<> inline MPI_Datatype MPI_Type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype MPI_Type<double*>() { return MPI_DOUBLE; }


// hostMalloc and hostFree: functions to use in replace of malloc and free
extern size_t HGC_used_memory;
template<typename T> inline void hostMalloc(T &ptr, size_t size) {
#ifdef PLEGMA_HAVE_MEMALIGN
  ptr = static_cast<T>(memalign(PLEGMA_ALIGNMENT, size));
#else
  ptr = static_cast<T>(malloc(size));
#endif
  if(ptr == static_cast<T>(NULL) ){
    warningQuda("Bad alloc. Total memory in use: %lu\n", HGC_used_memory);
    throw( std::bad_alloc() );
  }
  HGC_used_memory += sizeof(T)*size;
}
template<typename T> inline T* hostMalloc(size_t size) {
  T* ptr;
  hostMalloc(ptr, size);
  return ptr;
}
template<typename T> inline void hostFree(T &ptr, size_t size) {
  free(ptr);
  ptr=NULL;
  HGC_used_memory -= sizeof(T)*size;
}
template<typename T> inline void hostFree(T &ptr) {
  hostFree(ptr,0);
  warningQuda("Freeing without providing the size. This will create a mismatch in HGC_used_memory.\n");
}

// type_char(): return the preferred char used in printf for printing the variable
template<typename T> inline char type_char(){ return 'p';};
template<typename T> inline char type_char(T a){ return type_char<T>();};
template<> inline char type_char<char>() { return 'c'; }
template<> inline char type_char<int>() { return 'd'; }
template<> inline char type_char<float>() { return 'e'; }
template<> inline char type_char<double>() { return 'e'; }
template<> inline char type_char<char*>() { return 's'; }
template<> inline char type_char<const char*>() { return 's'; }
template<> inline char type_char<void*>() { return 'p'; }

// type_name(): return the demangled typename of a type
static inline std::string demangle( const char* mangled_name ) {
#ifdef __GNUG__ // gnu C++ compiler
  std::string result ;
  std::size_t len = 0 ;
    int status = 0 ;
    char* ptr = __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ) ;
    
    if( status == 0 ) result = ptr ; // hope that this won't throw
    else result = "demangle error" ;
    ::free(ptr) ;
    if(result.find("basic_string") != std::string::npos) return "std::string"; 
    return result ;
#else
    return std::string(mangled_name);
#endif
}
template<typename T> inline std::string type_name(){ return demangle(typeid(T).name());};
template<typename T> inline std::string type_name(T a){ return type_name<T>();};
