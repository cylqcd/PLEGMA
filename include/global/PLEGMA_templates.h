#pragma once
//======== Templated types and functions =========//

// MPI_Type(): return the MPI type to use in MPI reductions
template<typename Float> inline MPI_Datatype MPI_Type();
template<typename Float> inline MPI_Datatype MPI_Type(Float a){ return MPI_Type<Float>(); }
template<> inline MPI_Datatype MPI_Type<float>() { return MPI_FLOAT; }
template<> inline MPI_Datatype MPI_Type<float*>() { return MPI_FLOAT; }
template<> inline MPI_Datatype MPI_Type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype MPI_Type<double*>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype MPI_Type<int>() { return MPI_INT; }
template<> inline MPI_Datatype MPI_Type<int*>() { return MPI_INT; }
template<> inline MPI_Datatype MPI_Type<unsigned int>() { return MPI_UNSIGNED; }
template<> inline MPI_Datatype MPI_Type<long>() { return MPI_LONG; }
template<> inline MPI_Datatype MPI_Type<unsigned long>() { return MPI_UNSIGNED_LONG; }
template<> inline MPI_Datatype MPI_Type<long long>() { return MPI_LONG_LONG_INT; }

// hostMalloc and hostFree: functions to use in replace of malloc and free
extern long int HGC_used_memory;
template<typename T> inline void hostMalloc(T &ptr, size_t size) {
#ifdef PLEGMA_HAVE_MEMALIGN
  ptr = static_cast<T>(memalign(PLEGMA_ALIGNMENT, size));
#else
  ptr = static_cast<T>(malloc(size));
#endif
  if(ptr == static_cast<T>(NULL) ){
    PLEGMA_warning("Bad alloc. Total memory in use: %lu\n", HGC_used_memory);
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
  PLEGMA_warning("Freeing without providing the size. This will create a mismatch in HGC_used_memory.\n");
}

// Pinned memory allocation
template<typename T> inline void hostMallocPinned(T &ptr, size_t size){ 
  ptr=(T* )pinned_malloc(size);
  //cudaError_t err = cudaMallocHost((void**)&ptr, size);
  //if (err != cudaSuccess) {
  //  errorQuda("Failed to allocate host memory of size %zu \n", size);
  //}
  HGC_used_memory += size;
}

template<typename T> inline void hostFreePinned(T &ptr, size_t size) {
  //cudaError_t err = cudaFreeHost(ptr);
  //if (err != cudaSuccess) {
  //  errorQuda("Failed to free host memory of size %zu \n", size);
  //}
  host_free(ptr);
  //ptr=NULL;
  HGC_used_memory -= size;
}
template<typename T> inline void hostFreePinned(T &ptr) {
  hostFreePinned(ptr,0);
  PLEGMA_warning("Freeing without providing the size. This will create a mismatch in HGC_used_memory.\n");
}

template<typename T> inline void hostReAlloc(T &ptr_new, size_t size_new, T &ptr_old, size_t size_old){
  if(size_new == size_old) PLEGMA_warning("Reallocation of memory when old size is the same as the new is strange");
  ptr_new = static_cast<T>(realloc(ptr_old, size_new));
  if(ptr_new == static_cast<T>(NULL) ){
    fprintf(stderr,"Cannot reallocate memory of size %lu\n",size_new);
    exit(-1);
  }
  HGC_used_memory += size_new-size_old;  
}

// type_char(): identifying char for the variable. It is later used in type_print()
template<typename T> inline char type_char(){ return 'B';};
template<typename T> inline char type_char(T a){ return type_char<T>();};
template<> inline char type_char<int>() { return 'i'; }
template<> inline char type_char<long>() { return 'i'; }
template<> inline char type_char<long long>() { return 'i'; }
template<> inline char type_char<unsigned>() { return 'u'; }
template<> inline char type_char<unsigned long>() { return 'u'; }
template<> inline char type_char<unsigned long long>() { return 'u'; }
template<> inline char type_char<float>() { return 'f'; }
template<> inline char type_char<double>() { return 'f'; }
template<> inline char type_char<long double>() { return 'f'; }
template<> inline char type_char<char>() { return 'c'; }
template<> inline char type_char<char*>() { return 's'; }
template<> inline char type_char<void*>() { return 'p'; }
template<> inline char type_char<bool>() { return 'b'; }

inline std::string type_print(void* ptr, char type, int bytes) {
  switch(type) {
  case 'i':
    switch(bytes) {
    case 4:
      return std::to_string(*((int*)ptr));
    case 8:
      return std::to_string(*((long*)ptr));
    case 16:
      return std::to_string(*((long long*)ptr));
    default:
      goto end;
    }
  case 'u':
    switch(bytes) {
    case 4:
      return std::to_string(*((unsigned int*)ptr));
    case 8:
      return std::to_string(*((unsigned long*)ptr));
    case 16:
      return std::to_string(*((unsigned long long*)ptr));
    default:
      goto end;
    }
  case 'f':
    switch(bytes) {
    case 4:
      return std::to_string(*((float*)ptr));
    case 8:
      return std::to_string(*((double*)ptr));
    case 16:
      return std::to_string(*((long double*)ptr));
    default:
      goto end;
    }
  case 'c':
    return std::string(1,*((char*) ptr));
  case 's':
    return (char*) ptr;
  case 'b':
    if(*((bool*) ptr)) return "True";
    else return "False";
  default:
  end:
    constexpr char hexmap[] = {'0', '1', '2', '3', '4', '5', '6', '7',
			       '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string s(bytes * 2, ' ');
    for(int i=0; i<bytes; i++) {
      s[2 * i]     = hexmap[(((char*)ptr)[i] & 0xF0) >> 4];
      s[2 * i + 1] = hexmap[((char*)ptr)[i] & 0x0F];
    }
    return "0x"+s;
  }
}

inline std::string toString(){return "";}

template<typename T, typename... Pars>
inline std::string toString(T & p1, Pars & ... pars){
  std::stringstream cs;
  cs << std::setprecision(8);
  cs << " " << p1;
  return cs.str() + toString(pars...);
}

template<typename T>
inline std::string toString(std::vector<T> &vec){
  std::stringstream cs;
  cs << std::setprecision(8);
  for(T i : vec) cs << " " << i;
  return cs.str();
}

template<typename T1, typename T2>
inline std::string toString(std::map<T1,T2> &tpl){
  std::stringstream cs;
  cs << std::setprecision(8);
  typename std::map<T1,T2>::iterator it_b = tpl.begin();
  while(it_b != tpl.end()){
    cs << " (" <<it_b->first << ", " << it_b->second << ")";
    it_b++;
    }
  return cs.str();
}

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
