#ifndef PLEGMA_KERNEL_COUNTER_CUH
#define PLEGMA_KERNEL_COUNTER_CUH

class counter{

public:

  static size_t ops;
  static size_t reads;
  static size_t writes;

  // assignment and derived
  template<class T>
  counter& operator=(T t){ return *this; }
  template<class T>
  counter& operator+=(T& t){ ops++; return *this; }
  template<class T>
  counter& operator*=(T& t){ ops++; return *this; }
  template<class T>
  counter& operator/=(T& t){ ops++; return *this; }  
  
  // usual operations
  template<class T>
  counter& operator+(T t){ ops++; return *this; }
  template<class T>
  counter& operator-(T t){ ops++; return *this; }
  template<class T>
  counter& operator*(T t){ ops++; return *this; }
  template<class T>
  counter& operator/(T t){ ops++; return *this; }

  // increase methods
  static void incReads(){ reads++; }
  static void incWrites(){ writes++; }

  // get methods
  static size_t getOps(){ return ops; }
  static size_t getReads(){ return reads; }
  static size_t getWrites(){ return writes; }
  
};

#endif
