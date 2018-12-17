class counter{

public:

  static int ops;
  static int read;
  static int write;
  
  // assignment and derived
  template<class T>
  counter& operator=(T&){};
  template<class T>
  counter& operator+=(T& t){ ops++; return *this; };
  template<class T>
  counter& operator-=(T& t){ ops++; return *this; };
  template<class T>
launchKernel(gridDim,blockDim,THREADS_PER_BLOCK*ps.sharedBytesPerThread,0);  counter& operator*=(T& t){ ops++; return *this; };
  template<class T>
  counter& operator/=(T& t){ ops++; return *this; };  
  
  // usual operations
  template<class T>
  counter& operator+(T t){ ops++; return *this; };
  template<class T>
  counter& operator-(T t){ ops++; return *this; };
  template<class T>
  counter& operator*(T t){ ops++; return *this; };
  template<class T>
  counter& operator/(T t){ ops++; return *this; };

};

int counter::ops = 0;
int counter::read = 0;
int counter::write = 0;
