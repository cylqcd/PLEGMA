#pragma once

namespace plegma {
  inline int getVecToInd(std::vector<int> x, std::vector<int> L){
    if(x.size() != L.size()) errorQuda("Dimensions do not match");
    if(x.size() == 0)errorQuda("Size of the vector is zero");
    int D=x.size();
    for(int i = 0 ; i < D; i++)
      if(x[i] >= L[i])
	errorQuda("Error the position of the vector exceeds the extent of dimension %d", i);
    int acc=x[D-1];
    for(int i = D-2 ; i >= 0; i--) acc = acc*L[i] + x[i];
    return acc;
  }

  inline std::vector<int> getIndToVec(int ind, std::vector<int> L){
    int V=1;
    std::vector<int> x;
    if(L.size() == 0)errorQuda("Size of the vector is zero");
    if(ind < 0 )errorQuda("Ind provided is negative");
    int D = L.size();
    for(int i = 0 ; i < D-1; i++ ) V *= L[i];
    if(V<0) errorQuda("The volume is negative which is not allowed");
    if(V==0) errorQuda("One or more directions are zero");
    if(ind >= V*L[D-1]) errorQuda("The ind exceeds the total volume");
    int sub=0;
    for(int i = D-1; i >= 0; i--){
      ind -= sub;
      x.insert(x.begin(), ind/V);
      if(V==1) break;
      sub=x[0]*V;
      V /= L[i-1];
    }
    return x;
  }


  /* ================ Small introduction to Hierarchical probing ============
     # There is an unsigned integer "k" running from 1 until ...
     # From this integer we can specify several important quantities regarding the coloring
     # The total number of colors is given by N_{hc} = 2 * 2^{d(k-1)} where d is the number of dimensions
     # The distance seperating neighbors carrying the same color is D=2^k
     # The extent of the elementary coloring block is given L_u=2^{k-1}
     # A condition must be fulfilled in order to be able to do the coloring for a specific k
     # The condition must be that the number of blocks in each direction must be even
     # And that Ls%(2*Lu)=0 and Lt%(2*Lu)=0
  */

  class PLEGMA_Hprobing{
  private:
    int k;  // index for the coloring distance
    int Nc; // Number of colors = Number of Hadamard vectors
    short d; // Number of dimension of Hprob (For now d=4)
    short D; // Distance of coloring D=2^k
    short Lu; // extent of the elementaty coloring block (assume symmetric block)
    int* h_arrVc; // array to hold the coloring of the lattice on HOST
    int* d_arrVc; // array to hold the coloring of the lattice on Device
    int* arrlc; // array to hold the elementary coloring block
    void createElemColBlock(){for(int i = 0; i < Nc; i++) arrlc[i]=i;}
    void createColLattice(){
      std::vector<int> lL = {HGC_localL[0], HGC_localL[1], HGC_localL[2], HGC_localL[3]};
      std::vector<int> lu = {Lu,Lu,Lu,Lu};
      std::vector<int> bx(d);
      std::vector<int> lx(d);
      for(int i=0; i < HGC_localVolume; i++){
	std::vector<int> x = getIndToVec(i,lL);
	for(int j = 0 ; j < d; j++) bx[j] = x[j]/lu[j];
	int eo=0;
	for(int j = 0 ; j < d; j++) eo += bx[j];
	eo=eo & 1;
	for(int j = 0 ; j < d; j++)  lx[j] = x[j] - lu[j] * bx[j];
	h_arrVc[i] = arrlc[getVecToInd(lx,lu)*2+eo];
      }
    }
    //  void checkColoring();
  public:
    PLEGMA_Hprobing(int k_probing, int d=4):k(k_probing),Nc(0),d(d),D(0),Lu(0),h_arrVc(nullptr),d_arrVc(nullptr),arrlc(nullptr){
      if(!HGC_init_PLEGMA_flag){ fprintf(stderr, "Error PLEGMA should be initialized before use this class"); exit(-1);}
      if(d != 4) errorQuda("Hierarchical probing supports only 4D coloring up to now");
      if(k<=0) errorQuda("The index of the Hprobing should greater than zero");
      Nc = 2*std::pow(2,d*(k-1));
      D = std::pow(2,k);
      Lu = std::pow(2,k-1);
      printfQuda("Number of colors for hierarchical probing is %d\n",Nc);
      printfQuda("Distance of neigbors is %d\n",D);
      printfQuda("The extent of the elementary symmetric color block is %d\n",Lu);
      for(int i = 0 ; i < d ; i++){
	if(D >= HGC_localL[i]) errorQuda("The coloring distance is larger than the lattice extent in direction %d\n",i);
	if( (HGC_localL[i] % (2*Lu)) != 0 )
	  errorQuda("2*Lu cannot fit in the local lattice extent in direction %d. Try to increase local size in this direction",i);
      }
      try{
	h_arrVc = new int[HGC_localVolume];
	arrlc = new int[Nc];
      }
      catch (const std::bad_alloc& err) {
	errorQuda(err.what());
      }
      createElemColBlock();
      createColLattice();
      //  if(check)checkColoring();
      cudaMalloc((void**)&d_arrVc, HGC_localVolume*sizeof(int));
      checkCudaError();
      cudaMemcpy(d_arrVc, h_arrVc, HGC_localVolume*sizeof(int), cudaMemcpyHostToDevice);
      checkCudaError();    
    }
    ~PLEGMA_Hprobing(){
      delete[] h_arrVc;
      delete[] arrlc;
      cudaFree(d_arrVc);
      checkCudaError();
    }
    int* H_arrVc() const{return h_arrVc;}
    int* D_arrVc() const{return d_arrVc;}
    int get_NHad() const{return Nc;}
  };
}
/* inline static int boundaryCheck(int x, int L){ */
/*   int y=x; */
/*   if (y >= L) */
/*     y=y%L; */
/*   if (y < 0) */
/*     y=y+L; */
/*   return y; */
/* } */

/* void Hprobing::checkColoring(){ */
/*   if(HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3] != 1) errorQuda("The coloring check works only with 1 MPI task"); */
/*   std::vector<int> lL = {HGC_localL[0], HGC_localL[1], HGC_localL[2], HGC_localL[3]}; */
/*   for(int t = 0 ; t < HGC_totalL[3] ; t++) */
/*     for(int z = 0 ; z < HGC_totalL[2] ; z++) */
/*       for(int y = 0 ; y < HGC_totalL[1] ; y++) */
/* 	for(int x = 0 ; x < HGC_totalL[0] ; x++){ */
/* 	  std::vector<int> xx = {x,y,z,t}; */
/* 	  int c1 = h_arrVc[getVecToInd(xx, lL)]; */
/* 	  for(int dx = -D+1 ; dx < D ; dx++) */
/* 	    for(int dy = -D+1 ; dy < D ; dy++) */
/* 	      for(int dz = -D+1 ; dz < D ; dz++) */
/* 		for(int dt = -D+1 ; dt < D ; dt++){ */
/* 		  int ds = abs(dx) + abs(dy) + abs(dz) + abs(dt); */
/* 		  if ((ds<D) && (ds != 0)){ */
/* 		    int xn = x + dx; */
/* 		    xn = boundaryCheck(xn,HGC_totalL[0]); */
/* 		    int yn = y + dy; */
/* 		    yn = boundaryCheck(yn,HGC_totalL[1]); */
/* 		    int zn = z + dz; */
/* 		    zn = boundaryCheck(zn,HGC_totalL[2]); */
/* 		    int tn = t + dt; */
/* 		    tn = boundaryCheck(tn,HGC_totalL[3]); */
/* 		    xx[0] = xn; xx[1] = yn; xx[2] = zn; xx[3] = tn; */
/* 		    int c2 = h_arrVc[getVecToInd(xx, lL)]; */
/* 		    if(c1 == c2){ */
/* 		      printfQuda("Colors (%d,%d)\n",c1,c2); */
/* 		      errorQuda("Mistake found in the coloring with (%d,%d,%d,%d) and (%d,%d,%d,%d)",x,y,z,t,xn,yn,zn,tn); */
/* 		    } */
/* 		  } */
/* 		} */
/* 	} */
/*   printfQuda("Check in coloring passed successfully\n"); */
/* } */
