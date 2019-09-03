#include <PLEGMA_Correlator.h>
#include <string>
#include <PLEGMA_mesons.cuh>
#include <PLEGMA_baryons.cuh>
#include <functional>
using namespace plegma;

//--------------------------------//
// class PLEGMA_Correlator //
//--------------------------------//

template<typename Float>
void PLEGMA_Correlator<Float>::
initialize() {
  if(isAlloc && site_size == getSiteSize())
    return;
  finalize();
  site_size = getSiteSize();
  if(corr_space == MOMENTUM_SPACE) {
    if(fixMomVec.empty()) corr_mom_space = new PLEGMA_FT<Float>(Q2_max);
    else corr_mom_space = new PLEGMA_FT<Float>(this->fixMomVec);
    corr_mom_space->checkAllocation(site_size);
    corr = corr_mom_space->H_elem();
    vol_size = corr_mom_space->Nmoms()*corr_mom_space->DimT();
  }
  else if(corr_space == POSITION_SPACE) {
    corr_pos_space = new PLEGMA_Field<Float>(HOST, site_size, NO_GHOSTS);
    corr = corr_pos_space->H_elem();
    vol_size = corr_pos_space->Total_length();
  }
  else {
    PLEGMA_error("corr_space not supported by correlator");
  }
  isAlloc = true;
}

template<typename Float>
void PLEGMA_Correlator<Float>::
finalize() {
  if (isAlloc) {
    if(corr_space == POSITION_SPACE)
      delete corr_pos_space;
    else if(corr_space == MOMENTUM_SPACE)	 
      delete corr_mom_space;
    else
      PLEGMA_error("corr_space not supported by correlator");
  }
  isAlloc = false;
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractMesons(PLEGMA_Propagator<Float> &prop1,
	       PLEGMA_Propagator<Float> &prop2, 
	       int source[4]){

  setSource(source);
  n_datasets = 2;
  n_groups = N_MESONS;
  shape = {};
  datasets =  {"twop_meson_1", "twop_meson_2"};
  groups =  {"mesons/pseudoscalar", "mesons/scalar", "mesons/g5g1", "mesons/g5g2",
	     "mesons/g5g3", "mesons/g5g4", "mesons/g1", "mesons/g2", "mesons/g3", "mesons/g4"};
  description = "";
  
  initialize();
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  contract_mesons(prop1Tex,prop2Tex,*this);

  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
contractBaryons(PLEGMA_Propagator<Float> &prop1,
		PLEGMA_Propagator<Float> &prop2, 
		int source[4]){

  setSource(source);
  n_datasets = 2;
  n_groups = N_BARYONS;
  shape = {16};
  datasets = {"twop_baryon_1", "twop_baryon_2"};
  groups =  {"nucl_nucl",
#ifdef ALL_BARYONS
	     "nucl_nucl2","nucl2_nucl","nucl2_nucl2","deltap_deltaz_11","deltap_deltaz_22","deltap_deltaz_33",
	     "deltapp_deltamm_11","deltapp_deltamm_22","deltapp_deltamm_33"
#endif
  };
  description = "1,g1,g2,g3,g4,g5,g5g1,g5g2,g5g3,g5g4,s12,s13,s23,s41,s42,s43";

  initialize();
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  contract_baryons(prop1Tex,prop2Tex,*this);

  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}

template<typename FloatC,typename FloatA, typename FloatB>
void contractPropOpProp_local(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			      int signProps, int it, std::vector<GAMMAS> gammas);
template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_local(PLEGMA_Propagator<Float> &bwdProp,
			  PLEGMA_Propagator<Float> &fwdProp,
			  int signProps, std::vector<GAMMAS> gammas,
			  int source[4]){
  n_datasets = 1;
  n_groups = 1;
  shape = {(int) gammas.size()};
  setSource(source);
  datasets = {"threep"};
  groups =  {"Local"};
  description = getGammasString(gammas);
  initialize();

  propTex<Float> bwdPropTex, fwdPropTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");

  for(int it = 0; it < HGC_localL[3]; it++)
    contractPropOpProp_local(*this,bwdPropTex,fwdPropTex,signProps,it,gammas);
  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_oneD(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			     int signProps, su3Tex<FloatS> su3, int it, int dir,std::vector<GAMMAS> gammas);
template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_noe(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			    int signProps, su3Tex<FloatS> su3, int it, int dir,std::vector<GAMMAS> gammas);

template<typename Float>
static void contractNucleonThrp_derGen(PLEGMA_Correlator<Float> &corr, PLEGMA_Propagator<Float> &bwdProp,
				       PLEGMA_Propagator<Float> &fwdProp, PLEGMA_Gauge<Float> &gauge,
				       int signProps, std::vector<GAMMAS> gammas,
				       std::function<void(PLEGMA_Correlator<Float>&,propTex<Float>,
							  propTex<Float>,int,su3Tex<Float>,int,int,
							  std::vector<GAMMAS>)> funcContract){
  // gauge should have the sign for the antiperiodic boundary conditions

  PLEGMA_Su3field<Float> gsu3(DEVICE);
  propTex<Float> bwdPropTex, fwdPropTex;
  su3Tex<Float> gsu3Tex;
  gsu3Tex.tex = gsu3.createTexObject();
  bwdProp.communicateGhost();
  fwdProp.communicateGhost();
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  for(int idir = 0; idir < N_DIMS; idir++){
    gsu3.absorbDir_device(gauge,idir);
    gsu3.communicateGhost(idir+N_DIMS);
    for(int it = 0; it < HGC_localL[3]; it++)
      funcContract(corr,bwdPropTex,fwdPropTex,signProps,gsu3Tex,it, idir,gammas);
  }
  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
  gsu3.destroyTexObject(gsu3Tex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_oneD(PLEGMA_Propagator<Float> &bwdProp,
			 PLEGMA_Propagator<Float> &fwdProp,
			 PLEGMA_Gauge<Float> &gauge,
			 int signProps, std::vector<GAMMAS> gammas,
			 int source[4]){
  n_datasets = 1;
  n_groups = 1;
  shape = {N_DIMS, (int) gammas.size()};
  setSource(source);
  datasets = {"threep"};
  groups =  {"OneD"};
  description = "x,y,z,t / "+getGammasString(gammas);
  initialize();

  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");
  contractNucleonThrp_derGen<Float>(*this,bwdProp,fwdProp,gauge,signProps,gammas,
				    contractPropOpProp_oneD<Float,Float,Float,Float>);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
			PLEGMA_Propagator<Float> &fwdProp,
			PLEGMA_Gauge<Float> &gauge,
			int signProps, int source[4]){
  n_datasets = 1;
  n_groups = 1;
  shape = {N_DIMS};
  setSource(source);
  datasets = {"threep"};
  groups =  {"Noether"};
  description = "x,y,z,t";
  initialize();

  std::vector<GAMMAS> gammas = {};
  contractNucleonThrp_derGen<Float>(*this,bwdProp,fwdProp,gauge,signProps,gammas,
				    contractPropOpProp_noe<Float,Float,Float,Float>);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_wilsonLine(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1,
				   propTex<FloatB> prop2, int signProps,
				   su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas);
template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_wilsonLine(PLEGMA_Propagator<Float> &bwdProp,
			       PLEGMA_Propagator<Float> &fwdProp,
			       PLEGMA_Su3field<Float> &su3,
			       int signProps, std::vector<GAMMAS> gammas,
			       int source[4]){
  n_datasets = 1;
  n_groups = 1;
  shape = {(int) gammas.size()};
  setSource(source);
  datasets = {"threep"};
  groups =  {"wilsonLine"};
  description = getGammasString(gammas);
  initialize();
  
  propTex<Float> bwdPropTex, fwdPropTex;
  su3Tex<Float> sTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  sTex.tex = su3.createTexObject();
  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");

  for(int it = 0; it < HGC_localL[3]; it++)
    contractPropOpProp_wilsonLine(*this,bwdPropTex,fwdPropTex,signProps,sTex,it,gammas);

  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
  su3.destroyTexObject(sTex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeASCII(std::string filename_out) {
  MPI_Comm comm;
  size_t g_vol_size;
  int rank;
  
  if(corr_space == MOMENTUM_SPACE && (HGC_timeRank > HGC_nProc[3] || HGC_timeRank <0 ))
    return;

  switch(corr_space) {
  case MOMENTUM_SPACE:
    g_vol_size = vol_size*HGC_nProc[3];
    comm = HGC_timeComm;
    rank = HGC_timeRank;
    break;
  case POSITION_SPACE:
    g_vol_size = vol_size*HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3];
    comm = MPI_COMM_WORLD;
    rank = comm_rank();
    PLEGMA_error("WriteASCII do not support writing in position space.\n");
    break;
  default:
    PLEGMA_error("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  Float *corrGlobal = NULL;
  if(corr_space == MOMENTUM_SPACE) {
    int Nmoms = corr_mom_space->Nmoms();
    // ===============================================================================
    // reorder data to have time running latest
    Float *corrReorder;
    hostMalloc(corrReorder, vol_size*site_size*2*sizeof(Float));
    memcpy(corrReorder,corr,vol_size*site_size*2*sizeof(Float));

    int site_sizeR=site_size/(n_datasets*n_groups);

    for(int it=0; it<HGC_localL[3]; it++)
      for(int imom=0; imom<Nmoms; imom++)
	for(int id=0; id < n_datasets; id++)
	  for(int ig=0; ig < n_groups; ig++)
	    for(int is=0; is < site_sizeR; is++)
	      for(int ri =0 ; ri < 2 ; ri++)
		corr[it*Nmoms*site_size*2+imom*site_size*2+id*n_groups*site_sizeR*2+ig*site_sizeR*2+is*2+ri]=
		  corrReorder[ig*n_datasets*HGC_localL[3]*Nmoms*site_sizeR*2 + id*HGC_localL[3]*Nmoms*site_sizeR*2 + it*Nmoms*site_sizeR*2 + imom*site_sizeR*2 + is*2+ri];
    hostFree(corrReorder, vol_size*site_size*2*sizeof(Float));
    if(rank == 0) hostMalloc(corrGlobal, g_vol_size*site_size*2*sizeof(Float));
    //=============================================================================
    // TODO: this works fine for timeComm (MOMENTUM_SPACE) but not for MPI_COMM_WORLD (POSITION SPACE)
    // in the second case requires reordering of the memory
    MPI_Gather(corr,site_size*vol_size*2,MPI_Type(corr),
	       corrGlobal,site_size*vol_size*2,MPI_Type(corr),
	       0,comm);
  }

  FILE *ptr_out = NULL;
  if(rank == 0){
    std::string fout,tmpS;
    char *conv;
    asprintf(&conv,"_sx%02dsy%02dsz%02dst%02d.dat", source_position[0], source_position[1], source_position[2],
	     source_position[3]);
    tmpS=conv;
    free(conv);
    fout = filename_out + tmpS;

    ptr_out = fopen(fout.c_str(),"w");
    if(ptr_out == NULL) PLEGMA_error("Error opening file for writing\n");

    if(corr_space == MOMENTUM_SPACE) {
      int Nmoms = corr_mom_space->Nmoms();
      std::vector<std::vector<int>> momV = corr_mom_space->MomList();
      for(int it=0; it<HGC_totalL[3]; it++) {
	int it_shift = (it + source_position[3])%HGC_totalL[3];
	for(int imom=0; imom<Nmoms; imom++) {
	  int ipos = (it_shift*Nmoms + imom)*site_size;
	  fprintf(ptr_out, "%d  %+d  %+d  %+d ", it, momV[imom][0], momV[imom][1], momV[imom][2]);
	  for(int is = 0; is<site_size; is++)
	    fprintf(ptr_out, "%+e %+eI ", corrGlobal[ipos*2+is*2], corrGlobal[ipos*2+is*2+1]);
	  fprintf(ptr_out, "\n");
	}
      }
    }
    else if (corr_space == POSITION_SPACE) {
      //TODO
      PLEGMA_error("WriteASCII do not support writing in position space.\n");
    }
    fclose(ptr_out);
    if(corr_space == MOMENTUM_SPACE)hostFree(corrGlobal, g_vol_size*site_size*2*sizeof(Float));
  }  
}


template<typename Float>
std::string PLEGMA_Correlator<Float>::
fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start) {
  std::string descr = "shape: ";
  switch(corr_space) {
  case MOMENTUM_SPACE:
    descr += "/time/moms";
    // Time
    shape.push_back(HGC_totalL[3]);
    lshape.push_back(HGC_localL[3]);
    start.push_back((HGC_procPosition[3]*HGC_localL[3] + HGC_totalL[3] - source_position[3]) % HGC_totalL[3]);
    // Moms
    shape.push_back((hsize_t)corr_mom_space->Nmoms());
    lshape.push_back((hsize_t)corr_mom_space->Nmoms());
    start.push_back(0);
    break;
  case POSITION_SPACE:
    descr += "/x/y/z/t";
    // Volume
    for(int i=0; i<N_DIMS; i++) {
      shape.push_back(HGC_totalL[i]);
      lshape.push_back(HGC_localL[i]);
      start.push_back((HGC_procPosition[i]*HGC_localL[i] + HGC_totalL[i] - source_position[i]) % HGC_totalL[i]);
    }
    break;
  default:
    PLEGMA_error("Corralator: corr_space not supported: %d\n", corr_space);
  }
  // Correlator shape
  if(!this->shape.empty()) {
    descr += "/" + description;
    for(auto s : this->shape) {
      shape.push_back(s);
      lshape.push_back(s);
      start.push_back(0);    
    }
  }
  //re-im
  descr += "/re-im";
  shape.push_back(2);
  lshape.push_back(2);
  start.push_back(0);

  return descr;
}

static size_t
use_multiple_writers(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start, int &nWriters, int id) {
  if(nWriters==1) return 0;
  int usedWriters = 1;
  size_t shift = 0;
  for(int i=0; i<lshape.size(); i++) {
    int iSize = (lshape[i] + nWriters-1)/nWriters;
    int iWriters = (lshape[i] + iSize-1)/iSize;
    nWriters /= iWriters;
    usedWriters *= iWriters;
    int iId = id % iWriters;
    id /= iWriters;
    int iShift = iSize*iId;
    shift = shift*lshape[i] + iShift;
    if(iShift+iSize > lshape[i])
      lshape[i] -= iShift;
    else
      lshape[i] = iSize;
    start[i] = (start[i] + iShift) % shape[i];
  }
  nWriters = usedWriters;
  return shift;
}

template<typename T>
static std::string str(T begin, T end) {
  std::stringstream ss;
  bool first = true;
  for (; begin != end; begin++) {
    if (!first) ss << ", ";
    ss << *begin;
    first = false;
  }
  return ss.str();
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeHDF5(std::string filename) {

  std::vector<hsize_t> shape, lshape, start;
  std::string descr = fill_H5_shapes(shape, lshape, start);

  hsize_t corrSize = 2*getVolSize();
  hsize_t writeSize = 1;
  for(auto l: this->shape) corrSize*=l;
  for(auto l: lshape) writeSize*=l;
  assert(corrSize==writeSize);

  // In case of MOMENTUM_SPACE, all the processes in HGC_spaceComm has the same information.
  // All of them will write a different piece
  int nWriters = (corr_space == MOMENTUM_SPACE) ? HGC_spaceSize : 1;
  int id = (corr_space == MOMENTUM_SPACE) ? HGC_spaceRank : 0;
  size_t corrShift = use_multiple_writers(shape, lshape, start, nWriters, id);
  if(id >= nWriters) lshape[0] = 0; // not writing
  if(nWriters>1) {
    if(HGC_verbosity > 3) {
      std::string out = "rank: "+std::to_string(id)+
	", shape: ("+str(shape.begin(), shape.end())+
	"), lshape: ("+str(lshape.begin(), lshape.end())+
	"), start: ("+str(start.begin(), start.end())+
	"), shift: "+std::to_string(corrShift)+"\n";
      printf(out.c_str());
    }
  }

  HDF5 writer(filename, MPI_COMM_WORLD);

  char *source;
  asprintf(&source,"/sx%02dsy%02dsz%02dst%02d/", source_position[0], source_position[1], source_position[2],
	   source_position[3]);
  std::string top=(std::string) "/" + source; 
  free(source);

  
  std::vector<hsize_t> momShape = { 3 };
  std::vector<int> mvec;
  if(corr_space == MOMENTUM_SPACE) for(auto mv: corr_mom_space->MomList()) for(auto m: mv) mvec.push_back(m);
  
  for(int g=0; g<n_groups; g++){
    writer.cd(top+groups[g]);
    if(corr_space == MOMENTUM_SPACE) {
      writer.write_dataset("mvec", mvec, momShape);
    }
    for(int d=0; d<n_datasets; d++) {
      Float *writeBuf = corr + (g*n_datasets+d)*writeSize + corrShift;
      writer.write_dataset(datasets[d], writeBuf, shape, lshape, start);
      writer.write_attribute(datasets[d], "description", descr);
    }
  }
}

template class PLEGMA_Correlator<float>;
template class PLEGMA_Correlator<double>;
