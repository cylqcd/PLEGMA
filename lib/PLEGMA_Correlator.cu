#include <PLEGMA_Gauge.h>
#include <PLEGMA_Correlator.h>
#include <PLEGMA_Propagator.h>
#include <string>
#include <PLEGMA_mesons.cuh>
#include <PLEGMA_baryons.cuh>
#include <PLEGMA_threep.cuh>
#include <functional>
#ifdef PLEGMA_UDSC_BARYONS
#include <PLEGMA_baryons_udsc.cuh>
#endif

using namespace plegma;

//--------------------------------//
// class PLEGMA_Correlator //
//--------------------------------//

template<typename Float>
void PLEGMA_Correlator<Float>::
initialize() {
  comm.reset(new MPI_Comm(HGC_fullComm), [](MPI_Comm* ptr){MPI_Comm_free(ptr); delete ptr;});
  MPI_Comm_dup( HGC_fullComm, comm.get() );
  if(corr_space == MOMENTUM_SPACE) {
    corr_mom_space.reset(new PLEGMA_FT<Float>(*corr_mom_space));
    corr_mom_space->checkAllocation(getSiteSize());
  }
  else if(corr_space == POSITION_SPACE) {
    corr_pos_space.reset(new PLEGMA_Field<Float>(HOST, getSiteSize(), HGC_localVolume3D*localT(),
						 NO_GHOSTS));
  }
  else {
    PLEGMA_error("corr_space not supported by correlator");
  }
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractMesons(PLEGMA_Propagator<Float> &prop1,
	       PLEGMA_Propagator<Float> &prop2 ){

  shape = {10};
  datasets =  {"twop_meson_1", "twop_meson_2"};
  groups =  {"mesons"};
  description = "pseudoscalar, scalar, g5g1, g5g2, g5g3, g5g4, g1, g2, g3, g4";
  
  initialize();
  contract_mesons(prop1,prop2,*this);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
contractBaryons(PLEGMA_Propagator<Float> &prop1,
		PLEGMA_Propagator<Float> &prop2 ){

  shape = {16};
  datasets = {"twop_baryon_1", "twop_baryon_2"};
  groups =  {"baryons/nucl_nucl",
#ifdef PLEGMA_LIGHT_BARYONS
	     "baryons/nucl_nucl2","baryons/nucl2_nucl","baryons/nucl2_nucl2",
	     "baryons/deltap_deltaz_11","baryons/deltap_deltaz_22","baryons/deltap_deltaz_33",
	     "baryons/deltapp_deltamm_11","baryons/deltapp_deltamm_22","baryons/deltapp_deltamm_33"
#endif
  };
  description = "1,g1,g2,g3,g4,g5,g5g1,g5g2,g5g3,g5g4,s12,s13,s23,s41,s42,s43";

  initialize();
  contract_baryons(prop1,prop2,*this);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractBaryonsUDSC(PLEGMA_Propagator<Float> &propUP,
		    PLEGMA_Propagator<Float> &propDN, 
		    PLEGMA_Propagator<Float> &propST, 
		    PLEGMA_Propagator<Float> &propCH, 
		    bool only_st, bool only_ch){

#ifdef PLEGMA_UDSC_BARYONS
  shape = {};
  description = "";
  datasets = {};
  groups = {};

  bool not_up = propUP.getAllocation() == NONE;
  bool not_dn = propDN.getAllocation() == NONE;
  bool not_st = propST.getAllocation() == NONE;
  bool not_ch = propCH.getAllocation() == NONE;

  std::vector<int> todo;
  for(int i=0; i<BP_prop_prods.size(); i++) {
    if(not_up && BP_prop_prods[i].find('u')!=std::string::npos)
      continue;
    if(not_dn && BP_prop_prods[i].find('d')!=std::string::npos)
      continue;
    if(not_st && BP_prop_prods[i].find('s')!=std::string::npos)
      continue;
    if(not_ch && BP_prop_prods[i].find('c')!=std::string::npos)
      continue;
    if(only_st && BP_prop_prods[i].find('s')==std::string::npos)
      continue;
    if(only_ch && BP_prop_prods[i].find('c')==std::string::npos)
      continue;
    if(true) {
      todo.push_back(i);
      for(auto name: BP_prop_prods_names[i])
	datasets.push_back(name);
    }
  }

  if(HGC_verbosity > 2) {
    PLEGMA_printf("contractBaryonsUDSC is going to run: ");
    for(auto name: datasets)
      PLEGMA_printf("%s, ", name.c_str());
    PLEGMA_printf("\n");
  }

  initialize();
  contract_baryons_udsc(propUP, propDN, propST, propCH, *this, todo);
#else
  PLEGMA_error("Flag PLEGMA_UDSC_BARYONS not defined");
#endif
}


template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_local(PLEGMA_Propagator<Float> &bwdProp,
			  PLEGMA_Propagator<Float> &fwdProp,
			  int signProps, std::vector<GAMMAS> gammas ){
  shape = {(int) gammas.size()};
  datasets = {"threep"};
  groups =  {"Local"};
  description = getGammasString(gammas);
  initialize();

  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");
  threep_local(*this,bwdProp,fwdProp,signProps,gammas);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_oneD(PLEGMA_Propagator<Float> &bwdProp,
			 PLEGMA_Propagator<Float> &fwdProp,
			 PLEGMA_Gauge<Float> &gauge,
			 int signProps, std::vector<GAMMAS> gammas){
  shape = {N_DIMS, (int) gammas.size()};
  datasets = {"threep"};
  groups =  {"OneD"};
  description = "x,y,z,t / "+getGammasString(gammas);
  initialize();

  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");
  
  gauge.communicateSideGhost();
  bwdProp.communicateGhost();
  fwdProp.communicateGhost();
  
  threep_oneD(*this,bwdProp,fwdProp,signProps,gauge,gammas);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
			PLEGMA_Propagator<Float> &fwdProp,
			PLEGMA_Gauge<Float> &gauge, int signProps){
  shape = {N_DIMS};
  datasets = {"threep"};
  groups =  {"Noether"};
  description = "x,y,z,t";
  initialize();

  gauge.communicateSideGhost();
  bwdProp.communicateGhost();
  fwdProp.communicateGhost();
  
  threep_noe(*this,bwdProp,fwdProp,signProps,gauge);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_wilsonLine(PLEGMA_Propagator<Float> &bwdProp,
			       PLEGMA_Propagator<Float> &fwdProp,
			       PLEGMA_Su3field<Float> &su3,
			       int signProps, std::vector<GAMMAS> gammas){
  shape = {(int) gammas.size()};
  datasets = {"threep"};
  groups =  {"wilsonLine"};
  description = getGammasString(gammas);
  initialize();
  
  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");
  threep_wilsonLine(*this,bwdProp,fwdProp,signProps,su3,gammas);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeASCII(std::string filename_out) const {
  MPI_Comm comm;
  size_t g_vol_size = getVolSize();
  int rank;
  
  if(corr_space == MOMENTUM_SPACE && (HGC_timeRank > HGC_nProc[3] || HGC_timeRank <0 ))
    return;

  switch(corr_space) {
  case MOMENTUM_SPACE:
    g_vol_size *= HGC_nProc[3];
    comm = HGC_timeComm;
    rank = HGC_timeRank;
    break;
  case POSITION_SPACE:
    g_vol_size *= HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3];
    comm = HGC_fullComm;
    rank = comm_rank();
    PLEGMA_error("WriteASCII do not support writing in position space.\n");
    break;
  default:
    PLEGMA_error("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  Float corrGlobal[rank == 0 ? (g_vol_size*getSiteSize()*2) : 0];
  
  if(corr_space == MOMENTUM_SPACE) {
    int Nmoms = corr_mom_space->Nmoms();
    // ===============================================================================
    // reorder data to have time running latest
    Float corrReorder[getTotalSize()*2];

    int site_sizeR=getSiteSize()/(nDatasets()*nGroups());

    for(int it=0; it<HGC_localL[3]; it++)
      for(int imom=0; imom<Nmoms; imom++)
	for(int id=0; id < nDatasets(); id++)
	  for(int ig=0; ig < nGroups(); ig++)
	    for(int is=0; is < site_sizeR; is++)
	      for(int ri =0 ; ri < 2 ; ri++)
		corrReorder[((((it*Nmoms+imom)*nDatasets()+id)*nGroups()+ig)*site_sizeR+is)*2+ri]=
		  H_elem()[((((ig*nDatasets()+id)*HGC_localL[3]+it)*Nmoms+imom)*site_sizeR+is)*2+ri];

    //=============================================================================
    // TODO: this works fine for timeComm (MOMENTUM_SPACE) but not for HGC_fullComm (POSITION SPACE)
    // in the second case requires reordering of the memory
    MPI_Gather(corrReorder,sizeof(corrReorder)/sizeof(Float),MPI_Type(corrReorder),
	       corrGlobal,sizeof(corrReorder)/sizeof(Float),MPI_Type(corrReorder),
	       0,comm);
  }

  FILE *ptr_out = NULL;
  if(rank == 0){
    std::string fout,tmpS;
    char *conv;
    asprintf(&conv,"_sx%02dsy%02dsz%02dst%02d.dat", source[0], source[1], source[2], source[3]);
    tmpS=conv;
    free(conv);
    fout = filename_out + tmpS;

    ptr_out = fopen(fout.c_str(),"w");
    if(ptr_out == NULL) PLEGMA_error("Error opening file for writing\n");

    if(corr_space == MOMENTUM_SPACE) {
      int Nmoms = corr_mom_space->Nmoms();
      std::vector<std::vector<Float>> momV = corr_mom_space->MomList();
      for(int it=0; it<HGC_totalL[3]; it++) {
	int it_shift = (it + source[3])%HGC_totalL[3];
	for(int imom=0; imom<Nmoms; imom++) {
	  int ipos = (it_shift*Nmoms + imom)*getSiteSize();
	  fprintf(ptr_out, "%d  %+d  %+d  %+d ", it, (int) round(momV[imom][0]),(int) round(momV[imom][1]),(int) round(momV[imom][2]));
	  for(int is = 0; is<getSiteSize(); is++)
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
  }  
}


template<typename Float>
std::string PLEGMA_Correlator<Float>::
fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start) const{
  std::string descr = "shape: ";
  switch(corr_space) {
  case MOMENTUM_SPACE:
    descr += "/time/moms";
    // Time
    shape.push_back(totalT);
    lshape.push_back(localT());
    start.push_back(startT());
    // Moms
    shape.push_back((hsize_t)corr_mom_space->Nmoms());
    lshape.push_back((hsize_t)corr_mom_space->Nmoms());
    start.push_back(0);
    break;
  case POSITION_SPACE:
    descr += "/t/z/y/x";
    // Volume
    for(int i=N_DIMS-1; i>=0; i--) {
      shape.push_back(i==DIM_T ? totalT : HGC_totalL[i]);
      lshape.push_back(i==DIM_T ? localT() : HGC_localL[i]);
      start.push_back(i==DIM_T ? startT() : ((HGC_procPosition[i]*HGC_localL[i] + HGC_totalL[i] - source[i]) % HGC_totalL[i]));
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
writeHDF5(std::string filename) const {
  std::vector<hsize_t> shape, lshape, start;
  std::string descr = fill_H5_shapes(shape, lshape, start);

  hsize_t corrSize = 2*getVolSize();
  for(auto l: this->shape) corrSize*=l;

  hsize_t writeSize = 1;
  for(auto l: lshape) writeSize*=l;
  assert(corrSize==writeSize);

  // In case of MOMENTUM_SPACE, all the processes in HGC_spaceComm has the same information.
  // All of them will write a different piece
  int nWriters = writeSize==0 ? 0 : ((corr_space == MOMENTUM_SPACE) ? HGC_spaceSize : 1);
  int id = (corr_space == MOMENTUM_SPACE) ? HGC_spaceRank : 0;
  size_t corrShift = writeSize==0 ? 0 : use_multiple_writers(shape, lshape, start, nWriters, id);
  
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

  HDF5 writer(filename, *comm);
    
  char *ssource;
  asprintf(&ssource,"/sx%02dsy%02dsz%02dst%02d/", source[0], source[1], source[2], source[3]);
  std::string top=(std::string) "/" + ssource; 
  free(ssource);
  
  std::vector<hsize_t> momShape;
  std::vector<int> mvec;

  if(corr_space == MOMENTUM_SPACE){
    momShape = { corr_mom_space->MomList()[0].size() };
    for(auto mv: corr_mom_space->MomList()) for(auto m: mv) mvec.push_back(m);
  }
  
  for(size_t g=0; g<nGroups(); g++){
    writer.cd(top + (groups.size()>0 ? groups[g] : "/"));
    //PLEGMA_printf("DEBUG: - Change directory for group: %d\n",g);
    //MPI_Barrier(*comm);
    if(corr_space == MOMENTUM_SPACE) {
      writer.write_dataset("mvec", mvec, momShape);
    }
    for(size_t d=0; d<nDatasets(); d++) {
      Float *writeBuf = H_elem() + (g*nDatasets()+d)*writeSize + corrShift;
      std::string dataset = datasets.size() > 0 ? datasets[d] : "arr";
      writer.write_dataset(dataset, writeBuf, shape, lshape, start);
      //PLEGMA_printf("DEBUG: -- dataset %d written\n",d);
      //MPI_Barrier(*comm);
      writer.write_attribute(dataset, "description", descr);
      //PLEGMA_printf("DEBUG: -- attribute to dataset %d written\n",g);
      //MPI_Barrier(*comm);
    }
  }
}


template class PLEGMA_Correlator<float>;
template class PLEGMA_Correlator<double>;
