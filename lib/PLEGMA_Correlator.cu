#include <PLEGMA_Correlator.h>
#include <hdf5.h>
#include <string>
#include <PLEGMA_mesons.cuh>
#include <PLEGMA_baryons.cuh>
#include <PLEGMA_contractPropOpProp.cuh> 
using namespace plegma;

//--------------------------------//
// class PLEGMA_Correlator //
//--------------------------------//

template<typename Float>
void PLEGMA_Correlator<Float>::
initialize(CORR_TYPE CorrType, CORR_SPACE CorrSpace) {
  if(isAlloc && corr_type==CorrType && corr_space==CorrSpace)
    return;

  finalize();
  corr_type=CorrType;
  corr_space=CorrSpace;

  switch(CorrSpace) {
  case MOMENTUM_SPACE:
    vol_size = GK_localL[3]*GK_Nmoms;
    break;
  case POSITION_SPACE:
    vol_size = GK_localVolume;
    break;
  default:
    errorQuda("Corralator: CorrSpace not supported: %d\n", CorrSpace);
  }

  n_groups = nGroups[CorrType];
  n_flavors = nFlavors[CorrType];
  n_comp = nComp[CorrType];
  site_size = n_groups*n_flavors*n_comp;

  bytes_total_length = vol_size*site_size*2*sizeof(Float);
  corr = (Float*) malloc(bytes_total_length);
  if(corr == NULL)
     errorQuda("Correlator: Cannot allocate memory of size %d.", bytes_total_length);
  else
    isAlloc = true;
  memset(corr,0,bytes_total_length);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractMesons(PLEGMA_Propagator<Float> &prop1,
	       PLEGMA_Propagator<Float> &prop2, 
	       int isource, CORR_SPACE corrSpace){

  initialize(MESONS,corrSpace);
  this->isource = isource;
  
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  printfQuda("contractMesons: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");

  for(int it = 0 ; it < GK_localL[3] ; it++) {
    contract_mesons(prop1Tex,prop2Tex,*this,it);
  }

  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractBaryons(PLEGMA_Propagator<Float> &prop1,
	       PLEGMA_Propagator<Float> &prop2, 
	       int isource, CORR_SPACE corrSpace){

  initialize(BARYONS,corrSpace);
  this->isource = isource;
  
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  printfQuda("contractMesons: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");

  for(int it = 0; it < GK_localL[3]; it++) {
    contract_baryons(prop1Tex,prop2Tex,*this,it);
  }

  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}

// template<typename Float>
// void PLEGMA_Correlator<Float>::contractNucleonThrp(PLEGMA_Propagator<Float> &bwdProp, PLEGMA_Propagator<Float> &fwdProp,
// 		    int signProps, PLEGMA_Su3field<Float> &su3, std::vector<GAMMAS> gammas, int isource, CORR_SPACE corrSpace){
//   initialize(THRP_LOCAL,corrSpace);
  
//   propTex<Float> bwdPropTex, fwdPropTex;
//   su3Tex<Float> sTex;
//   bwdPropTex.tex = bwdProp.createTexObject();
//   fwdPropTex.tex = fwdProp.createTexObject();
//   sTex.tex = su3.createTexObject();
//   this->isource = isource;
//   printfQuda("contractNucleonThrp: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");

//   for(int it = 0; it < GK_localL[3]; it++) contractPropOpProp(*this,bwdPropTex,fwdPropTex,signProps,sTex,it,gammas);

//   bwdProp.destroyTexObject(bwdPropTex.tex);
//   fwdProp.destroyTexObject(fwdPropTex.tex);
//   su3.destroyTexObject(sTex.tex);
// }

template<typename Float>
void PLEGMA_Correlator<Float>::contractNucleonThrp(PLEGMA_Propagator<Float> &bwdProp, PLEGMA_Propagator<Float> &fwdProp,
		    int signProps, std::vector<GAMMAS> gammas, int isource, CORR_SPACE corrSpace){
  initialize(THRP_LOCAL,corrSpace);
  propTex<Float> bwdPropTex, fwdPropTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  this->isource = isource;
  printfQuda("contractNucleonThrp: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");
  for(int it = 0; it < GK_localL[3]; it++) contractPropOpProp(*this,bwdPropTex,fwdPropTex,signProps,it,gammas);
  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
writeFile(char *filename, PLEGMA_params &params) {
  if(params.CorrFileFormat == ASCII_FORM) {
    printfQuda("Going to write file %s in ASCII format\n",filename);
    writeASCII(filename);
  }
  else if(params.CorrFileFormat == HDF5_FORM) {
    printfQuda("Going to write file %s in HDF5 format\n",filename);
    writeHDF5(filename, params);
  }
  else {
    errorQuda("FILE_WRITE_FORMAT not supported: %d\n", params.CorrFileFormat);
  }
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeFile(PLEGMA_params &params) {
  char *filename, *Qsq, *name2;
  std::string ext="", name="";
  if(params.CorrSpace==MOMENTUM_SPACE) asprintf(&Qsq,"Qsq%d_",params.Q_sq);
  else asprintf(&Qsq,"");
  if(params.CorrFileFormat == ASCII_FORM) ext = ".dat";
  else if(params.CorrFileFormat == HDF5_FORM) ext = ".h5";
  switch(corr_type) {
  case MESONS:
    name = "twop.%04d_mesons";
    break;
  case BARYONS:
    name = "twop.%04d_baryons";
    break;
  case THRP_LOCAL:
    name = "thrp.%04d_local";
    break;
  case THRP_NOETHER:
    name = "thrp.%04d_noether";
    break;
  case THRP_ONED:
    name = "thrp.%04d_oneD";
    break;
  default:
    name = "unknown.%04d";
  }
  asprintf(&name2, name.c_str(), params.traj);
  asprintf(&filename,"%s/%s_%sSS.%02d.%02d.%02d.%02d%s" ,
	   params.corr_dir, name2, Qsq,
	   params.sourcePosition[isource][0],
	   params.sourcePosition[isource][1],
	   params.sourcePosition[isource][2],
	   params.sourcePosition[isource][3], ext.c_str());

  writeFile(filename, params);
  free(name2);
  free(Qsq);
  free(filename);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
writeASCII(char *filename_out) {
  MPI_Comm comm;
  size_t g_vol_size;
  int rank;
  
  if(corr_space == MOMENTUM_SPACE && (GK_timeRank > GK_nProc[3] || GK_timeRank <0 ))
    return;

  switch(corr_space) {
  case MOMENTUM_SPACE:
    g_vol_size = vol_size*GK_nProc[3];
    comm = GK_timeComm;
    rank = GK_timeRank;
    break;
  case POSITION_SPACE:
    g_vol_size = vol_size*GK_nProc[0]*GK_nProc[1]*GK_nProc[2]*GK_nProc[3];
    comm = MPI_COMM_WORLD;
    rank = comm_rank();
    break;
  default:
    errorQuda("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  Float *corrGlobal;
  if(rank == 0){
    corrGlobal = (Float*) malloc(g_vol_size*site_size*2*sizeof(Float));
    if( corrGlobal == NULL ) errorQuda("writeASCII: Cannot allocate memory.");
  }

  MPI_Gather(corr,site_size*vol_size*2,MPI_Type(corr),
	     corrGlobal,site_size*vol_size*2,MPI_Type(corr),
	     0,comm);

  FILE *ptr_out = NULL;
  if(rank == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    if(n_flavors != 1) errorQuda("For now works with n_flavors = 1\n");
    for(size_t v=0; v<g_vol_size*site_size; v++) {
      int it = v/(GK_Nmoms*site_size);
      int imom = v/(site_size) - it*GK_Nmoms;
      int is = v%site_size;
      int it_shift = (it + GK_sourcePosition[isource][3])%GK_totalL[3];
      int ipos = it_shift*GK_Nmoms*site_size + imom*site_size + is;
      if(corr_space == MOMENTUM_SPACE) {
	if(is == 0)fprintf(ptr_out, "%d  %+d  %+d  %+d ", v/(GK_Nmoms*site_size), GK_moms[imom][0], GK_moms[imom][1], GK_moms[imom][2]);
      }
      else if (corr_space == POSITION_SPACE) {
	//TODO
      }
      fprintf(ptr_out, "%+e %+eI ", corrGlobal[ipos*2], corrGlobal[ipos*2+1]);
      if(is == site_size-1)fprintf(ptr_out, "\n");
    }
    fclose(ptr_out);
    free(corrGlobal);
  }  
}

static int getNDims(CORR_TYPE CorrType, CORR_SPACE CorrSpace) {
  int ndims = 1; //re-im
  switch(CorrSpace) {
  case MOMENTUM_SPACE:
    ndims += 2; // mom, t
    break;
  case POSITION_SPACE:
    ndims += 4; // t, z, y, x
    break;
  default:
    errorQuda("Corralator: CorrSpace not supported: %d\n", CorrSpace);
  }
  ndims+=nDims[CorrType];
 
  return ndims;
}

static void fillDims(CORR_TYPE CorrType, CORR_SPACE CorrSpace, int ndims,
		     hsize_t* dims, hsize_t* ldims, hsize_t* start, int *sourcePosition, bool shift_source) {
  start[ndims-1] = 0; ldims[ndims-1] = dims[ndims-1] = 2; //re-im
  switch(CorrType) {
  case MESONS:
    break;
  case BARYONS:
    start[ndims-2] = 0; ldims[ndims-2] = dims[ndims-2] = 16; //n-gamma
    break;
  case THRP_LOCAL:
    start[ndims-2] = 0; ldims[ndims-2] = dims[ndims-2] = 16; //n-gamma
    break;
  case THRP_NOETHER:
    start[ndims-2] = 0; ldims[ndims-2] = dims[ndims-2] = 4; //n-gamma
    break;
  case THRP_ONED:
    start[ndims-2] = 0; ldims[ndims-2] = dims[ndims-2] = 16; //n-gamma
    start[ndims-3] = 0; ldims[ndims-3] = dims[ndims-3] = 4; //n-dirs
    break;
  default:
    errorQuda("Corralator: CorrType not supported: %d\n", CorrType);
  }
  switch(CorrSpace) {
  case MOMENTUM_SPACE:
    start[1] = 0; ldims[1] = dims[1] = GK_Nmoms; //Nmoms
    start[0] = GK_timeRank*GK_localL[3]; //starting point
    ldims[0] = GK_localL[3]; //LT
    dims[0] = GK_totalL[3]; //T
    if(shift_source) {
      start[0] = (start[0] + GK_totalL[3] - sourcePosition[3]) % GK_totalL[3];
    }
    break;
  case POSITION_SPACE:
    for(int i=0; i<N_DIMS; i++) {
      start[i] = comm_coords(default_topo)[i]*GK_localL[i]; //starting
      ldims[i] = GK_localL[i]; //LT
      dims[i] = GK_totalL[i]; //T
      if(shift_source) {
	start[i] = (start[i] + GK_totalL[i] - sourcePosition[i]) % GK_totalL[i];
      }
    }
    break;
  default:
    errorQuda("Corralator: CorrSpace not supported: %d\n", CorrSpace);
  }
}

/* Attribute writing */
static void write_text_attribute(hid_t group_id,const char* attr_name, char* attr_value) {
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(attr_value));
  hid_t attr_id = H5Acreate2(group_id, attr_name, type_id, 
			     attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, attr_value);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);
}

static inline size_t get_volume(int ndims, hsize_t* dims){
  size_t volume = 1;
  for(int i=0; i<ndims; i++){
    volume*=dims[i];
  }
  return volume;
}

static inline size_t get_id(int ndims,hsize_t* ids,hsize_t* dims){
  size_t id = ids[0];
  for(int i=1; i<ndims; i++){
    id = id*dims[i] + ids[i];
  }
  return id;
}

static inline void get_ids(int ndims, size_t id, hsize_t* ids, hsize_t* dims){
  for(int i=ndims-1; i>=0; i--){
    ids[i] = id % dims[i];
    id/=dims[i];
  }
}

static inline void add_ids(int ndims, hsize_t* ids, hsize_t* ids2){
  for(int i=ndims-1; i>=0; i--){
    ids[i] += ids2[i];
  }
}


// This writing function supports shifted chunk, i.e. when start[i]+ldims[i] > dims[i].
// The part exceeding is written starting from 0 as in case of periodic boundaries. 
template<typename Float>
static void write_dataset(hid_t group_id, const char* name, Float *buf, int ndims, hsize_t* dims, hsize_t* ldims, hsize_t* start) {
  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
  }

  // checking for exceeding dims
  int exceeding = 0;
  int exceeding_dim[ndims];
  hsize_t exceeding_ammount[ndims];
  for(int i=0; i<ndims; i++) {
    if(start[i]+ldims[i] > dims[i]) {
      exceeding_dim[exceeding] = i;
      exceeding_ammount[i] = start[i]+ldims[i]-dims[i];
      //printf("rank %d: Dim %d exceeds of %d -> shifting\n",comm_rank(),i,exceeding_ammount[exceeding]);
      if(exceeding_dim[exceeding]>ldims[i]) {
	errorQuda("Exceeding is too high. The method may have problems.");
      }
      exceeding++;
    }
    else {
      exceeding_ammount[i] = 0;
    }
  }
  
  hid_t filespace  = H5Screate_simple(ndims, dims,  NULL);
  hid_t dataset_id = H5Dcreate(group_id, name, DATATYPE_H5, filespace,
			       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // We need 2^exceeding loops for writing everything 
  for(int i=0; i<(1<<exceeding); i++) {
    filespace = H5Dget_space(dataset_id);
    hsize_t tmp_ldims[ndims];
    hsize_t tmp_start[ndims];
    hsize_t tmp_shift[ndims];
    for(int j=0; j<ndims; j++) {
      tmp_ldims[j] = ldims[j] - exceeding_ammount[j];
      tmp_start[j] = start[j];
      tmp_shift[j] = 0;
    }
    // checking if we shift the exceeding direction
    int tmp=i, j=0;
    while(tmp>0) {
      if(tmp%2) {
	int dim = exceeding_dim[j];
	tmp_shift[dim] = tmp_ldims[dim];
	tmp_ldims[dim] = exceeding_ammount[dim];
	tmp_start[dim] = 0;
      }
      tmp/=2;
      j++;
    }
    //printf("rank %d: iter %d:\n",comm_rank(),i);
    //for(int j=0; j<ndims; j++) {
    //  printf("rank %d: i=%d, dims[i]=%d tmp_ldims[i]=%d, tmp_start[i]=%d, tmp_shift[i]=%d,\n",comm_rank(),j,dims[j], tmp_ldims[j], tmp_start[j], tmp_shift[j]);
    //}
    // copying the part of the buffer
    size_t vol = get_volume(ndims, tmp_ldims);
    Float tmp_buf[get_volume(ndims, tmp_ldims)];
    for(size_t j=0; j<vol; j++) {
      hsize_t ids[ndims];
      get_ids(ndims, j, ids, tmp_ldims);
      add_ids(ndims, ids, tmp_shift);
      tmp_buf[j] = buf[get_id(ndims,ids,ldims)];
    }
    
    hid_t subspace   = H5Screate_simple(ndims, tmp_ldims, NULL);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, tmp_start, NULL, tmp_ldims, NULL);
    
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // TODO: we need to fix the parallel writing in case of more than one processor has an exceeding direction.
    herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, 
			     i==0 ? plist_id : H5P_DEFAULT, tmp_buf);
    if(status<0) errorQuda("write_dataset: Unsuccessful writing of the dataset. Exiting\n");
    H5Sclose(subspace);
    H5Pclose(plist_id);    
    H5Sclose(filespace);
  }

  H5Dclose(dataset_id);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
writeHDF5(char *filename, PLEGMA_params &params) {
  // only one per time writes in momentum space
  if(corr_space == MOMENTUM_SPACE && (GK_timeRank > GK_nProc[3] || GK_timeRank <0 ))
    return;

  MPI_Comm comm;
  bool shift_source = false;
  switch(corr_space) {
  case MOMENTUM_SPACE:
    comm = GK_timeComm;
    shift_source = true;
    break;
  case POSITION_SPACE:
    comm = MPI_COMM_WORLD;
    // shift source not yet supported in position space. the parallel writing needs to be fixed.
    // TODO: command_line_flag
    shift_source = false;
    break;
  default:
    errorQuda("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  int ndims = getNDims(corr_type, corr_space);
  hsize_t dims[ndims], ldims[ndims], start[ndims];
  fillDims(corr_type, corr_space, ndims, dims, ldims, start, GK_sourcePosition[isource], shift_source);

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",params.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  write_text_attribute(group2_id, "source-position", src_pos);
  free(src_pos);

  //- Index identification-ordering, precision
  //char *corr_info;
  //asprintf(&corr_info,"Position-space meson 2pt-correlator\nIndex Order: [flav, t, z, y, x, real/imag]\nPrecision: %s\0",(typeid(Float) == typeid(float)) ? "single" : "double");
  //write_text_attribute(file_id, "Correlator-info", src_pos);
  //free(corr_info);

  size_t spaceSize = get_volume(ndims, ldims);

  for(int g=0; g<n_groups; g++){
    hid_t group3_id = H5Gcreate(group2_id, corr_groups_names[corr_type][g], H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);
    for(int d=0; d<n_flavors; d++){
      Float *writeBuf = corr + (g*n_flavors+d)*spaceSize;
      write_dataset(group3_id, corr_flavors_names[corr_type][d], writeBuf,
		    ndims, dims, ldims, start);
    }
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);
}

template class PLEGMA_Correlator<float>;
template class PLEGMA_Correlator<double>;
