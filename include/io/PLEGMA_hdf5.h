#include <hdf5.h>

static hid_t H5_open_file(std::string name, MPI_Comm comm=MPI_COMM_WORLD) {
  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);
  if(HGC_verbosity > 2) PLEGMA_printf("Opened file %s\n", name.c_str());
  
  return file_id;
}

inline bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

static hid_t H5_open_group(hid_t group, std::string name) {
  
  if(name != "") {
    // replacing spaces with _
    while(replace(name, " ", "_")) {}
    
    // splitting / and creating sub groups
    if (name.find("/") != std::string::npos) {
      std::vector<std::string> sub_groups;
      std::stringstream ss(name);
      std::string part;
      while (std::getline(ss, part, '/')) {
	sub_groups.push_back(part);
      }
      for (std::string  n : sub_groups) {
	if(n != "")
	  group = H5Gcreate(group, n.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }
    } else {
      group = H5Gcreate(group, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    if(HGC_verbosity > 2) PLEGMA_printf("Opened group %s\n", name.c_str());
  }
  return group;
}

/* Attribute writing */
template<typename T>
static void H5_write_attribute(hid_t group_id, std::string attr_name, T attr_value);
static void H5_write_attribute(hid_t group_id, std::string attr_name, std::string attr_value) {
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, attr_value.length());
  hid_t attr_id = H5Acreate2(group_id, attr_name.c_str(), type_id, 
			     attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, attr_value.c_str());
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

template<typename Float>
static void H5_write_dataset(hid_t group_id, std::string name, Float *buf, int ndims, hsize_t* dims, hsize_t* ldims, hsize_t* start) {
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
      if(HGC_verbosity > 2) PLEGMA_printf("rank %d: Dim %d exceeds of %d -> shifting\n",comm_rank(),i,exceeding_ammount[exceeding]);
      if(exceeding_dim[exceeding]>ldims[i]) {
	PLEGMA_error("Exceeding is too high. The method may have problems.");
      }
      exceeding++;
    }
    else {
      exceeding_ammount[i] = 0;
    }
  }
  
  hid_t filespace  = H5Screate_simple(ndims, dims,  NULL);
  hid_t dataset_id = H5Dcreate(group_id, name.c_str(), DATATYPE_H5, filespace,
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
    if(HGC_verbosity > 2) { 
      printf("rank %d: iter %d:\n",comm_rank(),i);
      for(int j=0; j<ndims; j++) {
	printf("rank %d: i=%d, dims[i]=%d tmp_ldims[i]=%d, tmp_start[i]=%d, tmp_shift[i]=%d,\n",comm_rank(),j,dims[j], tmp_ldims[j], tmp_start[j], tmp_shift[j]);
      }
    }
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
    if(status<0) PLEGMA_error("write_dataset: Unsuccessful writing of the dataset. Exiting\n");
    H5Sclose(subspace);
    H5Pclose(plist_id);    
    H5Sclose(filespace);
  }

  H5Dclose(dataset_id);
  if(HGC_verbosity > 2) PLEGMA_printf("Written dataset %s\n", name.c_str());
}
