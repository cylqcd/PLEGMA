#include <hdf5.h>

//TODO: This function should be overloaded for different data type
template<typename T> inline hid_t datatype();
template<typename T> inline hid_t datatype(T a) { return datatype<T>(); }
template<> inline hid_t datatype<int>() {return H5T_NATIVE_INT; }
template<> inline hid_t datatype<int*>() {return H5T_NATIVE_INT; }
template<> inline hid_t datatype<float>() {return H5T_NATIVE_FLOAT; }
template<> inline hid_t datatype<float*>() {return H5T_NATIVE_FLOAT; }
template<> inline hid_t datatype<double>() {return H5T_NATIVE_DOUBLE; }
template<> inline hid_t datatype<double*>() {return H5T_NATIVE_DOUBLE; }

/**
 *    @brief Class for HDF5 data writing (TODO: and reading)
 **/
class HDF5 {
protected:
  MPI_Comm comm;
  hid_t file_id;
  std::string filename;
  std::vector<hid_t> path_id;
  std::vector<std::string> path_str;
  std::string prev_path;

  static std::vector<std::string> open_files;
  
  inline std::string join_path(std::vector<std::string> vp, bool fromTop = true) {
    std::string ret = fromTop ? "" : "." ;
    for (auto s : vp) ret += "/" + s;
    return ret;
  }

public:
  // here the descriprion of classes that will be implemented after.
  /*
   * @brief Creates or opens an existing HDF5 file.
   * @param name the filename. The extension '.h5' will be added if not provided. It can also contain a list of groups to open, e.g. name="./sample.h5/group1/group2" would create the file sample.h5 an dthen go to group1 and group2.
   * @param comm the communicator to use during the file writing.
   **/
  //  HDF5(std::string name, MPI_Comm comm=MPI_COMM_WORLD);

  /*
   * @brief Does sanity checks and close the file.
   */
  //  ~HDF5();
  
  /*
   * @brief Returns the path to the current group
   * @return a string contining the path to the current
   */

  //static std::vector<std::string> open_files;
  
  inline std::string pwd() {
    return join_path(path_str);
  }
  
  /*
   * @brief Creates or opens the groups to reach the path.
   * @param path a string containing the path. Similar rules to filesystem are used: 
   *  - If the path starts with "/" then is considered as an absolute path starting from the file
   *    otherwise it is considered as a relative path from the last location
   *  - ../ ./ are implemented. (TODO: ~/ to go home)
   */
  //  void cd(std::string path);

  /*
   * @brief Writes an attribute to an object
   * @param object the name of the object. It can aslo contain a path in front, e.g. path/object.
   * @param attr_name the name of the attribute
   * @param attr_value the value of the attribute
   * @param path the path to the object. Default the last location (pwd).
   */
  //  template<typename T>
  //  void write_attribute(std::string object, std::string attr_name, T attr_value, std::string path=".");

  /*
   * @brief Writes a dataset
   * @param name the dataset name. It can aslo contain a path in front, e.g. path/name.
   * @param buf the pointer to the buffer to write.
   * @param shape the global shape of the dataset.
   * @param lshape the local shape of the dataset. If empty then only one process will write the dataset.
   * @param start the starting point of the writing. Can be empty if lshape is empty.
   * @param path the path to the object. Default the last location (pwd). 
   */
  //  template<typename T>
  //  void write_dataset(std::string name, T *buf, std::vector<hsize_t> shape,  std::vector<hsize_t> lshape={},
  //		     std::vector<hsize_t> start={}, std::string path=".");

  /*
   * @brief Writes a dataset
   * @param name the dataset name. It can aslo contain a path in front, e.g. path/name.
   * @param buf a vector containing the data to write
   * @param shape the desired shape of the vector. Can also be partial and the rest is deduced from the buf size
   * @param path the path to the object. Default the last location (pwd). 
   */
  //  template<typename T>
  //  void write_dataset(std::string name, std::vector<T> buf, std::vector<hsize_t> shape, std::string path=".");
  
protected:
  inline int getRank(){
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }

  inline void wait(){
    if(HGC_verbosity > 2) PLEGMA_printf("Waiting...\n");
    MPI_Barrier(comm);
  }

  // Some tools for hsize_t
  inline hsize_t product(std::vector<hsize_t> dims){
    hsize_t product = 1;
    for(auto i : dims) product *= i;
    return product;
  }
  inline hsize_t to_id(std::vector<hsize_t> ids, std::vector<hsize_t> shape){
    hsize_t id = ids[0];
    for(size_t i = 1; i < ids.size(); i++) id = id*shape[i] + ids[i];
    return id;
  }
  inline std::vector<hsize_t> from_id(hsize_t id, std::vector<hsize_t> shape){
    std::vector<hsize_t> ids;
    for (auto s = shape.rbegin(); s != shape.rend(); ++s ) { 
      ids.push_back(id % *s);
      id /= *s;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
  }
  inline std::vector<hsize_t> add(std::vector<hsize_t> shape1, std::vector<hsize_t> shape2){
    std::vector<hsize_t> res;
    for(size_t i=0; i < shape1.size(); i++) res.push_back(shape1[i] + shape2[i]);
    return res;
  }
  inline std::vector<hsize_t> zeros_like(std::vector<hsize_t> shape){
    std::vector<hsize_t> res;
    for(size_t i=0; i < shape.size(); i++) res.push_back(0);
    return res;
  }
  inline std::vector<hsize_t> ones_like(std::vector<hsize_t> shape){
    std::vector<hsize_t> res;
    for(size_t i=0; i < shape.size(); i++) res.push_back(1);
    return res;
  }

  // replace the first finding in a string
  inline bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
      return false;
    str.replace(start_pos, from.length(), to);
    return true;
  }

  // splitting path using "/"
  inline std::vector<std::string> split_path(std::string path) {
    std::vector<std::string> spath;
    if(path == "" || path == "/") return spath;
    if(path.find("/") != std::string::npos) {
      std::stringstream ss(path);
      std::string part;
      while (std::getline(ss, part, '/')) {
	spath.push_back(part);
      }
    } else {
      spath.push_back(path);
    }
    return spath;
  }

  // Removin empty, '.' and in middle '..'
  inline std::vector<std::string> clean_path(std::vector<std::string> vp) {
    for (auto it = vp.begin(); it != vp.end(); ) {
      if (*it == "" || *it == ".") {
	it = vp.erase(it);
      } else if(*it == ".." && it != vp.begin() && *(it-1) != "..") {
	it = vp.erase(it-1); // removing previous
	it = vp.erase(it); // removing ..
      } else {
	++it;
      }
    }
    return vp;
  }

  // pop and close last group from path
  inline void go_back() {
    if(path_id.empty()) PLEGMA_error("Tried to go back but path_id is empty\n");
    H5Gclose(path_id.back());
    path_id.pop_back();
    if(HGC_verbosity > 2) PLEGMA_printf("Closed group %s\n", path_str.back().c_str());
    path_str.pop_back();
  }

  // pop and close all groups
  inline void go_top() {
    while(!path_id.empty()) go_back();
  }

  // Splitting, cleaning and checking until what point path is the same with the current path
  inline std::vector<std::string> prepare_path(std::string path) {
    if(HGC_verbosity > 2) PLEGMA_printf("Path before cleaning %s\n", path.c_str());
    std::vector<std::string> vp = clean_path(split_path(path));
    if(HGC_verbosity > 2) PLEGMA_printf("Path after cleaning %s\n", join_path(vp, path[0]=='/').c_str());
    // checking if starts with '/'
    if(!path_id.empty() && path[0]=='/') {
      if(vp.empty() || vp[0] != path_str[0]) go_top();
      else {
	auto it = vp.begin();
	auto pit = path_str.begin();
	// checking until the paths match
	while(*it == *pit && it != vp.end() && pit != path_str.end()) {
	  it = vp.erase(it);
	  pit++;
	}
	// closing the remaining
	while(pit != path_str.end()) go_back();
      }
    }
    return vp;
  }

  // returns current group
  inline hid_t current() {
    if(path_id.empty()) return file_id;
    else return path_id.back();
  }

  inline bool exists(std::string s) {
    return H5Lexists(current(), s.c_str(), H5P_DEFAULT);
  }

  // Creates or open a group. Replaces also spaces with underscore.
  inline void open(std::string dir) {
    if(dir == "" || dir == ".") {
      PLEGMA_warning("HDF5 open called with '%s'. This should have been clean\n", dir.c_str());
      return;
    } else if(dir == "..") {
      PLEGMA_warning("HDF5 open called with '..'. Calling go_back instead\n");
      go_back();
      return;      
    }
    // replacing " " with "_"
    while(replace(dir, " ", "_")) {}
    // Opening or creating dir
    if(exists(dir)){
      if(H5Oexists_by_name(current(), dir.c_str(), H5P_DEFAULT)) {
	path_id.push_back(H5Gopen(current(), dir.c_str(), H5P_DEFAULT));
	if(HGC_verbosity > 2) PLEGMA_printf("Opened group %s\n", dir.c_str());
      }
      else {
	PLEGMA_error("A link with dir %s exists but it is not a group\n File: %s\n Path: %s", dir.c_str(),
		     filename.c_str(), pwd().c_str());
      }
    }
    else {
      path_id.push_back(H5Gcreate(current(), dir.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
      if(HGC_verbosity > 2) PLEGMA_printf("Created group %s\n", dir.c_str());
    }
    path_str.push_back(dir);
  }

  //TODO: This function should be overloaded for different type of attr_value
  inline void _write_attribute(std::string object, std::string attr_name, std::string attr_value) {
    hid_t obj_id = H5Oopen(current(), object.c_str(), H5P_DEFAULT);
    hid_t attrdat_id = H5Screate(H5S_SCALAR);
    hid_t type_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id, attr_value.length());
    hid_t attr_id = H5Acreate2(obj_id, attr_name.c_str(), type_id, 
			       attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, type_id, attr_value.c_str());
    H5Aclose(attr_id);
    H5Tclose(type_id);
    H5Sclose(attrdat_id);
    H5Oclose(obj_id);
  }


  template<typename T>
  inline hid_t require_dataset(std::string name, std::vector<hsize_t> shape) {
    hid_t filespace  = H5Screate_simple(shape.size(), shape.data(),  NULL);
    // TODO: check if datasets exists and handle such case
    hid_t dataset_id = H5Dcreate(current(), name.c_str(), datatype<T>(), filespace,
				 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);
    return dataset_id;
  }

  template<typename T>
  inline void _write_dataset_parallel(hid_t dataset_id, T *buf, std::vector<hsize_t> shape, std::vector<hsize_t> lshape, std::vector<hsize_t> start, bool serial=false) {
    hid_t filespace = H5Dget_space(dataset_id);
    hid_t subspace   = H5Screate_simple(lshape.size(), lshape.data(), NULL);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start.data(), NULL, lshape.data(), NULL);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, serial ? H5FD_MPIO_INDEPENDENT : H5FD_MPIO_COLLECTIVE);
    
    herr_t status = H5Dwrite(dataset_id, datatype<T>(), subspace, filespace, plist_id, buf);
    if(status<0) PLEGMA_error("write_dataset: Unsuccessful writing of the dataset. Exiting\n");
    H5Sclose(subspace);
    H5Pclose(plist_id);
  }

  template<typename T>
  inline void _write_dataset_single(std::string name, T *buf, std::vector<hsize_t> shape, std::vector<hsize_t> start) {
    
    hid_t dataset_id = require_dataset<T>(name, shape);

    // In this function only one processor writes
    if(getRank() == 0) {
      bool needs_shift = false;
      T* tmp = buf;
      if(!start.empty()) for (auto i: start) if(i != 0) needs_shift = true;
      
      // Shifting the data accordingly to start
      if(needs_shift) {
	hostMalloc(tmp, product(shape)*sizeof(T));
	for(hsize_t i = 0; i<product(shape); i++) {
	  hsize_t j = to_id( add( from_id(i, shape), start), shape);
	  tmp[i] = buf[j];
	}
      }

      _write_dataset_parallel(dataset_id, tmp, shape, shape, zeros_like(shape), true);

      if(needs_shift) {
	hostFree(tmp, product(shape)*sizeof(T));
      }
    } else {
      _write_dataset_parallel(dataset_id, buf, shape, ones_like(shape), start.empty() ? zeros_like(shape) : start, true);
    }
    H5Dclose(dataset_id);
  }
  
  template<typename T>
  inline void _write_dataset_parallel(std::string name, T *buf, std::vector<hsize_t> shape, std::vector<hsize_t> lshape, std::vector<hsize_t> start) {
    
    hid_t dataset_id = require_dataset<T>(name, shape);

    // Counting how many writings we need to do
    size_t my_n_writings = 1, n_writings = 1;
    std::vector<int> exceeding_id;
    std::vector<hsize_t> exceeding_shape;
    for(size_t i=0; i<shape.size(); i++) {
      int exceeding = start[i] + lshape[i] - shape[i];
      if(exceeding > 0) { // then i it's exceeding
	if(HGC_verbosity > 2)
	  printf("rank %d: dir %d: exceeds of %d\n", comm_rank(), i, exceeding);
	exceeding_id.push_back(i);
	exceeding_shape.push_back(exceeding);
      } else {
	exceeding_shape.push_back(0);
      }
    }
    if(!exceeding_id.empty())
      my_n_writings = 1<<exceeding_id.size();
    MPI_Allreduce( &my_n_writings, &n_writings, 1, MPI_Type(n_writings), MPI_MAX, comm);
    if(HGC_verbosity > 2) PLEGMA_printf("%s: %d writing(s) are needed for writing the dataset\n", name.c_str(), n_writings);

    if(n_writings>1) {
      // looping over the writings 
      for(size_t i=0; i<n_writings; i++) {
	// standard behaviour
	T* tmp = buf;
	std::vector<hsize_t> tmp_lshape = lshape;
	std::vector<hsize_t> tmp_start = start;
	// creating the shifted case
	if(!exceeding_id.empty() && i < my_n_writings) {
	  std::vector<hsize_t> shift = zeros_like(start);
	  for(size_t j=0; j<lshape.size(); j++)
	    tmp_lshape[j] = lshape[j] - exceeding_shape[j];

	  // checking which direction we shift in this iteration
	  int j=0;
	  while((i>>j) > 0) {
	    if((i>>j) & 1) {
	      int id = exceeding_id[j];
	      shift[id] = tmp_lshape[id];
	      tmp_lshape[id] = exceeding_shape[id];
	      tmp_start[id] = 0;
	      if(HGC_verbosity > 2)
		printf("rank %d: iter %d: shifting id %d, shift[id] = %d, tmp_lshape[id] = %d\n", comm_rank(), i,
		       id, shift[id], tmp_lshape[id]);
	    }
	    j++;
	  }

	  // copying the part of the buffer to write
	  hostMalloc(tmp, product(tmp_lshape)*sizeof(T));
	  for(hsize_t i = 0; i<product(tmp_lshape); i++) {
	    hsize_t j = to_id( add( from_id(i, tmp_lshape), shift), lshape);
	    tmp[i] = buf[j];
	  }
	} else if(i >= my_n_writings) {
	  // do a dummy write to keep the communications active
	  tmp_lshape = ones_like(lshape);
	}
	_write_dataset_parallel(dataset_id, tmp, shape, tmp_lshape, tmp_start);
	if(tmp != buf) hostFree(tmp, product(tmp_lshape)*sizeof(T));
      }
    } else {
      _write_dataset_parallel(dataset_id, buf, shape, lshape, start);
    }

    H5Dclose(dataset_id);
  }

  bool isFileOpen( std::string filename){
    if( std::find(open_files.begin(), open_files.end(), filename) != open_files.end() )
      return true;
    else
      return false;
  }
  
public:
  /*
   * Creates or opens a path.
   * - If the path starts with "/" then is considered as an absolute path starting from the file
   * - Else it is considered as a relative path from the last location
   * Similar rules to bash cd are used (i.e. ../ ./ are implemented).
   */
  void cd(std::string path) {
    if(path=="-") return cd(prev_path);
    prev_path = pwd();
    if(path=="") return;
    if(path==".") return;
    else if(path=="/") go_top();
    auto vp = prepare_path(path);
    for (auto it = vp.begin(); it != vp.end(); it++) {
      if(*it == "..") go_back();
      else if(path==".") continue;
      else open(*it);
    }
  }

  /*
   * Creates or opens an existing HDF5 file.
   * - It checks if .h5 extension is given or adds it.
   * - It checks if path to a group has been given:
   *    i.e. name="./sample.h5/group1/group2" would create the file sample.h5 and
   *    then go to group1 and group2
   */
  HDF5(std::string name, MPI_Comm comm=MPI_COMM_WORLD) : comm(comm) {
    
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

    // Creating filename and path from name
    std::string path = "/";
    // checking if .h5 is given and at the end of file
    size_t ext = name.rfind(".h5");
    if(ext == std::string::npos || ext + 3 < name.length()) {
      // checking if followed by '/'
      if (ext != std::string::npos && name[ext+3] == '/') {
	filename = name.substr(0, ext+3);
	path = name.substr(ext+3);
      } else {
	// Adding it because either doesn't exist or it's something else
	filename = name+".h5";
      }
    } else {
      filename = name;
    }

    // check if filename is open by another instance
    while( isFileOpen(filename) )
      sleep(0.001);
    
    // checking if file exists or creating it
    if(access( filename.c_str(), F_OK ) != -1) {
      file_id = H5Fopen(filename.c_str(),  H5F_ACC_RDWR, fapl_id);
      if(HGC_verbosity > 2) PLEGMA_printf("Opened file %s\n", filename.c_str());
    } else {
      file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
      if(HGC_verbosity > 2) PLEGMA_printf("Created file %s\n", filename.c_str());
    }
    H5Pclose(fapl_id);

    if(path != "/") {
      cd(path);
    }
  }

  ~HDF5() {
    go_top();
    int open_obj = 1, my_open_obj = H5Fget_obj_count(file_id, H5F_OBJ_ALL);
    if(my_open_obj > 1) {
      printf("ERROR: rank %d has %d objects open. Closing will not work\n",
	     comm_rank(), (int) H5Fget_obj_count(file_id, H5F_OBJ_ALL));
    }
    MPI_Allreduce( &my_open_obj, &open_obj, 1, MPI_Type(open_obj), MPI_MAX, comm);
    if(open_obj > 1) {
      PLEGMA_error("More than one objects open. The closing will hang, so we crash the code here.");
    }
    H5Fclose(file_id);
    if(HGC_verbosity > 2) PLEGMA_printf("Closed file %s\n", filename.c_str());
    // remove opened file from vector
    std::vector<std::string>::iterator posix = std::find(open_files.begin(), open_files.end(), filename);
    if (posix != open_files.end())
      open_files.erase(posix);
  }

  /*
   * Writes an attribute with name attr_name and value attr_value in the current group or given by path
   * - If attr_name contains a path then it's splitted in path/attr_name
   */
  template<typename T>
  void write_attribute(std::string object, std::string attr_name, T attr_value,
		       std::string path=".") {
    // checking for / in attr_name
    size_t check = object.rfind("/");
    if(check != std::string::npos)
      return write_attribute(object.substr(check+1), attr_name, attr_value,
			     path+"/"+object.substr(0,check));
    if(HGC_verbosity > 2) PLEGMA_printf("Going to write attribute %s in path %s \n", attr_name.c_str(),
					path.c_str());
    cd(path);
    _write_attribute(object, attr_name, attr_value);
    if(HGC_verbosity > 2) PLEGMA_printf("%s: written attribute %s: %s\n", object.c_str(), attr_name.c_str(), attr_value.c_str());
    cd("-");
  }

  /*
   * Writes a dataset from buf with name in the current group or given by path
   * - If name contains a path then it's splitted in path/name
   * Shape, lshape and start are respectively the global and local shape and the starting point.
   * Periodic boundary conditions are applied and if start+lshape exceeds the global shape, 
   * the exceeding is written at the beginning.
   */
  template<typename T>
  void write_dataset(std::string name, T *buf, std::vector<hsize_t> shape,  std::vector<hsize_t> lshape={},
			    std::vector<hsize_t> start={}, std::string path=".") {
    // checking for / in name
    size_t check = name.rfind("/");
    if(check != std::string::npos)
      return write_dataset(name.substr(check+1), buf, shape, lshape, start,
			   (name[0]=='/' ? "/" : path)+"/"+name.substr(0,check));
    if(HGC_verbosity > 2) PLEGMA_printf("Going to write dataset %s in path %s \n", name.c_str(),
					path.c_str());
    cd(path);

    // Sanity check
    if( !lshape.empty() && lshape.size() != shape.size())
      PLEGMA_error("lshape has wrong size\n");
    if( !lshape.empty() && start.empty())
      PLEGMA_error("start cannot be empty in parallel writing\n");
    if( !start.empty() && start.size() != shape.size())
      PLEGMA_error("start has wrong size\n");

    if(exists(name)) {
      PLEGMA_warning("An object with name %s already exists in %s. Skipping...", name.c_str(),
		     pwd().c_str());
    } else {    
      int comm_size;
      MPI_Comm_size(comm, &comm_size);
      if(lshape.empty() || comm_size == 1)
	_write_dataset_single(name,buf,shape,start);
      else
	_write_dataset_parallel(name,buf,shape,lshape,start);

      if(HGC_verbosity > 2) PLEGMA_printf("Written dataset %s in %s mode\n", name.c_str(),
					  (lshape.empty() || comm_size == 1) ? "single" : "parallel");
    }
    cd("-");
  }

  template<typename T>
  void write_dataset(std::string name, std::vector<T> buf, std::vector<hsize_t> shape, std::string path=".") {
    if(buf.size() != product(shape)) {
      if(buf.size() % product(shape) == 0) {
	shape.insert(shape.begin(), buf.size()/product(shape));
      } else {
	PLEGMA_error("buf.size() is not multiple of shape\n");
      }
    }
    return write_dataset(name, buf.data(), shape, {}, {}, path);
  }
};
