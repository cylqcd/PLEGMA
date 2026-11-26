#pragma once

#include <PLEGMA_QLoops.h>
#include <PLEGMA_FT.h>
#include <utils/PLEGMA_auxiliary.h>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <unistd.h>

using namespace plegma;

// ------------------------------------------------------------
// PLEGMA_QLoops_patterns
//
// Derived class parallel to PLEGMA_Correlator_patterns, used for
// disconnected quark loops with higher-order derivative patterns.
//
// PLEGMA_QLoops only has h_loc / h_oneD / h_twoD (up to 2nd order).
// This class extends it to store and output contractG5 results
// with arbitrary pattern labels.
//
// Usage:
//   1. Apply derivatives to spinors beforehand
//   2. Call oneEnd_trick() / contractG5() (inherited) to contract
//   3. Call dump() to do FT + write to file
// ------------------------------------------------------------
template<typename Float>
class PLEGMA_QLoops_patterns : public PLEGMA_QLoops<Float> {
  using Base = PLEGMA_QLoops<Float>;

protected:
  // pattern metadata
  std::string pat_group;
  std::string pat_dataset;
  std::string pat_description;

public:
  using Base::Base;  // 继承构造函数

  void configurePatternMetadata(const std::string& group,
                                const std::string& dataset,
                                const std::string& descr)
  {
    pat_group       = group;
    pat_dataset     = dataset;
    pat_description = descr;
  }

  const std::string& patternGroup()       const { return pat_group; }
  const std::string& patternDataset()     const { return pat_dataset; }
  const std::string& patternDescription() const { return pat_description; }

  // Write h_loc to HDF5 via FT, using the stored pattern metadata for HDF5 group path.
  // filenamePrefix : e.g. "./qloops_min_out/pat_stoch_part_std"
  // confID         : e.g. "0000"
  // format         : ASCII_FORMAT or HDF5_FORMAT
  // isc            : source index (-1 = no Ns tag)
  void dump(PLEGMA_FT<Float>& ft,
            const std::string& filenamePrefix,
            const std::string& confID,
            FILE_FORMAT format,
            int isc = -1)
  {
    using sv = std::vector<std::string>;
    std::string suffix = (format == ASCII_FORMAT) ? ".dat" : ".h5";

    std::string fname_base;
    if (isc >= 0)
      fname_base = join(sv({"Conf" + confID,
                            "Ns" + std::to_string(isc)}), "/");
    else
      fname_base = join(sv({"Conf" + confID}), "/");

    std::string fname = fname_base
                      + join(sv({pat_group, pat_dataset}), "/");

    this->load(this->H_loc());
    ft.apply(*this, FT_GEMV);
    ft.writeFile((format == HDF5_FORMAT)
                     ? filenamePrefix + suffix + fname
                     : filenamePrefix + findAndReplace(fname, '/', '_') + suffix,
                 format);
  }

  // Same as dump() but uses token-passing serial HDF5 writes to avoid
  // Lustre MDT lock contention from parallel H5Gcreate / H5Dcreate calls.
  // For ASCII format, falls back to dump().
  void dump_serial(PLEGMA_FT<Float>& ft,
                   const std::string& filenamePrefix,
                   const std::string& confID,
                   FILE_FORMAT format,
                   int isc = -1)
  {
    if (format != HDF5_FORMAT) {
      dump(ft, filenamePrefix, confID, format, isc);
      return;
    }

    using sv = std::vector<std::string>;
    std::string suffix = ".h5";

    std::string fname_base;
    if (isc >= 0)
      fname_base = join(sv({"Conf" + confID, "Ns" + std::to_string(isc)}), "/");
    else
      fname_base = join(sv({"Conf" + confID}), "/");

    std::string fname = fname_base + join(sv({pat_group, pat_dataset}), "/");

    this->load(this->H_loc());
    ft.apply(*this, FT_GEMV);

    // Build PLEGMA-style fullpath: "prefix.h5/Conf.../Group/Dataset"
    std::string fullpath = filenamePrefix + suffix + fname;

    // Parse: h5file = "prefix.h5", group_path = "Conf.../Group", dataset_name = "Dataset"
    size_t ext_pos = fullpath.rfind(".h5");
    std::string h5file = fullpath.substr(0, ext_pos + 3);
    std::string remainder = (ext_pos + 3 < fullpath.size()) ? fullpath.substr(ext_pos + 4) : "";

    std::string dataset_name = "FT_data";
    std::string group_path;
    if (!remainder.empty()) {
      size_t last_slash = remainder.rfind("/");
      if (last_slash != std::string::npos && last_slash + 1 < remainder.size()) {
        dataset_name = remainder.substr(last_slash + 1);
        group_path   = remainder.substr(0, last_slash);
      } else if (last_slash == std::string::npos) {
        dataset_name = remainder;
      } else {
        group_path = remainder.substr(0, last_slash);
      }
    }

    // Get data layout (fill_H5_shapes is public in PLEGMA_FT)
    //   - dims==3 with time decomposition: shape=[T,nMom,2], each rank gets its localT slice
    //   - use_multiple_writers is a no-op when HGC_spaceSize==1 (nWriters=1)
    std::vector<hsize_t> shape, lshape, start;
    std::string descr = ft.fill_H5_shapes(shape, lshape, start, 0);

    const Float* data_ptr = ft.H_elem();

    // mvec
    std::vector<int>    mvec;
    std::vector<hsize_t> momShape = { (hsize_t)ft.Nmoms() };
    for (auto& mv : ft.MomList())
      for (auto m : mv) mvec.push_back((int)m);

    // ---- token-passing serial write ----
    int rank, nranks;
    MPI_Comm_rank(HGC_fullComm, &rank);
    MPI_Comm_size(HGC_fullComm, &nranks);

    if (rank > 0) {
      int token;
      MPI_Recv(&token, 1, MPI_INT, rank - 1, 98, HGC_fullComm, MPI_STATUS_IGNORE);
    }

    {
      // Open / create file with serial HDF5 (no H5Pset_fapl_mpio)
      hid_t fapl    = H5Pcreate(H5P_FILE_ACCESS);
      hid_t file_id = (access(h5file.c_str(), F_OK) != -1)
          ? H5Fopen  (h5file.c_str(), H5F_ACC_RDWR,              fapl)
          : H5Fcreate(h5file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
      H5Pclose(fapl);

      // Navigate / create group hierarchy
      hid_t cur = file_id;
      std::vector<hid_t> open_groups;
      if (!group_path.empty()) {
        std::istringstream ss(group_path);
        std::string part;
        while (std::getline(ss, part, '/')) {
          if (part.empty()) continue;
          hid_t g = (H5Lexists(cur, part.c_str(), H5P_DEFAULT) > 0)
              ? H5Gopen  (cur, part.c_str(), H5P_DEFAULT)
              : H5Gcreate(cur, part.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          open_groups.push_back(g);
          cur = g;
        }
      }

      // Write mvec once (first rank to reach here creates it)
      if (H5Lexists(cur, "mvec", H5P_DEFAULT) <= 0) {
        hid_t fsp = H5Screate_simple(1, momShape.data(), NULL);
        hid_t ds  = H5Dcreate(cur, "mvec", H5T_NATIVE_INT, fsp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mvec.data());
        H5Dclose(ds);
        H5Sclose(fsp);
      }

      // Create dataset on first write, open on subsequent writes
      hid_t ds_id;
      if (H5Lexists(cur, dataset_name.c_str(), H5P_DEFAULT) > 0) {
        ds_id = H5Dopen(cur, dataset_name.c_str(), H5P_DEFAULT);
      } else {
        hid_t fsp = H5Screate_simple(shape.size(), shape.data(), NULL);
        ds_id = H5Dcreate(cur, dataset_name.c_str(), datatype<Float>(), fsp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(fsp);
      }

      // Write description attribute (once)
      if (H5Aexists(ds_id, "description") <= 0) {
        hid_t str_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type, descr.size() + 1);
        hid_t aspace   = H5Screate(H5S_SCALAR);
        hid_t attr_id  = H5Acreate(ds_id, "description", str_type, aspace,
                                    H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr_id, str_type, descr.c_str());
        H5Aclose(attr_id);
        H5Sclose(aspace);
        H5Tclose(str_type);
      }

      // Write this rank's hyperslab
      hid_t filespace = H5Dget_space(ds_id);
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                           start.data(), NULL, lshape.data(), NULL);
      hid_t memspace = H5Screate_simple(lshape.size(), lshape.data(), NULL);
      H5Dwrite(ds_id, datatype<Float>(), memspace, filespace, H5P_DEFAULT, data_ptr);
      H5Sclose(memspace);
      H5Sclose(filespace);
      H5Dclose(ds_id);

      for (auto it = open_groups.rbegin(); it != open_groups.rend(); ++it)
        H5Gclose(*it);
      H5Fclose(file_id);
    }

    if (rank < nranks - 1) {
      int token = 1;
      MPI_Send(&token, 1, MPI_INT, rank + 1, 98, HGC_fullComm);
    }
    MPI_Barrier(HGC_fullComm);
  }

  // ---------------------------------------------------------------
  // Batch mode: accumulate FT results in host memory, then write
  // all of them to HDF5 in a single file-open/close pass.
  // Mirrors PLEGMA_Correlator_patterns::batchStore / batchFlush.
  // ---------------------------------------------------------------
protected:
  struct BatchEntry_ {
    std::string group;
    std::string dataset;
    std::vector<Float> data;
  };
  std::vector<BatchEntry_> m_batch_;

public:
  /// FT the current H_loc and store the result to the batch.
  /// If accumulate=true and an entry with the same (group,dataset) key
  /// already exists, element-wise ADD to it (for spin-color dilution).
  /// The pattern group/dataset are read from the metadata set by
  /// configurePatternMetadata() (called inside contractG5_patterns).
  void batchStore(PLEGMA_FT<Float>& ft, bool accumulate = false)
  {
    this->load(this->H_loc());
    ft.apply(*this, FT_GEMV);

    std::vector<hsize_t> shape, lshape, start;
    ft.fill_H5_shapes(shape, lshape, start, 0);
    hsize_t n = 1;
    for (auto l : lshape) n *= l;

    if (accumulate) {
      for (auto& e : m_batch_) {
        if (e.group == pat_group && e.dataset == pat_dataset) {
          const Float* src = ft.H_elem();
          for (hsize_t j = 0; j < n; ++j) e.data[j] += src[j];
          return;
        }
      }
    }

    BatchEntry_ e;
    e.group   = pat_group;
    e.dataset = pat_dataset;
    e.data.assign(ft.H_elem(), ft.H_elem() + n);
    m_batch_.push_back(std::move(e));
  }

  size_t batchSize() const { return m_batch_.size(); }

  /// Write all accumulated batch entries to HDF5 in one file-open pass
  /// using token-passing serial I/O (same as dump_serial pattern).
  void batchFlush(PLEGMA_FT<Float>& ft,
                  const std::string& filenamePrefix,
                  const std::string& confID,
                  FILE_FORMAT format,
                  int isc = -1)
  {
    if (m_batch_.empty()) {
      MPI_Barrier(HGC_fullComm);
      return;
    }

    using sv = std::vector<std::string>;
    std::string suffix = ".h5";

    std::string fname_base;
    if (isc >= 0)
      fname_base = join(sv({"Conf" + confID, "Ns" + std::to_string(isc)}), "/");
    else
      fname_base = join(sv({"Conf" + confID}), "/");

    // FT geometry (same for all entries — same FT object)
    std::vector<hsize_t> shape, lshape, start;
    std::string descr = ft.fill_H5_shapes(shape, lshape, start, 0);
    hsize_t n = 1;
    for (auto l : lshape) n *= l;
    if (n == 0) {
      MPI_Barrier(HGC_fullComm);
      m_batch_.clear();
      return;
    }

    // mvec
    std::vector<int> mvec;
    std::vector<hsize_t> momShape = { (hsize_t)ft.Nmoms() };
    for (auto& mv : ft.MomList())
      for (auto m : mv) mvec.push_back((int)m);

    int rank, nranks;
    MPI_Comm_rank(HGC_fullComm, &rank);
    MPI_Comm_size(HGC_fullComm, &nranks);

    // ---- token-passing serial write ----
    if (rank > 0) {
      int token;
      MPI_Recv(&token, 1, MPI_INT, rank - 1, 98, HGC_fullComm, MPI_STATUS_IGNORE);
    }

    {
      std::string h5file = filenamePrefix + suffix;

      // Ensure parent directory exists (H5Fcreate won't create directories)
      {
        size_t last_slash = h5file.rfind('/');
        if (last_slash != std::string::npos) {
          std::string dir = h5file.substr(0, last_slash);
          if (!dir.empty() && access(dir.c_str(), F_OK) == -1) {
            std::string cmd = "mkdir -p " + dir;
            if (rank == 0) system(cmd.c_str());
            MPI_Barrier(HGC_fullComm);
          }
        }
      }

      hid_t fapl    = H5Pcreate(H5P_FILE_ACCESS);
      hid_t file_id = (access(h5file.c_str(), F_OK) != -1)
          ? H5Fopen  (h5file.c_str(), H5F_ACC_RDWR,              fapl)
          : H5Fcreate(h5file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
      H5Pclose(fapl);

      if (file_id < 0) {
        PLEGMA_printf("ERROR: batchFlush could not open/create HDF5 file: %s\n",
                      h5file.c_str());
        // Pass token and bail out — don't cascade invalid-ID errors
        if (rank < nranks - 1) {
          int token = 1;
          MPI_Send(&token, 1, MPI_INT, rank + 1, 98, HGC_fullComm);
        }
        MPI_Barrier(HGC_fullComm);
        m_batch_.clear();
        return;
      }

      std::set<std::string> mvec_written;

      for (const auto& entry : m_batch_) {
        std::string full_path = fname_base + join(sv({entry.group}), "/");

        // Navigate / create group hierarchy
        hid_t cur = file_id;
        std::vector<hid_t> open_ids;
        {
          std::istringstream ss(full_path);
          std::string part;
          while (std::getline(ss, part, '/')) {
            if (part.empty()) continue;
            hid_t g = (H5Lexists(cur, part.c_str(), H5P_DEFAULT) > 0)
                ? H5Gopen  (cur, part.c_str(), H5P_DEFAULT)
                : H5Gcreate(cur, part.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (g < 0) {
              PLEGMA_printf("ERROR: batchFlush failed to open/create group '%s' in path '%s'\n",
                            part.c_str(), full_path.c_str());
              break;
            }
            open_ids.push_back(g);
            cur = g;
          }
        }

        // Write mvec once per group
        if (mvec_written.find(full_path) == mvec_written.end()
            && H5Lexists(cur, "mvec", H5P_DEFAULT) <= 0) {
          hid_t fsp = H5Screate_simple(1, momShape.data(), NULL);
          hid_t ds  = H5Dcreate(cur, "mvec", H5T_NATIVE_INT, fsp,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mvec.data());
          H5Dclose(ds);
          H5Sclose(fsp);
          mvec_written.insert(full_path);
        }

        // Write dataset
        const std::string& dsname = entry.dataset;
        hid_t ds_id;
        if (H5Lexists(cur, dsname.c_str(), H5P_DEFAULT) > 0) {
          ds_id = H5Dopen(cur, dsname.c_str(), H5P_DEFAULT);
        } else {
          hid_t fsp = H5Screate_simple(shape.size(), shape.data(), NULL);
          ds_id = H5Dcreate(cur, dsname.c_str(), datatype<Float>(), fsp,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          H5Sclose(fsp);
        }

        hid_t filespace = H5Dget_space(ds_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                            start.data(), NULL, lshape.data(), NULL);
        hid_t memspace = H5Screate_simple(lshape.size(), lshape.data(), NULL);
        H5Dwrite(ds_id, datatype<Float>(), memspace, filespace,
                 H5P_DEFAULT, entry.data.data());
        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Dclose(ds_id);

        for (auto it = open_ids.rbegin(); it != open_ids.rend(); ++it)
          H5Gclose(*it);
      }

      H5Fclose(file_id);
    }

    if (rank < nranks - 1) {
      int token = 1;
      MPI_Send(&token, 1, MPI_INT, rank + 1, 98, HGC_fullComm);
    }
    MPI_Barrier(HGC_fullComm);

    if (HGC_verbosity >= 2)
      PLEGMA_printf("QLoops_patterns batchFlush: %zu entries written to %s%s\n",
                    m_batch_.size(), filenamePrefix.c_str(), suffix.c_str());
    m_batch_.clear();
  }

};

// ---------------------------------------------------------------------------
// writeFT_serial
//
// Token-passing serial HDF5 write for a PLEGMA_FT<double>.
// Replaces ft->writeFile(fullpath, HDF5_FORMAT) to avoid Lustre MDT lock
// contention from parallel H5Gcreate / H5Dcreate calls.
// ---------------------------------------------------------------------------
inline void writeFT_serial(PLEGMA_FT<double>* ft, const std::string& fullpath)
{
    std::vector<hsize_t> shape, lshape, start;
    std::string descr = ft->fill_H5_shapes(shape, lshape, start, 0);

    int nWriters = (ft->Dims() == 4) ? HGC_fullSize : HGC_spaceSize;
    int id       = (ft->Dims() == 4) ? HGC_fullRank : HGC_spaceRank;

    // Inline use_multiple_writers (static in PLEGMA_FT.cu, not exported)
    size_t shift = 0;
    if (nWriters > 1) {
        int tmp_nw = nWriters, tmp_id = id, usedWriters = 1;
        for (size_t i = 0; i < lshape.size(); i++) {
            int iSize    = ((int)lshape[i] + tmp_nw - 1) / tmp_nw;
            int iWriters = ((int)lshape[i] + iSize  - 1) / iSize;
            tmp_nw      /= iWriters;
            usedWriters *= iWriters;
            int iId    = tmp_id % iWriters;
            tmp_id    /= iWriters;
            int iShift = iSize * iId;
            shift      = shift * lshape[i] + iShift;
            lshape[i]  = (iShift + iSize > (int)lshape[i]) ? lshape[i] - iShift : (hsize_t)iSize;
            start[i]   = (start[i] + iShift) % shape[i];
        }
        nWriters = usedWriters;
    }
    if (id >= nWriters) lshape[0] = 0;

    hsize_t thisWriteSize = 1;
    for (auto l : lshape) thisWriteSize *= l;

    // Parse PLEGMA filename: "prefix.h5/group/.../dataset"
    std::string h5file = fullpath;
    std::string group_path, dataset_name = "FT_data";
    size_t ext = fullpath.rfind(".h5");
    if (ext != std::string::npos && ext + 3 < fullpath.size() && fullpath[ext+3] == '/') {
        size_t last = fullpath.rfind("/");
        if (last + 1 < fullpath.size()) {
            dataset_name = fullpath.substr(last + 1);
            h5file       = fullpath.substr(0, ext + 3);
            std::string remainder = fullpath.substr(ext + 4);
            size_t ds_slash = remainder.rfind("/");
            group_path = (ds_slash != std::string::npos) ? remainder.substr(0, ds_slash) : "";
        }
    }

    std::vector<int>     mvec;
    std::vector<hsize_t> momShape = { (hsize_t)ft->Dims() };
    for (auto& mv : ft->MomList()) for (auto m : mv) mvec.push_back((int)m);

    int rank, nranks;
    MPI_Comm_rank(HGC_fullComm, &rank);
    MPI_Comm_size(HGC_fullComm, &nranks);

    if (rank > 0) {
        int token;
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 97, HGC_fullComm, MPI_STATUS_IGNORE);
    }

    if (thisWriteSize > 0) {
        hid_t fapl    = H5Pcreate(H5P_FILE_ACCESS);
        hid_t file_id = (access(h5file.c_str(), F_OK) != -1)
            ? H5Fopen  (h5file.c_str(), H5F_ACC_RDWR,              fapl)
            : H5Fcreate(h5file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);

        hid_t cur = file_id;
        std::vector<hid_t> open_groups;
        if (!group_path.empty()) {
            std::istringstream ss(group_path);
            std::string part;
            while (std::getline(ss, part, '/')) {
                if (part.empty()) continue;
                hid_t g = (H5Lexists(cur, part.c_str(), H5P_DEFAULT) > 0)
                    ? H5Gopen  (cur, part.c_str(), H5P_DEFAULT)
                    : H5Gcreate(cur, part.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                open_groups.push_back(g);
                cur = g;
            }
        }

        if (H5Lexists(cur, "mvec", H5P_DEFAULT) <= 0) {
            hsize_t mvec_total = (hsize_t)mvec.size();
            hsize_t base       = momShape[0];
            std::vector<hsize_t> ms = (mvec_total != base)
                ? std::vector<hsize_t>{mvec_total / base, base} : momShape;
            hid_t fsp = H5Screate_simple(ms.size(), ms.data(), NULL);
            hid_t ds  = H5Dcreate(cur, "mvec", H5T_NATIVE_INT, fsp,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mvec.data());
            H5Dclose(ds); H5Sclose(fsp);
        }

        hid_t ds_id;
        if (H5Lexists(cur, dataset_name.c_str(), H5P_DEFAULT) > 0) {
            ds_id = H5Dopen(cur, dataset_name.c_str(), H5P_DEFAULT);
        } else {
            hid_t fsp = H5Screate_simple(shape.size(), shape.data(), NULL);
            ds_id = H5Dcreate(cur, dataset_name.c_str(), datatype<double>(), fsp,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(fsp);
        }

        if (H5Aexists(ds_id, "description") <= 0) {
            hid_t str_type = H5Tcopy(H5T_C_S1);
            H5Tset_size(str_type, descr.size() + 1);
            hid_t aspace  = H5Screate(H5S_SCALAR);
            hid_t attr_id = H5Acreate(ds_id, "description", str_type, aspace,
                                       H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attr_id, str_type, descr.c_str());
            H5Aclose(attr_id); H5Sclose(aspace); H5Tclose(str_type);
        }

        hid_t filespace = H5Dget_space(ds_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start.data(), NULL, lshape.data(), NULL);
        hid_t memspace = H5Screate_simple(lshape.size(), lshape.data(), NULL);
        H5Dwrite(ds_id, datatype<double>(), memspace, filespace, H5P_DEFAULT, ft->H_elem() + shift);
        H5Sclose(memspace); H5Sclose(filespace); H5Dclose(ds_id);

        for (auto it = open_groups.rbegin(); it != open_groups.rend(); ++it)
            H5Gclose(*it);
        H5Fclose(file_id);
    }

    if (rank < nranks - 1) {
        int token = 1;
        MPI_Send(&token, 1, MPI_INT, rank + 1, 97, HGC_fullComm);
    }
    MPI_Barrier(HGC_fullComm);
}

// ---------------------------------------------------------------------------
// dumpLoops
//
// Write all loop components (local, oneD, twoD) of a PLEGMA_QLoops<double>
// to file after applying the Fourier transform. Uses writeFT_serial for
// HDF5 to avoid Lustre MDT lock contention.
// ---------------------------------------------------------------------------
inline void dumpLoops(PLEGMA_QLoops<double>& qLoops, PLEGMA_FT<double>* ft[2],
                      std::string filenamePrefix, std::string confID,
                      FILE_FORMAT format, int isc = -1)
{
    using sv = std::vector<std::string>;
    if (format != ASCII_FORMAT && format != HDF5_FORMAT)
        PLEGMA_error("This executable can write only in ascii and hdf5 format");
    std::string suffix = (format == ASCII_FORMAT) ? ".dat" : ".h5";

    std::string fname_base;
    if (isc >= 0)
        fname_base = join(sv({"Conf" + confID, "Ns" + std::to_string(isc)}), "/");
    else
        fname_base = join(sv({"Conf" + confID}), "/");

    auto doWrite = [&](PLEGMA_FT<double>* ft_ptr, const std::string& fname_inner) {
        if (format == HDF5_FORMAT)
            writeFT_serial(ft_ptr, filenamePrefix + suffix + fname_inner);
        else
            ft_ptr->writeFile(filenamePrefix + findAndReplace(fname_inner, '/', '_') + suffix, format);
    };

    qLoops.load(qLoops.H_loc());
    ft[0]->apply(qLoops, FT_GEMV);
    doWrite(ft[0], fname_base + join(sv({"localLoops", "loop"}), "/"));

    if (qLoops.IsOneD())
        for (int mu = 0; mu < N_DIMS; mu++) {
            std::string fnameOneD  = fname_base + join(sv({"oneD",  "dir" + std::to_string(mu), "loop"}), "/");
            std::string fnameOneDC = fname_base + join(sv({"oneDC", "dir" + std::to_string(mu), "loop"}), "/");
            qLoops.load(qLoops.H_oneD()[mu]);
            ft[0]->apply(qLoops, FT_GEMV);
            ft[0]->scale(0.25);
            doWrite(ft[0], fnameOneD);
            qLoops.load(qLoops.H_oneDC()[mu]);
            ft[0]->apply(qLoops, FT_GEMV);
            ft[0]->scale(0.25);
            doWrite(ft[0], fnameOneDC);
        }

    int count = 0;
    if (qLoops.IsTwoD())
        for (auto munu : qLoops.get_twoD_index()) {
            int mu = std::get<0>(munu), nu = std::get<1>(munu);
            std::string fnameTwoD = fname_base + join(sv({"twoD", "dirs" + std::to_string(mu) + std::to_string(nu), "loop"}), "/");
            qLoops.load(qLoops.H_twoD()[count]);
            ft[1]->apply(qLoops, FT_GEMV);
            ft[1]->scale((mu != 3 && nu != 3) ? 0.25 : 0.125);
            doWrite(ft[1], fnameTwoD);
            count++;
        }
}
