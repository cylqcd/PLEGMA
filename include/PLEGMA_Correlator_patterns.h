#pragma once

#include <PLEGMA_Correlator.h>
#include <PLEGMA_io.h>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <cstdio>
#include <unistd.h>

using namespace plegma;

template<typename Float>
class PLEGMA_Correlator_patterns : public PLEGMA_Correlator<Float> {
  using Base = PLEGMA_Correlator<Float>;

public:
  using Base::Base;  // 继承构造函数

  void configurePatternMetadata(const std::string& group,
                              const std::string& dataset,
                              const std::string& descr,
                              bool isZfac = false)
  {
    if (isZfac)
      this->shape = {N_SPINS, N_SPINS, N_COLS, N_COLS, 1};
    else
      this->shape = {1};

    this->datasets = {dataset};
    this->groups = {group};
    this->description = descr;

    this->initialize();
  }

  // Write to an already-open HDF5 writer, avoiding the expensive file
  // open/close that writeHDF5(filename) performs on every call.
  // Inlines the use_multiple_writers logic from PLEGMA_Correlator.cu since
  // that function is not exposed in the header.
  void writeToHDF5Writer(HDF5& writer) const {
    // Ensure all ranks enter HDF5 simultaneously.  H5Gcreate is collective
    // in PHDF5, but H5Lexists (used inside cd()) is independent.  Without a
    // barrier, one rank may see the group as already existing (H5Gopen,
    // independent) while the other sees it as missing (H5Gcreate,
    // collective), causing a 30-second HDF5 collective-timeout.
    MPI_Barrier(HGC_fullComm);
    std::vector<hsize_t> h5shape, lshape, start;
    std::string descr = this->fill_H5_shapes(h5shape, lshape, start);

    hsize_t writeSize = 1;
    for (auto l : lshape) writeSize *= l;

    int nWriters = writeSize == 0 ? 0
                 : ((this->corr_space == MOMENTUM_SPACE) ? HGC_spaceSize : 1);
    int id = (this->corr_space == MOMENTUM_SPACE) ? HGC_spaceRank : 0;

    // Inline use_multiple_writers (static function in PLEGMA_Correlator.cu)
    size_t corrShift = 0;
    if (nWriters > 1) {
      int tmp_nw = nWriters;
      int tmp_id = id;
      int usedWriters = 1;
      for (int i = 0; i < (int)lshape.size(); i++) {
        int iSize    = ((int)lshape[i] + tmp_nw - 1) / tmp_nw;
        int iWriters = ((int)lshape[i] + iSize  - 1) / iSize;
        tmp_nw /= iWriters;
        usedWriters *= iWriters;
        int iId    = tmp_id % iWriters;
        tmp_id    /= iWriters;
        int iShift = iSize * iId;
        corrShift  = corrShift * lshape[i] + iShift;
        if (iShift + iSize > (int)lshape[i])
          lshape[i] -= iShift;
        else
          lshape[i]  = iSize;
        start[i] = (start[i] + iShift) % h5shape[i];
      }
      nWriters = usedWriters;
    }
    if (id >= nWriters) lshape[0] = 0;

    char ssource[64];
    snprintf(ssource, sizeof(ssource), "/sx%02dsy%02dsz%02dst%02d/",
             this->source[0], this->source[1], this->source[2], this->source[3]);
    std::string top = ssource;

    std::vector<hsize_t> momShape;
    std::vector<int> mvec;
    if (this->corr_space == MOMENTUM_SPACE) {
      momShape = { this->corr_mom_space->MomList()[0].size() };
      for (auto mv : this->corr_mom_space->MomList())
        for (auto m : mv) mvec.push_back((int)m);
    }

    for (size_t g = 0; g < this->nGroups(); g++) {
      writer.cd(top + (this->groups.size() > 0 ? this->groups[g] : "/"));
      if (this->corr_space == MOMENTUM_SPACE)
        writer.write_dataset("mvec", mvec, momShape);
      for (size_t d = 0; d < this->nDatasets(); d++) {
        Float *writeBuf = this->H_elem()
                        + (g * this->nDatasets() + d) * writeSize
                        + corrShift;
        std::string dataset = this->datasets.size() > 0 ? this->datasets[d] : "arr";
        writer.write_dataset(dataset, writeBuf, h5shape, lshape, start);
        writer.write_attribute(dataset, "description", descr);
      }
    }
  }

  // Serial (non-MPI) HDF5 write.  Ranks take turns (token-passing) so only one
  // rank holds the file open at a time.  No H5Pset_fapl_mpio → no collective
  // metadata ops → no Lustre MDT lock contention.
  void writeHDF5_serial(const std::string &filename) const {
    // ---- compute per-rank write geometry (mirrors PLEGMA_Correlator::writeHDF5) ----
    std::vector<hsize_t> shape, lshape, start;
    this->fill_H5_shapes(shape, lshape, start);

    // Size before use_multiple_writers modifies lshape (needed for buffer offset)
    hsize_t writeSize = 1;
    for (auto l : lshape) writeSize *= l;

    int nWriters = (writeSize == 0) ? 0
                 : ((this->corr_space == MOMENTUM_SPACE) ? HGC_spaceSize : 1);
    int id = (this->corr_space == MOMENTUM_SPACE) ? HGC_spaceRank : 0;

    // Inline use_multiple_writers (static in PLEGMA_Correlator.cu, not exported)
    size_t corrShift = 0;
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
        corrShift  = corrShift * lshape[i] + iShift;
        lshape[i]  = (iShift + iSize > (int)lshape[i]) ? lshape[i] - iShift : (hsize_t)iSize;
        start[i]   = (start[i] + iShift) % shape[i];
      }
      nWriters = usedWriters;
    }
    if (id >= nWriters) lshape[0] = 0;

    hsize_t thisWriteSize = 1;
    for (auto l : lshape) thisWriteSize *= l;

    int rank, nranks;
    MPI_Comm_rank(HGC_fullComm, &rank);
    MPI_Comm_size(HGC_fullComm, &nranks);

    // ---- token-passing: wait → write → pass ----
    if (rank > 0) {
      int token;
      MPI_Recv(&token, 1, MPI_INT, rank - 1, 99, HGC_fullComm, MPI_STATUS_IGNORE);
    }

    if (thisWriteSize > 0) {
      // Resolve .h5 filename (same logic as plegma::HDF5 constructor)
      std::string h5file = filename;
      size_t ext = h5file.rfind(".h5");
      if (ext == std::string::npos || ext + 3 < h5file.length())
        h5file += ".h5";

      hid_t fapl  = H5Pcreate(H5P_FILE_ACCESS);   // serial — no H5Pset_fapl_mpio
      hid_t file_id = (access(h5file.c_str(), F_OK) != -1)
          ? H5Fopen  (h5file.c_str(), H5F_ACC_RDWR,              fapl)
          : H5Fcreate(h5file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
      H5Pclose(fapl);

      char ssource[64];
      snprintf(ssource, sizeof(ssource), "sx%02dsy%02dsz%02dst%02d",
               this->source[0], this->source[1], this->source[2], this->source[3]);

      // Momentum vector written once (first rank that reaches the group)
      std::vector<int> mvec;
      std::vector<hsize_t> momShape;
      if (this->corr_space == MOMENTUM_SPACE) {
        momShape = { this->corr_mom_space->MomList()[0].size() };
        for (auto &mv : this->corr_mom_space->MomList())
          for (auto m : mv) mvec.push_back((int)m);
      }

      for (size_t g = 0; g < this->nGroups(); g++) {
        // Build absolute path components: ssource / group
        std::string full_path = std::string(ssource) + "/"
                              + (this->groups.size() > 0 ? this->groups[g] : "");

        // Navigate / create each path level
        hid_t cur = file_id;
        std::vector<hid_t> open_ids;
        std::istringstream ss_path(full_path);
        std::string part;
        while (std::getline(ss_path, part, '/')) {
          if (part.empty()) continue;
          hid_t gid = (H5Lexists(cur, part.c_str(), H5P_DEFAULT) > 0)
              ? H5Gopen  (cur, part.c_str(), H5P_DEFAULT)
              : H5Gcreate(cur, part.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          open_ids.push_back(gid);
          cur = gid;
        }

        // Write mvec once
        if (this->corr_space == MOMENTUM_SPACE
            && H5Lexists(cur, "mvec", H5P_DEFAULT) <= 0) {
          hid_t fsp = H5Screate_simple(1, momShape.data(), NULL);
          hid_t ds  = H5Dcreate(cur, "mvec", H5T_NATIVE_INT, fsp,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mvec.data());
          H5Dclose(ds);
          H5Sclose(fsp);
        }

        for (size_t d = 0; d < this->nDatasets(); d++) {
          const std::string &dsname = (this->datasets.size() > 0)
                                    ? this->datasets[d] : "arr";
          Float *writeBuf = this->H_elem()
                          + (g * this->nDatasets() + d) * writeSize
                          + corrShift;

          // Open existing dataset, or create with the correct full shape
          hid_t ds_id = -1;
          if (H5Lexists(cur, dsname.c_str(), H5P_DEFAULT) > 0) {
            ds_id = H5Dopen(cur, dsname.c_str(), H5P_DEFAULT);
            // If a wrong-shape stub exists (leftover pre-creation), delete & recreate
            hid_t fsp = H5Dget_space(ds_id);
            int ndims = H5Sget_simple_extent_ndims(fsp);
            std::vector<hsize_t> cur_shape(ndims);
            H5Sget_simple_extent_dims(fsp, cur_shape.data(), NULL);
            H5Sclose(fsp);
            if ((int)shape.size() != ndims || cur_shape != shape) {
              H5Dclose(ds_id);
              ds_id = -1;
              H5Ldelete(cur, dsname.c_str(), H5P_DEFAULT);
            }
          }
          if (ds_id < 0) {
            hid_t fsp = H5Screate_simple(shape.size(), shape.data(), NULL);
            ds_id = H5Dcreate(cur, dsname.c_str(), datatype<Float>(), fsp,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(fsp);
          }

          // Write our hyperslab portion of the dataset.
          // Handle time-wrapping: when the source is inside the local
          // lattice, the correlator buffer is contiguous in memory but wraps
          // around totalT in the HDF5 dataset (start[0]+lshape[0] > shape[0]).
          // Split into two non-wrapping writes in that case.
          hid_t filespace = H5Dget_space(ds_id);
          if (start[0] + lshape[0] > shape[0]) {
            // Part 1: from start[0] to shape[0]-1
            hsize_t part1_len = shape[0] - start[0];
            std::vector<hsize_t> ls1 = lshape, st1 = start;
            ls1[0] = part1_len;
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                st1.data(), NULL, ls1.data(), NULL);
            hid_t ms1 = H5Screate_simple(ls1.size(), ls1.data(), NULL);
            H5Dwrite(ds_id, datatype<Float>(), ms1, filespace,
                     H5P_DEFAULT, writeBuf);
            H5Sclose(ms1);

            // Part 2: from 0 to the remainder
            hsize_t part2_len = lshape[0] - part1_len;
            std::vector<hsize_t> ls2 = lshape, st2 = start;
            ls2[0] = part2_len;
            st2[0] = 0;
            // Compute buffer offset for part 2: skip part1_len slices
            hsize_t slice_size = 1;
            for (size_t ii = 1; ii < lshape.size(); ii++) slice_size *= lshape[ii];
            Float *buf2 = writeBuf + part1_len * slice_size;
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                st2.data(), NULL, ls2.data(), NULL);
            hid_t ms2 = H5Screate_simple(ls2.size(), ls2.data(), NULL);
            H5Dwrite(ds_id, datatype<Float>(), ms2, filespace,
                     H5P_DEFAULT, buf2);
            H5Sclose(ms2);
          } else {
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                start.data(), NULL, lshape.data(), NULL);
            hid_t memspace = H5Screate_simple(lshape.size(), lshape.data(), NULL);
            H5Dwrite(ds_id, datatype<Float>(), memspace, filespace,
                     H5P_DEFAULT, writeBuf);
            H5Sclose(memspace);
          }
          H5Sclose(filespace);
          H5Dclose(ds_id);
        }

        for (auto it = open_ids.rbegin(); it != open_ids.rend(); ++it)
          H5Gclose(*it);
      }
      H5Fclose(file_id);
    }  // thisWriteSize > 0

    // Pass token to next rank; last rank just waits at barrier
    if (rank < nranks - 1) {
      int token = 1;
      MPI_Send(&token, 1, MPI_INT, rank + 1, 99, HGC_fullComm);
    }
    MPI_Barrier(HGC_fullComm);
  }

  // ---------------------------------------------------------------
  // Batch mode: accumulate contraction results in host memory, then
  // write all of them to HDF5 in a single file-open/close pass.
  // ---------------------------------------------------------------
private:
  struct BatchEntry_ {
    std::string group;
    std::string dataset;
    std::string description;
    std::vector<Float> data;
  };
  std::vector<BatchEntry_> m_batch_;

public:
  /// Save current correlator buffer + metadata to the batch.
  /// Call this right after each contractNucleonThrp_patterns().
  void batchStore() {
    std::vector<hsize_t> h5shape, lshape, start;
    this->fill_H5_shapes(h5shape, lshape, start);
    hsize_t writeSize = 1;
    for (auto l : lshape) writeSize *= l;

    BatchEntry_ e;
    e.group       = (this->groups.size()   > 0) ? this->groups[0]   : "";
    e.dataset     = (this->datasets.size() > 0) ? this->datasets[0] : "arr";
    e.description = this->description;
    e.data.assign(this->H_elem(), this->H_elem() + writeSize);
    m_batch_.push_back(std::move(e));
  }

  size_t batchSize() const { return m_batch_.size(); }

  /// Write all accumulated entries to a single HDF5 file using
  /// serial token-passing (same semantics as writeHDF5_serial but
  /// the file is opened/closed only once per rank).
  void batchFlush(const std::string &filename) {
    if (m_batch_.empty()) {
      MPI_Barrier(HGC_fullComm);
      return;
    }

    // --- geometry (identical for every entry: same corr_space/src/maxQsq, shape={1}) ---
    std::vector<hsize_t> shape, lshape, start;
    this->fill_H5_shapes(shape, lshape, start);
    hsize_t writeSize = 1;
    for (auto l : lshape) writeSize *= l;

    int nWriters = (writeSize == 0) ? 0
                 : ((this->corr_space == MOMENTUM_SPACE) ? HGC_spaceSize : 1);
    int id = (this->corr_space == MOMENTUM_SPACE) ? HGC_spaceRank : 0;

    size_t corrShift = 0;
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
        corrShift  = corrShift * lshape[i] + iShift;
        lshape[i]  = (iShift + iSize > (int)lshape[i])
                       ? lshape[i] - iShift : (hsize_t)iSize;
        start[i]   = (start[i] + iShift) % shape[i];
      }
      nWriters = usedWriters;
    }
    if (id >= nWriters) lshape[0] = 0;

    hsize_t thisWriteSize = 1;
    for (auto l : lshape) thisWriteSize *= l;

    int rank, nranks;
    MPI_Comm_rank(HGC_fullComm, &rank);
    MPI_Comm_size(HGC_fullComm, &nranks);

    // ---- token-passing ----
    if (rank > 0) {
      int token;
      MPI_Recv(&token, 1, MPI_INT, rank - 1, 99, HGC_fullComm, MPI_STATUS_IGNORE);
    }

    if (thisWriteSize > 0) {
      std::string h5file = filename;
      {
        size_t ext = h5file.rfind(".h5");
        if (ext == std::string::npos || ext + 3 < h5file.length())
          h5file += ".h5";
      }

      hid_t fapl    = H5Pcreate(H5P_FILE_ACCESS);
      hid_t file_id = (access(h5file.c_str(), F_OK) != -1)
          ? H5Fopen  (h5file.c_str(), H5F_ACC_RDWR,              fapl)
          : H5Fcreate(h5file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
      H5Pclose(fapl);

      char ssource[64];
      snprintf(ssource, sizeof(ssource), "sx%02dsy%02dsz%02dst%02d",
               this->source[0], this->source[1], this->source[2], this->source[3]);

      // Momentum vector (shared by all entries)
      std::vector<int> mvec;
      std::vector<hsize_t> momShape;
      if (this->corr_space == MOMENTUM_SPACE) {
        momShape = { this->corr_mom_space->MomList()[0].size() };
        for (auto &mv : this->corr_mom_space->MomList())
          for (auto m : mv) mvec.push_back((int)m);
      }

      std::set<std::string> mvec_written;

      // ---------- loop over accumulated entries ----------
      for (const auto &entry : m_batch_) {
        std::string full_path = std::string(ssource) + "/" + entry.group;

        // Navigate / create group hierarchy
        hid_t cur = file_id;
        std::vector<hid_t> open_ids;
        {
          std::istringstream ss_path(full_path);
          std::string part;
          while (std::getline(ss_path, part, '/')) {
            if (part.empty()) continue;
            hid_t gid = (H5Lexists(cur, part.c_str(), H5P_DEFAULT) > 0)
                ? H5Gopen  (cur, part.c_str(), H5P_DEFAULT)
                : H5Gcreate(cur, part.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            open_ids.push_back(gid);
            cur = gid;
          }
        }

        // Write mvec once per group
        if (this->corr_space == MOMENTUM_SPACE
            && mvec_written.find(full_path) == mvec_written.end()
            && H5Lexists(cur, "mvec", H5P_DEFAULT) <= 0) {
          hid_t fsp = H5Screate_simple(1, momShape.data(), NULL);
          hid_t ds  = H5Dcreate(cur, "mvec", H5T_NATIVE_INT, fsp,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mvec.data());
          H5Dclose(ds);
          H5Sclose(fsp);
          mvec_written.insert(full_path);
        }

        // ---- write dataset ----
        const std::string &dsname = entry.dataset;
        const Float *writeBuf = entry.data.data() + corrShift;

        hid_t ds_id = -1;
        if (H5Lexists(cur, dsname.c_str(), H5P_DEFAULT) > 0) {
          ds_id = H5Dopen(cur, dsname.c_str(), H5P_DEFAULT);
          hid_t fsp = H5Dget_space(ds_id);
          int ndims = H5Sget_simple_extent_ndims(fsp);
          std::vector<hsize_t> cur_shape(ndims);
          H5Sget_simple_extent_dims(fsp, cur_shape.data(), NULL);
          H5Sclose(fsp);
          if ((int)shape.size() != ndims || cur_shape != shape) {
            H5Dclose(ds_id);
            ds_id = -1;
            H5Ldelete(cur, dsname.c_str(), H5P_DEFAULT);
          }
        }
        if (ds_id < 0) {
          hid_t fsp = H5Screate_simple(shape.size(), shape.data(), NULL);
          ds_id = H5Dcreate(cur, dsname.c_str(), datatype<Float>(), fsp,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          H5Sclose(fsp);
        }

        hid_t filespace = H5Dget_space(ds_id);
        if (start[0] + lshape[0] > shape[0]) {
          // Part 1
          hsize_t part1_len = shape[0] - start[0];
          std::vector<hsize_t> ls1 = lshape, st1 = start;
          ls1[0] = part1_len;
          H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                              st1.data(), NULL, ls1.data(), NULL);
          hid_t ms1 = H5Screate_simple(ls1.size(), ls1.data(), NULL);
          H5Dwrite(ds_id, datatype<Float>(), ms1, filespace,
                   H5P_DEFAULT, writeBuf);
          H5Sclose(ms1);

          // Part 2
          hsize_t part2_len = lshape[0] - part1_len;
          std::vector<hsize_t> ls2 = lshape, st2 = start;
          ls2[0] = part2_len;
          st2[0] = 0;
          hsize_t slice_size = 1;
          for (size_t ii = 1; ii < lshape.size(); ii++) slice_size *= lshape[ii];
          const Float *buf2 = writeBuf + part1_len * slice_size;
          H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                              st2.data(), NULL, ls2.data(), NULL);
          hid_t ms2 = H5Screate_simple(ls2.size(), ls2.data(), NULL);
          H5Dwrite(ds_id, datatype<Float>(), ms2, filespace,
                   H5P_DEFAULT, buf2);
          H5Sclose(ms2);
        } else {
          H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                              start.data(), NULL, lshape.data(), NULL);
          hid_t memspace = H5Screate_simple(lshape.size(), lshape.data(), NULL);
          H5Dwrite(ds_id, datatype<Float>(), memspace, filespace,
                   H5P_DEFAULT, writeBuf);
          H5Sclose(memspace);
        }
        H5Sclose(filespace);
        H5Dclose(ds_id);

        for (auto it = open_ids.rbegin(); it != open_ids.rend(); ++it)
          H5Gclose(*it);
      }  // entry loop

      H5Fclose(file_id);
    }  // thisWriteSize > 0

    if (rank < nranks - 1) {
      int token = 1;
      MPI_Send(&token, 1, MPI_INT, rank + 1, 99, HGC_fullComm);
    }
    MPI_Barrier(HGC_fullComm);

    m_batch_.clear();
  }
};