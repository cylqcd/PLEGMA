#pragma once

#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <iostream>

#include <cupti.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#ifdef DEBUG
  template<typename... Args>
  void _LOG(const char *msg, Args&&... args) {
    fprintf(stderr, "[Log]: ");
    fprintf(stderr, msg, args...);
    fprintf(stderr, "\n");
  }
  void _LOG(const char *msg) {
    fprintf(stderr, "[Log]: %s\n", msg);
  }
  template<typename... Args>
  void _DBG(const char *msg, Args&&... args) {
    fprintf(stderr, msg, args...);
  }
  void _DBG(const char *msg) {
    fprintf(stderr, "%s", msg);
  }
#else
  #define _LOG(...)
  #define _DBG(...)
#endif

namespace cupti_profiler {
  static const char *dummy_kernel_name = "^^ DUMMY ^^";

namespace detail {

  // Pass-specific data
  struct pass_data_t {
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet *event_groups;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t num_events;
    // array of event ids
    std::vector<CUpti_EventID> event_ids;
    // array of event values
    std::vector<uint64_t> event_values;
  };

  struct kernel_data_t {
    typedef std::vector<uint64_t> event_val_t;
    typedef std::vector<CUpti_MetricValue> metric_val_t;

    kernel_data_t() : m_current_pass(0) {}

    std::vector<pass_data_t> m_pass_data;
    std::string m_name;

    int m_metric_passes;
    int m_event_passes;
    int m_current_pass;
    int m_total_passes;
    CUdevice m_device;

    event_val_t m_event_values;
    metric_val_t m_metric_values;

  };

  void CUPTIAPI
  get_value_callback(void *userdata,
                     CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid,
                     const CUpti_CallbackData *cbInfo);
  
  void print_metric(CUpti_MetricID& id,
                    CUpti_MetricValue& value,
                    std::ostream& s);

} // namespace detail

  struct profiler {
    typedef std::vector<std::string> strvec_t;
    using event_val_t = detail::kernel_data_t::event_val_t;
    using metric_val_t = detail::kernel_data_t::metric_val_t;

    profiler(const strvec_t& events,
             const strvec_t& metrics,
             const int device_num = 0);
    
    ~profiler() {}

    int get_passes() { return m_metric_passes + m_event_passes; }

    void start() {}

    void stop();
   
    void print_event_values(std::ostream& s,
                            bool print_names=true,
                            const char* kernel_separator = "; ");

    void print_metric_values(std::ostream& s,
                             bool print_names=true,
                             const char* kernel_separator = "; ");
    
    void print_events_and_metrics(std::ostream& s,
                                  bool print_names = true,
                                  const char* kernel_separator = "; ");
    
    std::vector<std::string> get_kernel_names();
    
    event_val_t
    get_event_values(const char *kernel_name);
    
    metric_val_t get_metric_values(const char *kernel_name);

  private:
    int m_device_num;
    int m_num_metrics, m_num_events;
    const strvec_t& m_event_names;
    const strvec_t& m_metric_names;
    std::vector<CUpti_MetricID> m_metric_ids;
    std::vector<CUpti_EventID> m_event_ids;

    CUcontext m_context;
    CUdevice m_device;
    CUpti_SubscriberHandle m_subscriber;

    CUpti_EventGroupSets *m_metric_pass_data;
    CUpti_EventGroupSets *m_event_pass_data;

    int m_metric_passes, m_event_passes;
    // Kernel-specific (indexed by name) trace data
    std::map<std::string,
             detail::kernel_data_t> m_kernel_data;
    std::vector<std::string> m_kernel_names;
    int m_num_kernels;
  };

#ifndef __CUPTI_PROFILER_NAME_SHORT
  #define __CUPTI_PROFILER_NAME_SHORT 128
#endif

  std::vector<std::string> available_metrics(CUdevice device);
  
  std::vector<std::string> available_events(CUdevice device);

} // namespace cupti_profiler
