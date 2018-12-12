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
void _LOG(const char *msg, Args&&... args);
void _LOG(const char *msg);

template<typename... Args>
void _DBG(const char *msg, Args&&... args);
void _DBG(const char *msg);

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

  template<typename stream_t>
  void print_metric(CUpti_MetricID& id,
                    CUpti_MetricValue& value,
                    stream_t& s);
 
} // namespace detail

  struct profiler {
    typedef std::vector<std::string> strvec_t;
    using event_val_t = detail::kernel_data_t::event_val_t;
    using metric_val_t = detail::kernel_data_t::metric_val_t;
    
    profiler(const strvec_t& events,
             const strvec_t& metrics,
             const int device_num = 0) :
      m_event_names(events),
      m_metric_names(metrics),
      m_device_num(device_num),
      m_num_metrics(metrics.size()),
      m_num_events(events.size()),
      m_metric_passes(0),
      m_event_passes(0) {
      
      int device_count = 0;
      
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
      DRIVER_API_CALL(cuInit(0));
      DRIVER_API_CALL(cuDeviceGetCount(&device_count));
      if (device_count == 0) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        exit(1);
      }

      m_metric_ids.resize(m_num_metrics);
      m_event_ids.resize(m_num_events);

      // Init device, context and setup callback
      DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
      DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
      CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                 (CUpti_CallbackFunc)detail::get_value_callback,
                 &m_kernel_data));
      CUPTI_CALL(cuptiEnableCallback(1, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));

      CUpti_MetricID metric_ids[m_num_metrics];
      for(int i = 0; i < m_num_metrics; ++i) {
        CUPTI_CALL(cuptiMetricGetIdFromName(m_device,
                   m_metric_names[i].c_str(),
                   &metric_ids[i]));
      }
      CUpti_EventID event_ids[m_num_events];
      for(int i = 0; i < m_num_events; ++i) {
        CUPTI_CALL(cuptiEventGetIdFromName(m_device,
                   m_event_names[i].c_str(),
                   &event_ids[i]));
      }

      if(m_num_metrics > 0) {
        CUPTI_CALL(cuptiMetricCreateEventGroupSets(m_context,
                   sizeof(metric_ids), metric_ids,
                   &m_metric_pass_data));
        m_metric_passes = m_metric_pass_data->numSets;

        std::copy(metric_ids, metric_ids + m_num_metrics,
                  m_metric_ids.begin());
      }
      if(m_num_events > 0) {
        CUPTI_CALL(cuptiEventGroupSetsCreate(m_context,
                   sizeof(event_ids), event_ids,
                   &m_event_pass_data));
        m_event_passes = m_event_pass_data->numSets;

        std::copy(event_ids, event_ids + m_num_events,
                  m_event_ids.begin());
      }

      _LOG("# Metric Passes: %d\n", m_metric_passes);
      _LOG("# Event Passes: %d\n", m_event_passes);

      assert((m_metric_passes + m_event_passes) > 0);

      detail::kernel_data_t dummy_data;
      dummy_data.m_name = dummy_kernel_name;
      dummy_data.m_metric_passes = m_metric_passes;
      dummy_data.m_event_passes = m_event_passes;
      dummy_data.m_device = m_device;
      dummy_data.m_total_passes = m_metric_passes + m_event_passes;
      dummy_data.m_pass_data.resize(m_metric_passes + m_event_passes);

      auto& pass_data = dummy_data.m_pass_data;
      for(int i = 0; i < m_metric_passes; ++i) {
        int total_events = 0;
        _LOG("[metric] Looking at set (pass) %d", i);
        uint32_t num_events = 0;
        size_t num_events_size = sizeof(num_events);
        for(int j = 0; j < m_metric_pass_data->sets[i].numEventGroups; ++j) {
          CUPTI_CALL(cuptiEventGroupGetAttribute(
                m_metric_pass_data->sets[i].eventGroups[j],
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &num_events_size, &num_events));
          _LOG("  Event Group %d, #Events = %d", j, num_events);
          total_events += num_events;
        }
        pass_data[i].event_groups = m_metric_pass_data->sets + i;
        pass_data[i].num_events = total_events;
      }

      for(int i = 0; i < m_event_passes; ++i) {
        int total_events = 0;
        _LOG("[event] Looking at set (pass) %d", i);
        uint32_t num_events = 0;
        size_t num_events_size = sizeof(num_events);
        for(int j = 0; j < m_event_pass_data->sets[i].numEventGroups; ++j) {
          CUPTI_CALL(cuptiEventGroupGetAttribute(
                m_event_pass_data->sets[i].eventGroups[j],
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &num_events_size, &num_events));
          _LOG("  Event Group %d, #Events = %d", j, num_events);
          total_events += num_events;
        }
        pass_data[i + m_metric_passes].event_groups = m_event_pass_data->sets + i;
        pass_data[i + m_metric_passes].num_events = total_events;
      }

      m_kernel_data[dummy_kernel_name] = dummy_data;
    }

    ~profiler() {
    }

    int get_passes();
    
    void start();

    void stop();
    
    template<typename stream>
    void print_event_values(stream& s,
                            bool print_names,
                            const char* kernel_separator);
    
    template<typename stream>
    void print_metric_values(stream& s,
                             bool print_names,
                             const char* kernel_separator);

    template<typename stream>
    void print_events_and_metrics(stream& s,
                                  bool print_names,
                                  const char* kernel_separator);
    
    std::vector<std::string> get_kernel_names();
    
    event_val_t get_event_values(const char *kernel_name);

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

} // namespace cupti_profiler
