#include <cupti_profiler.h>

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
#endif

namespace cupti_profiler {

namespace detail {

  void CUPTIAPI
  get_value_callback(void *userdata,
                     CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid,
                     const CUpti_CallbackData *cbInfo) {

    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
      fprintf(stderr, "%s:%d: Unexpected cbid %d\n", __FILE__, __LINE__, cbid);
      exit(-1);
    }

    const char *current_kernel_name = cbInfo->symbolName;

    // Skip execution if kernel name is NULL string
    // TODO: Make sure this is fine
    if(!current_kernel_name) {
      _LOG("Empty kernel name string. Skipping...");
      return;
    }

    std::map<std::string, detail::kernel_data_t> *kernel_data =
      (std::map<std::string, detail::kernel_data_t> *)userdata;

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      // If this is kernel name hasn't been seen before
      if(kernel_data->count(current_kernel_name) == 0) {
        _LOG("New kernel encountered: %s", current_kernel_name);

        detail::kernel_data_t dummy =
          (*kernel_data)[dummy_kernel_name];
        detail::kernel_data_t k_data = dummy;

        k_data.m_name = current_kernel_name;

        auto& pass_data = k_data.m_pass_data;

        CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                   CUPTI_EVENT_COLLECTION_MODE_KERNEL));

        for (int i = 0; i < pass_data[0].event_groups->numEventGroups; i++) {
          _LOG("  Enabling group %d", i);
          uint32_t all = 1;
          CUPTI_CALL(cuptiEventGroupSetAttribute(
                pass_data[0].event_groups->eventGroups[i],
                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                sizeof(all), &all));
          CUPTI_CALL(cuptiEventGroupEnable(
                pass_data[0].event_groups->eventGroups[i]));

          (*kernel_data)[current_kernel_name] = k_data;
        }
      } else {
        auto& current_kernel = (*kernel_data)[current_kernel_name];
        auto const& pass_data = current_kernel.m_pass_data;

        int current_pass = current_kernel.m_current_pass;
        if(current_pass >= current_kernel.m_total_passes)
          return;

        _LOG("Current pass for %s: %d", current_kernel_name, current_pass);

        CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
              CUPTI_EVENT_COLLECTION_MODE_KERNEL));

        for (int i = 0; i < pass_data[current_pass].event_groups->numEventGroups; i++) {
          _LOG("  Enabling group %d", i);
          uint32_t all = 1;
          CUPTI_CALL(cuptiEventGroupSetAttribute(
                pass_data[current_pass].event_groups->eventGroups[i],
                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                sizeof(all), &all));
          CUPTI_CALL(cuptiEventGroupEnable(
                pass_data[current_pass].event_groups->eventGroups[i]));

        }
      }
    } else if(cbInfo->callbackSite == CUPTI_API_EXIT) {
      auto& current_kernel = (*kernel_data)[current_kernel_name];
      int current_pass = current_kernel.m_current_pass;

      if(current_pass >= current_kernel.m_total_passes)
        return;

      auto& pass_data =
        current_kernel.m_pass_data[current_pass];

      for (int i = 0; i < pass_data.event_groups->numEventGroups; i++) {
        CUpti_EventGroup group = pass_data.event_groups->eventGroups[i];
        CUpti_EventDomainID group_domain;
        uint32_t numEvents, numInstances, numTotalInstances;
        CUpti_EventID *eventIds;
        size_t groupDomainSize = sizeof(group_domain);
        size_t numEventsSize = sizeof(numEvents);
        size_t numInstancesSize = sizeof(numInstances);
        size_t numTotalInstancesSize = sizeof(numTotalInstances);
        uint64_t *values, normalized, sum;
        size_t valuesSize, eventIdsSize;

        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
              &groupDomainSize, &group_domain));
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
              current_kernel.m_device, group_domain,
              CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
              &numTotalInstancesSize, &numTotalInstances));
        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
              &numInstancesSize, &numInstances));
        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
              &numEventsSize, &numEvents));
        eventIdsSize = numEvents * sizeof(CUpti_EventID);
        eventIds = (CUpti_EventID *)malloc(eventIdsSize);
        CUPTI_CALL(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENTS,
              &eventIdsSize, eventIds));

        valuesSize = sizeof(uint64_t) * numInstances;
        values = (uint64_t *)malloc(valuesSize);

        for(int j = 0; j < numEvents; j++) {
          CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                     eventIds[j], &valuesSize, values));
          /*if (metric_data->eventIdx >= metric_data->numEvents) {
            fprintf(stderr, "[error]: Too many events collected, metric expects only %d\n",
                (int)metric_data->numEvents);
            exit(-1);
          }*/

          // sum collect event values from all instances
          sum = 0;
          for(int k = 0; k < numInstances; k++)
            sum += values[k];

          // normalize the event value to represent the total number of
          // domain instances on the device
          normalized = (sum * numTotalInstances) / numInstances;

          pass_data.event_ids.push_back(eventIds[j]);
          pass_data.event_values.push_back(normalized);

          // print collected value
          {
            char eventName[128];
            size_t eventNameSize = sizeof(eventName) - 1;
            CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                       &eventNameSize, eventName));
            eventName[127] = '\0';
            _DBG("\t%s = %llu (", eventName, (unsigned long long)sum);
            if (numInstances > 1) {
              for (int k = 0; k < numInstances; k++) {
                if (k != 0)
                  _DBG(", ");
                _DBG("%llu", (unsigned long long)values[k]);
              }
            }

            _DBG(")\n");
            _LOG("\t%s (normalized) (%llu * %u) / %u = %llu",
                eventName, (unsigned long long)sum,
                numTotalInstances, numInstances,
                (unsigned long long)normalized);
          }
        }
        free(values);
      }

      for (int i = 0;
           i < pass_data.event_groups->numEventGroups;
           i++) {
        _LOG("  Disabling group %d", i);
        CUPTI_CALL(cuptiEventGroupDisable(
                   pass_data.event_groups->eventGroups[i]));
      }
      ++(*kernel_data)[current_kernel_name].m_current_pass;
    }
  }

  template<typename stream_t>
  void print_metric(CUpti_MetricID& id,
		    CUpti_MetricValue& value,
		    stream_t& s) {
    CUpti_MetricValueKind value_kind;
    size_t value_kind_sz = sizeof(value_kind);
    CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND,
                                       &value_kind_sz, &value_kind));
    switch(value_kind) {
    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
      s << value.metricValueDouble;
      break;
    case CUPTI_METRIC_VALUE_KIND_UINT64:
      s << value.metricValueUint64;
      break;
    case CUPTI_METRIC_VALUE_KIND_INT64:
      s << value.metricValueInt64;
      break;
    case CUPTI_METRIC_VALUE_KIND_PERCENT:
      s << value.metricValuePercent;
      break;
    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
      s << value.metricValueThroughput;
      break;
    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
      s << value.metricValueUtilizationLevel;
      break;
    default:
      std::cerr << "[error]: unknown value kind\n";
      exit(-1);
    }
  }

} // namespace detail

  int profiler::get_passes()
    { return m_metric_passes + m_event_passes; }

  void profiler::start(){    }

  void profiler::stop() {
      for(auto &k: m_kernel_data) {
        auto& data = k.second.m_pass_data;

        if(k.first == dummy_kernel_name)
          continue;

        int total_events = 0;
        for(int i = 0; i < m_metric_passes; ++i) {
          //total_events += m_metric_data[i].num_events;
          total_events += data[i].num_events;
        }
        CUpti_MetricValue metric_value;
        CUpti_EventID *event_ids = new CUpti_EventID[total_events];
        uint64_t *event_values = new uint64_t[total_events];

        int running_sum = 0;
        for(int i = 0; i < m_metric_passes; ++i) {
          std::copy(data[i].event_ids.begin(),
                    data[i].event_ids.end(),
                    event_ids + running_sum);
          std::copy(data[i].event_values.begin(),
                    data[i].event_values.end(),
                    event_values + running_sum);
          running_sum += data[i].num_events;
        }

        for(int i = 0; i < m_num_metrics; ++i) {
          CUptiResult _status = cuptiMetricGetValue(m_device,
                     m_metric_ids[i],
                     total_events * sizeof(CUpti_EventID),
                     event_ids,
                     total_events * sizeof(uint64_t),
                     event_values,
                     0, &metric_value);
          if(_status != CUPTI_SUCCESS) {
            fprintf(stderr, "Metric value retrieval failed for metric %s\n",
                    m_metric_names[i].c_str());
            exit(-1);
          }
          k.second.m_metric_values.push_back(metric_value);
        }

        delete[] event_ids;
        delete[] event_values;

        std::map<CUpti_EventID, uint64_t> event_map;
        for(int i = m_metric_passes;
            i < (m_metric_passes + m_event_passes);
            ++i) {
          for(int j = 0; j < data[i].num_events; ++j) {
            event_map[data[i].event_ids[j]] =
              data[i].event_values[j];
          }
        }

        for(int i = 0; i < m_num_events; ++i) {
          k.second.m_event_values.push_back(
              event_map[m_event_ids[i]]);
        }
      }

      // Disable callback and unsubscribe
      CUPTI_CALL(cuptiEnableCallback(0, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
      CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }

    template<typename stream>
    void profiler::print_event_values(stream& s,
                            bool print_names=true,
                            const char* kernel_separator = "; ") {
      using ull_t = unsigned long long;

      for(auto const& k: m_kernel_data) {
        if(k.first == dummy_kernel_name)
          continue;

        //printf("%s: ",
        //       m_kernel_data[k.first].m_name.c_str());

        /*for(int i = 0; i < m_num_events; ++i) {
          printf("Event [%s] = %llu\n",
              m_event_names[i].c_str(),
              (ull_t)m_kernel_data[k.first].m_event_values[i]);
        }
        printf("\n");*/

        if(m_num_events <= 0)
          return;

        for(int i = 0; i < m_num_events; ++i) {
          if(print_names)
            s << "(" << m_event_names[i] << ","
              << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << ") ";
          else
            s << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << " ";
        }
        s << kernel_separator;
      }
      printf("\n");
    }

    template<typename stream>
    void profiler::print_metric_values(stream& s,
                             bool print_names=true,
                             const char* kernel_separator = "; ") {
      if(m_num_metrics <= 0)
        return;

      for(auto const& k: m_kernel_data) {
        if(k.first == dummy_kernel_name)
          continue;

        //printf("%s: ",
        //       m_kernel_data[k.first].m_name.c_str());

        for(int i = 0; i < m_num_metrics; ++i) {
          if(print_names)
            s << "(" << m_metric_names[i] << ",";

          detail::print_metric(
            m_metric_ids[i],
            m_kernel_data[k.first].m_metric_values[i],
            s);

          if(print_names) s << ") ";
          else s << " ";
        }
        s << kernel_separator;
      }
      printf("\n");
    }

    template<typename stream>
    void profiler::print_events_and_metrics(stream& s,
                                  bool print_names = true,
                                  const char* kernel_separator = "; ") {
      if(m_num_events <= 0 && m_num_metrics <= 0)
        return;

      using ull_t = unsigned long long;
      for(auto const& k: m_kernel_data) {
        if(k.first == dummy_kernel_name)
          continue;

        //printf("New kernel: %s \n",
        //       m_kernel_data[k.first].m_name.c_str());

        for(int i = 0; i < m_num_events; ++i) {
          if(print_names)
            s << "(" << m_event_names[i] << ","
              << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << ") ";
          else
            s << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << " ";
        }

        for(int i = 0; i < m_num_metrics; ++i) {
          if(print_names)
            s << "(" << m_metric_names[i] << ",";

          detail::print_metric(
            m_metric_ids[i],
            m_kernel_data[k.first].m_metric_values[i],
            s);

          if(print_names) s << ") ";
          else s << " ";
        }

        s << kernel_separator;
      }
      printf("\n");
    }

  std::vector<std::string> profiler::get_kernel_names() {
      if(m_kernel_names.size() == 0) {
        for(auto const& k: m_kernel_data) {
          if(k.first == dummy_kernel_name)
            continue;
          m_kernel_names.push_back(k.first);
        }
      }
      return m_kernel_names;
    }

  std::vector<uint64_t>  profiler::get_event_values(const char *kernel_name) {
    if(m_num_events > 0)
      return m_kernel_data[kernel_name].m_event_values;
    else
      return event_val_t{};
  }

  std::vector<CUpti_MetricValue> profiler::get_metric_values(const char *kernel_name) {
      if(m_num_metrics > 0)
        return m_kernel_data[kernel_name].m_metric_values;
      else
        return metric_val_t{};
    }

} // namespace cupti_profiler
