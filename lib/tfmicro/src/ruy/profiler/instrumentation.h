#ifndef RUY_PROFILER_INSTRUMENTATION_H_
#define RUY_PROFILER_INSTRUMENTATION_H_

namespace ruy {
namespace profiler {
class ScopeLabel {
 public:
  ScopeLabel(const char*) {}
  ~ScopeLabel() {}
};
}  // namespace profiler
}  // namespace ruy

#endif  // RUY_PROFILER_INSTRUMENTATION_H_