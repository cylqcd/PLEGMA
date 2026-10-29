#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace plegma {

// Output selection is independent of integration checkpoints. The first
// 25 checkpoints retain the original 2.5*t0 schedule, including its rounding.
class FlowOutputSchedule {
public:
  static constexpr double epsilon = 0.02;
  static constexpr double checkpoint_t0 = 0.1;

  FlowOutputSchedule(double t0, double max_t0 = 2.5,
                     double save_after_t0 = -1.0, double save_every_t0 = 0.1)
      : t0_(t0), save_after_t0_(save_after_t0),
        checkpoints_(ticks(max_t0, "flow-max-t0 must be a nonnegative multiple of 0.1")),
        save_stride_(ticks(save_every_t0, "flow-save-every-t0 must be a positive multiple of 0.1")) {
    if (!std::isfinite(t0_) || t0_ <= 0.0)
      throw std::invalid_argument("t0-ref must be finite and positive");
    if (!std::isfinite(save_after_t0_) ||
        (save_after_t0_ < 0.0 && save_after_t0_ != -1.0))
      throw std::invalid_argument("flow-save-after-t0 must be -1 (no cutoff) or nonnegative");
    if (save_stride_ < 1)
      throw std::invalid_argument("flow-save-every-t0 must be at least 0.1");
    if (save_after_t0_ >= maxRatio())
      throw std::invalid_argument("flow-save-after-t0 must be smaller than flow-max-t0");
    if (checkpoints_ > 0 && checkpoint_t0 * t0_ < epsilon)
      throw std::invalid_argument("t0-ref is too small for distinct 0.1*t0 checkpoints at epsilon=0.02");
    legacy_steps_ = roundedStep(2.5 * t0_ / epsilon);
    if (checkpoints_ > 0) (void)targetStep(checkpoints_);
  }

  int checkpoints() const { return checkpoints_; }
  double maxRatio() const { return checkpoints_ * checkpoint_t0; }
  double nominalRatio(int checkpoint) const {
    checkIndex(checkpoint);
    return checkpoint * checkpoint_t0;
  }
  int targetStep(int checkpoint) const {
    checkIndex(checkpoint);
    if (checkpoint <= 25)
      return roundedStep(static_cast<double>(legacy_steps_) * checkpoint / 25);
    return roundedStep(nominalRatio(checkpoint) * t0_ / epsilon);
  }
  bool save(int checkpoint) const {
    checkIndex(checkpoint);
    const double nominal = nominalRatio(checkpoint);
    const double tolerance = 64 * std::numeric_limits<double>::epsilon()
                             * std::max(1.0, std::abs(save_after_t0_));
    if (save_after_t0_ >= 0.0 && nominal <= save_after_t0_ + tolerance)
      return false;
    return checkpoint == 0 || checkpoint == checkpoints_ || checkpoint % save_stride_ == 0;
  }

private:
  static int ticks(double ratio, const char *message) {
    if (!std::isfinite(ratio) || ratio < 0.0)
      throw std::invalid_argument(message);
    const double count = ratio / checkpoint_t0;
    if (!std::isfinite(count) || count > std::numeric_limits<int>::max())
      throw std::invalid_argument(message);
    const double nearest = std::round(count);
    const double tolerance = 64 * std::numeric_limits<double>::epsilon()
                             * std::max(1.0, std::abs(count));
    if (std::abs(count - nearest) > tolerance)
      throw std::invalid_argument(message);
    return static_cast<int>(nearest);
  }
  static int roundedStep(double value) {
    if (!std::isfinite(value) || value < 0.0 ||
        value > static_cast<double>(std::numeric_limits<int>::max()) - 0.5)
      throw std::invalid_argument("flow step count exceeds the supported integer range");
    return static_cast<int>(std::lround(value));
  }
  void checkIndex(int checkpoint) const {
    if (checkpoint < 0 || checkpoint > checkpoints_)
      throw std::out_of_range("flow checkpoint index is outside the schedule");
  }

  double t0_;
  double save_after_t0_;
  int checkpoints_;
  int save_stride_;
  int legacy_steps_ = 0;
};

} // namespace plegma
