#ifndef SHORTFIN_LOCAL_SYSTEMS_CUDA_H
#define SHORTFIN_LOCAL_SYSTEMS_CUDA_H

#include <vector>

#include "iree/hal/drivers/cuda/api.h"
#include "shortfin/local/system.h"
#include "shortfin/local/systems/host.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local::systems {

class SHORTFIN_API CUDADevice : public Device {
 public:
  using Device::Device;
};

class SHORTFIN_API CUDASystemBuilder : public HostCPUSystemBuilder {
 public:
  CUDASystemBuilder(iree_allocator_t host_allocator,
                    ConfigOptions options = {});
  CUDASystemBuilder() : CUDASystemBuilder(iree_allocator_system()) {}
  ~CUDASystemBuilder();

  SystemPtr CreateSystem() override;

  bool &cpu_devices_enabled() { return cpu_devices_enabled_; }

  std::optional<std::vector<std::string>> &visible_devices() {
    return visible_devices_;
  };

  std::vector<std::string> &cuda_allocator_specs() {
    return cuda_allocator_specs_;
  }

  // Whether to use async allocations if the device supports them (default
  // true). There are various reasons to disable this in different usage
  // scenarios.
  bool &async_allocations() { return default_device_params_.async_allocations; }

  // "cuda_tracing_level": Matches IREE flag --cuda_tracing:
  // Permissible values are:
  //   0 : stream tracing disabled.
  //   1 : coarse command buffer level tracing enabled.
  //   2 : fine-grained kernel level tracing enabled.
  int32_t &tracing_level() { return default_device_params_.stream_tracing; }

  // The number of logical HAL devices to create per physical, visible device.
  // This form of topology can be useful in certain cases where we aim to have
  // oversubscription emulating what would usually be achieved with process
  // level isolation. Defaults to 1.
  size_t &logical_devices_per_physical_device() {
    return logical_devices_per_physical_device_;
  }

  std::vector<std::string> GetAvailableDeviceIds();

 private:
  void InitializeDefaultSettings();
  void Enumerate();

  iree_hal_cuda_device_params_t default_device_params_;

  bool cpu_devices_enabled_ = false;
  bool cuda_allow_device_reuse_ = false;
  std::optional<std::vector<std::string>> visible_devices_;
  size_t logical_devices_per_physical_device_ = 1;
  std::vector<std::string> cuda_allocator_specs_;

  iree::hal_driver_ptr cuda_hal_driver_;
  iree_host_size_t available_devices_count_ = 0;
  iree::allocated_ptr<iree_hal_device_info_t> available_devices_;
};

}  // namespace shortfin::local::systems

#endif  // SHORTFIN_LOCAL_SYSTEMS_CUDA_H
