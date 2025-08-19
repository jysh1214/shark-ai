#include "shortfin/local/systems/cuda.h"

#include <fmt/xchar.h>

#include "shortfin/support/logging.h"
#include "shortfin/support/sysconfig.h"

namespace shortfin::local::systems {

namespace {
const std::string_view SYSTEM_DEVICE_CLASS = "cuda";
const std::string_view LOGICAL_DEVICE_CLASS = "gpu";
const std::string_view HAL_DRIVER_PREFIX = "cuda";
}  // namespace

CUDASystemBuilder::CUDASystemBuilder(iree_allocator_t host_allocator,
                                     ConfigOptions options)
    : HostCPUSystemBuilder(host_allocator, std::move(options)),
      available_devices_(host_allocator) {
  iree_hal_cuda_device_params_initialize(&default_device_params_);
  InitializeDefaultSettings();
  config_options().ValidateUndef();
}

CUDASystemBuilder::~CUDASystemBuilder() = default;

void CUDASystemBuilder::InitializeDefaultSettings() {
  // Gets allocator specs from either "cuda_allocators" or the fallback
  // "allocators".
  cuda_allocator_specs_ = GetConfigAllocatorSpecs("cuda_allocators");

  default_device_params_.async_allocations =
      config_options().GetBool("cuda_async_allocations", true);

  cuda_allow_device_reuse_ =
      config_options().GetBool("cuda_allow_device_reuse", false);

  // CUDA options.
  // "amdgpu_tracing_level": Matches IREE flag --cuda_tracing:
  // Permissible values are:
  //   0 : stream tracing disabled.
  //   1 : coarse command buffer level tracing enabled.
  //   2 : fine-grained kernel level tracing enabled.
  auto tracing_level =
      config_options().GetInt("cuda_tracing_level", /*non_negative=*/true);
  default_device_params_.stream_tracing = tracing_level ? *tracing_level : 2;

  auto logical_devices_per_physical_device = config_options().GetInt(
      "cuda_logical_devices_per_physical_device", /*non_negative=*/true);
  if (logical_devices_per_physical_device) {
    logical_devices_per_physical_device_ = *logical_devices_per_physical_device;
  }

  cpu_devices_enabled_ = config_options().GetBool("cuda_cpu_devices_enabled");

  std::optional<std::string_view> visible_devices_option =
      config_options().GetOption("cuda_visible_devices");
  if (visible_devices_option) {
    auto splits = config_options().Split(*visible_devices_option, ';');
    visible_devices_.emplace();
    for (auto split_sv : splits) {
      visible_devices_->emplace_back(split_sv);
    }
  }
}

void CUDASystemBuilder::Enumerate() {
  if (cuda_hal_driver_) return;
  SHORTFIN_TRACE_SCOPE_NAMED("CUDASystemBuilder::Enumerate");

  iree_hal_cuda_driver_options_t driver_options;
  iree_hal_cuda_driver_options_initialize(&driver_options);

  SHORTFIN_THROW_IF_ERROR(iree_hal_cuda_driver_create(
      IREE_SV("cuda"), &driver_options, &default_device_params_,
      host_allocator(), cuda_hal_driver_.for_output()));

  // Get available devices and filter into visible_devices_.
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_query_available_devices(
      cuda_hal_driver_, host_allocator(), &available_devices_count_,
      available_devices_.for_output()));
  for (iree_host_size_t i = 0; i < available_devices_count_; ++i) {
    iree_hal_device_info_t *info = &available_devices_.get()[i];
    logging::info(
        "Enumerated available CUDA device {}: path='{}' name='{}' device_id={}",
        i, to_string_view(info->path), to_string_view(info->name),
        info->device_id);
  }
}

std::vector<std::string> CUDASystemBuilder::GetAvailableDeviceIds() {
  Enumerate();
  std::vector<std::string> results;
  for (iree_host_size_t i = 0; i < available_devices_count_; ++i) {
    iree_hal_device_info_t *info = &available_devices_.get()[i];
    results.emplace_back(to_string_view(info->path));
  }
  return results;
}

SystemPtr CUDASystemBuilder::CreateSystem() {
  SHORTFIN_TRACE_SCOPE_NAMED("CUDASystemBuilder::CreateSystem");
  auto lsys = std::make_shared<System>(host_allocator());
  Enumerate();

  // TODO: Real NUMA awareness.
  lsys->InitializeNodes(1);
  lsys->InitializeHalDriver(SYSTEM_DEVICE_CLASS, cuda_hal_driver_);

  // Must have some device visible.
  if (available_devices_count_ == 0 &&
      (!visible_devices_ || visible_devices_->empty())) {
    throw std::invalid_argument("No CUDA devices found/visible");
  }

  std::vector<iree_hal_device_id_t> used_device_ids;
  if (visible_devices_) {
    logging::info("Processing visible_devices list with {} entries",
                  visible_devices_->size());
    for (size_t i = 0; i < visible_devices_->size(); ++i) {
      logging::info("  visible_devices[{}] = '{}'", i, (*visible_devices_)[i]);
    }
    used_device_ids.reserve(visible_devices_->size());
    std::unordered_map<std::string_view,
                       std::vector<std::optional<iree_hal_device_id_t>>>
        visible_device_hal_ids;
    for (size_t i = 0; i < available_devices_count_; ++i) {
      iree_hal_device_info_t *info = &available_devices_.get()[i];
      visible_device_hal_ids[to_string_view(info->path)].push_back(
          info->device_id);
    }

    for (auto &visible_device_id : *visible_devices_) {
      auto found_it = visible_device_hal_ids.find(visible_device_id);
      if (found_it == visible_device_hal_ids.end()) {
        throw std::invalid_argument(fmt::format(
            "Requested visible device '{}' was not found on the system "
            "(available: '{}')",
            visible_device_id, fmt::join(GetAvailableDeviceIds(), ";")));
      }

      bool found = false;
      auto &bucket = found_it->second;
      for (auto &hal_id : bucket) {
        if (hal_id) {
          found = true;
          logging::info("Mapping visible device '{}' to HAL device_id {}",
                        visible_device_id, *hal_id);
          used_device_ids.push_back(*hal_id);
          if (!cuda_allow_device_reuse_) {
            hal_id.reset();
          }
        }
      }

      if (!found) {
        throw std::invalid_argument(
            fmt::format("Requested visible device '{}' was requested more "
                        "times than present on the system ({})",
                        visible_device_id, bucket.size()));
      }
    }
  } else {
    for (iree_host_size_t i = 0; i < available_devices_count_; ++i) {
      iree_hal_device_info_t *info = &available_devices_.get()[i];
      used_device_ids.push_back(info->device_id);
    }
  }

  size_t expected_device_count =
      used_device_ids.size() * logical_devices_per_physical_device_;
  if (!sysconfig::EnsureFileLimit(expected_device_count * 64 + 768)) {
    logging::error(
        "Could not ensure sufficient file handles for minimum operations: "
        "Suggest setting explicit limits with `ulimit -n` and system settings");
  }

  // Initialize all used GPU devices.
  for (size_t instance_ordinal = 0; instance_ordinal < used_device_ids.size();
       ++instance_ordinal) {
    iree_hal_device_id_t device_id = used_device_ids[instance_ordinal];
    for (size_t logical_index = 0;
         logical_index < logical_devices_per_physical_device_;
         ++logical_index) {
      iree::hal_device_ptr device;
      SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
          cuda_hal_driver_, device_id, 0, nullptr, host_allocator(),
          device.for_output()));
      DeviceAddress address(
          /*system_device_class=*/SYSTEM_DEVICE_CLASS,
          /*logical_device_class=*/LOGICAL_DEVICE_CLASS,
          /*hal_driver_prefix=*/HAL_DRIVER_PREFIX,
          /*instance_ordinal=*/instance_ordinal,
          /*queue_ordinal=*/0,
          /*instance_topology_address=*/{logical_index});
      logging::info(
          "Creating CUDA device with instance_ordinal={}, logical_index={}, "
          "device_name='{}', HAL device_id={}",
          instance_ordinal, logical_index, address.device_name, device_id);
      ConfigureAllocators(cuda_allocator_specs_, device, address.device_name);
      lsys->InitializeHalDevice(std::make_unique<CUDADevice>(
          address,
          /*hal_device=*/device,
          /*node_affinity=*/0,
          /*capabilities=*/static_cast<uint32_t>(Device::Capabilities::NONE)));
    }
  }

  if (cpu_devices_enabled_) {
    InitializeHostCPUDefaults();
    auto *driver = InitializeHostCPUDriver(*lsys);
    InitializeHostCPUDevices(*lsys, driver);
  }

  lsys->FinishInitialization();
  return lsys;
}

}  // namespace shortfin::local::systems
