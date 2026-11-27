cmake_minimum_required(VERSION 3.18)

project(RDMA_CPU_Proxy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RDMA_PROXY_REQUIRE_CUDA "Fail configuration if CUDA is not available" OFF)
option(RDMA_PROXY_REQUIRE_VERBS "Fail configuration if libibverbs is not available" OFF)
option(RDMA_PROXY_BUILD_TESTS "Build unit tests" ON)

include(CheckLanguage)
check_language(CUDA)
set(RDMA_PROXY_HAVE_CUDA 0)
if(CMAKE_CUDA_COMPILER)
  # Transport kernels have no architecture-specific MMA instructions. Ship
  # native Hopper and Blackwell code by default; explicit CMake/CUDAARCHS
  # selections remain authoritative (QuACK separately compiles its GEMMs).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "90;100" CACHE STRING "CUDA device architectures")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_LESS 12.9)
    message(FATAL_ERROR "RDMA_CPU_Proxy requires CUDA 12.9 or newer for cudaMemcpyBatchAsync and Blackwell support.")
  endif()
  message(STATUS "Native CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "Native CUDA toolkit: ${CUDAToolkit_VERSION} (${CUDAToolkit_NVCC_EXECUTABLE})")
  # All consumers embed/share CUDA handles and allocations with PyTorch. Avoid
  # implicitly adding cudart_static alongside the explicit CUDA::cudart link.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  set(RDMA_PROXY_HAVE_CUDA 1)
elseif(RDMA_PROXY_REQUIRE_CUDA)
  message(FATAL_ERROR "CUDA compiler not found. Install CUDA or configure with -DRDMA_PROXY_REQUIRE_CUDA=OFF.")
endif()

find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(IBVERBS_LIBRARY ibverbs)
set(RDMA_PROXY_HAVE_VERBS 0)
if(IBVERBS_INCLUDE_DIR AND IBVERBS_LIBRARY)
  set(RDMA_PROXY_HAVE_VERBS 1)
elseif(RDMA_PROXY_REQUIRE_VERBS)
  message(FATAL_ERROR "libibverbs not found. Install rdma-core/libibverbs or configure with -DRDMA_PROXY_REQUIRE_VERBS=OFF.")
endif()

set(RDMA_PROXY_SOURCES
  native/src/config.cpp
  native/src/cpu_affinity.cpp
  native/src/cuda_buffers.cpp
  native/src/down_wave_schedule.cpp
  native/src/return_map_worker.cpp
  native/src/logging.cpp
  native/src/protocol.cpp
  native/src/proxy.cpp
  native/src/qp_worker.cpp
  native/src/down_ready_trace.cpp
  native/src/return_copies.cpp
  native/src/rdma_return.cpp
  native/src/return_ready_publication.cpp
  native/src/rdma_connection.cpp
  native/src/rdma_context.cpp
  native/src/router_routing.cpp
  native/src/iteration_routing.cpp
  native/src/iteration_routing_transport.cpp
)

if(RDMA_PROXY_HAVE_CUDA)
  set(EXPERT_TOKEN_IDX_KERNEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/metadata/expert_token_idx_kernel.cu)
  set_source_files_properties(${EXPERT_TOKEN_IDX_KERNEL_SOURCE}
    PROPERTIES COMPILE_DEFINITIONS EXPERT_TOKEN_IDX_STANDALONE=1)
  list(APPEND RDMA_PROXY_SOURCES native/src/cuda_kernels.cu ${EXPERT_TOKEN_IDX_KERNEL_SOURCE})
  list(APPEND RDMA_PROXY_SOURCES native/src/iteration_metadata_exchange.cu)
endif()

add_library(rdma_cpu_proxy_lib STATIC ${RDMA_PROXY_SOURCES})
set_target_properties(rdma_cpu_proxy_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(rdma_cpu_proxy_lib PUBLIC include native/include)
if(RDMA_PROXY_HAVE_CUDA)
  target_include_directories(rdma_cpu_proxy_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/metadata)
endif()
target_compile_definitions(rdma_cpu_proxy_lib
  PUBLIC
    RDMA_PROXY_HAVE_CUDA=$<BOOL:${RDMA_PROXY_HAVE_CUDA}>
    RDMA_PROXY_HAVE_VERBS=$<BOOL:${RDMA_PROXY_HAVE_VERBS}>
)

if(RDMA_PROXY_HAVE_VERBS)
  target_include_directories(rdma_cpu_proxy_lib PRIVATE ${IBVERBS_INCLUDE_DIR})
  target_link_libraries(rdma_cpu_proxy_lib PRIVATE ${IBVERBS_LIBRARY})
endif()

if(RDMA_PROXY_HAVE_CUDA)
  set_target_properties(rdma_cpu_proxy_lib PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  # rdma_cpu_proxy_lib is static, so CUDA symbols referenced by its objects
  # must also be present on every consuming executable's final link line.
  target_link_libraries(rdma_cpu_proxy_lib PUBLIC CUDA::cudart CUDA::cuda_driver)
  # The header-only NVTX3 calls in proxy.cpp and cuda_buffers.cpp use the dynamic loader.
  target_link_libraries(rdma_cpu_proxy_lib PRIVATE ${CMAKE_DL_LIBS})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Older glibc provides the proxy's shm_open and shm_unlink in librt.
  target_link_libraries(rdma_cpu_proxy_lib PRIVATE rt)
endif()

find_package(Threads REQUIRED)
target_link_libraries(rdma_cpu_proxy_lib PUBLIC Threads::Threads)

add_executable(rdma_cpu_proxy native/src/main.cpp)
target_link_libraries(rdma_cpu_proxy PRIVATE rdma_cpu_proxy_lib)

# Stable C ABI for Python ctypes and other non-C++ embedding environments.
# Keep this separate from the executable so the legacy launch path remains
# available and both entry points share the exact same implementation library.
add_library(rdma_cpu_proxy_shared SHARED native/src/c_api.cpp)
set_target_properties(rdma_cpu_proxy_shared PROPERTIES
  OUTPUT_NAME rdma_cpu_proxy
  VERSION 1.0.0
  SOVERSION 1
)
target_include_directories(rdma_cpu_proxy_shared PUBLIC include)
target_link_libraries(rdma_cpu_proxy_shared PRIVATE rdma_cpu_proxy_lib)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND RDMA_PROXY_HAVE_CUDA AND RDMA_PROXY_HAVE_VERBS)
  add_library(router_metadata_exchange2 SHARED
    benchmarks/router_metadata_exchange2/native_proxy.cpp
    benchmarks/router_metadata_exchange2/exchange.cu)
  set_target_properties(router_metadata_exchange2 PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(router_metadata_exchange2 PRIVATE ${IBVERBS_INCLUDE_DIR})
  target_link_libraries(router_metadata_exchange2 PRIVATE rdma_cpu_proxy_lib ${IBVERBS_LIBRARY} rt)
  add_dependencies(rdma_cpu_proxy_shared router_metadata_exchange2)
endif()

if(RDMA_PROXY_BUILD_TESTS)
  enable_testing()

  add_executable(test_iteration_routing tests/native/test_iteration_routing.cpp)
  target_link_libraries(test_iteration_routing PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_iteration_routing COMMAND test_iteration_routing)
  set_tests_properties(test_iteration_routing PROPERTIES TIMEOUT 30)
  add_executable(test_benchmark_routing tests/native/test_benchmark_routing.cpp)
  target_link_libraries(test_benchmark_routing PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_benchmark_routing COMMAND test_benchmark_routing)
  set_tests_properties(test_benchmark_routing PROPERTIES TIMEOUT 90)
  add_executable(test_iteration_metadata_cuda tests/native/test_iteration_metadata_cuda.cpp)
  target_link_libraries(test_iteration_metadata_cuda PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_iteration_metadata_cuda COMMAND test_iteration_metadata_cuda)
  set_tests_properties(test_iteration_metadata_cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

  add_executable(test_dynamic_gather tests/native/test_dynamic_gather.cpp)
  target_link_libraries(test_dynamic_gather PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_dynamic_gather COMMAND test_dynamic_gather)
  add_test(NAME test_borrowed_routing_upload_cuda COMMAND test_dynamic_gather --cuda)
  set_tests_properties(test_borrowed_routing_upload_cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

  add_executable(test_forwarding_rounds tests/native/test_forwarding_rounds.cpp)
  target_include_directories(test_forwarding_rounds PRIVATE native/include)
  target_link_libraries(test_forwarding_rounds PRIVATE Threads::Threads)
  add_test(NAME test_forwarding_rounds COMMAND test_forwarding_rounds)

  add_executable(test_forward_round_scheduler tests/native/test_forward_round_scheduler.cpp)
  target_link_libraries(test_forward_round_scheduler PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_forward_round_scheduler COMMAND test_forward_round_scheduler)

  add_executable(test_forward_round_protocol tests/native/test_forward_round_protocol.cpp)
  target_link_libraries(test_forward_round_protocol PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_forward_round_protocol COMMAND test_forward_round_protocol)
  set_tests_properties(test_forward_round_protocol PROPERTIES TIMEOUT 20)

  add_executable(test_dual_rdma tests/native/test_dual_rdma.cpp)
  target_link_libraries(test_dual_rdma PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_dual_rdma COMMAND test_dual_rdma)
  set_tests_properties(test_dual_rdma PROPERTIES TIMEOUT 30)

  add_executable(test_iteration_control
    tests/native/test_iteration_control.cpp
    tests/native/test_proxy_iteration_barrier.cpp)
  target_link_libraries(test_iteration_control PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_iteration_control COMMAND test_iteration_control)
  set_tests_properties(test_iteration_control PROPERTIES TIMEOUT 30)

  add_executable(test_down_wave_schedule tests/native/test_down_wave_schedule.cpp)
  target_link_libraries(test_down_wave_schedule PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_down_wave_schedule COMMAND test_down_wave_schedule)

  add_executable(test_async_return_maps tests/native/test_async_return_maps.cpp)
  target_link_libraries(test_async_return_maps PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_async_return_maps COMMAND test_async_return_maps)
  add_test(NAME test_async_return_maps_cuda COMMAND test_async_return_maps --cuda)
  set_tests_properties(test_async_return_maps PROPERTIES TIMEOUT 30)
  set_tests_properties(test_async_return_maps_cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

  add_executable(test_cpu_scatter tests/native/test_cpu_scatter.cpp)
  target_link_libraries(test_cpu_scatter PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_cpu_scatter COMMAND test_cpu_scatter)
  add_test(NAME test_cpu_scatter_cuda COMMAND test_cpu_scatter --cuda)
  set_tests_properties(test_cpu_scatter PROPERTIES TIMEOUT 30)
  set_tests_properties(test_cpu_scatter_cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

  add_executable(test_chunking tests/native/test_chunking.cpp)
  target_link_libraries(test_chunking PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_chunking COMMAND test_chunking)

  add_executable(test_gemm_geometry tests/native/test_gemm_geometry.cpp)
  target_link_libraries(test_gemm_geometry PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_gemm_geometry COMMAND test_gemm_geometry)

  add_executable(test_pooled_gather tests/native/test_pooled_gather.cpp)
  target_link_libraries(test_pooled_gather PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_pooled_gather COMMAND test_pooled_gather)

  add_executable(test_config tests/native/test_config.cpp)
  target_link_libraries(test_config PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_config COMMAND test_config ${CMAKE_CURRENT_SOURCE_DIR}/config/example_config.json)

  add_executable(test_measured_run tests/native/test_measured_run.cpp)
  target_link_libraries(test_measured_run PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_measured_run COMMAND test_measured_run)

  add_executable(test_qp_calibration tests/native/test_qp_calibration.cpp)
  target_link_libraries(test_qp_calibration PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_qp_calibration COMMAND test_qp_calibration)
  add_test(NAME test_qp_calibration_transport COMMAND test_qp_calibration --transport-only)
  set_tests_properties(test_qp_calibration test_qp_calibration_transport PROPERTIES TIMEOUT 30)

  add_executable(test_route_weights tests/native/test_route_weights.cpp)
  target_link_libraries(test_route_weights PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_route_weights COMMAND test_route_weights)
  add_test(NAME test_route_weights_cuda COMMAND test_route_weights --cuda)
  set_tests_properties(test_route_weights_cuda PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

  add_executable(test_router_routing tests/native/test_router_routing.cpp)
  target_link_libraries(test_router_routing PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_router_routing COMMAND test_router_routing)

  add_executable(test_global_metadata_exchange tests/native/test_global_metadata_exchange.cpp)
  target_link_libraries(test_global_metadata_exchange PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_global_metadata_exchange COMMAND test_global_metadata_exchange)

  add_executable(test_c_api tests/native/test_c_api.cpp)
  add_executable(test_down_ready_trace tests/native/test_down_ready_trace.cpp)
  add_executable(test_rdma_returns tests/native/test_rdma_returns.cpp)
  target_link_libraries(test_rdma_returns PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_rdma_returns COMMAND test_rdma_returns)
  set_tests_properties(test_rdma_returns PROPERTIES TIMEOUT 20)
  add_executable(test_return_copies tests/native/test_return_copies.cpp)
  add_executable(test_return_notifications tests/native/test_return_notifications.cpp)
  add_executable(test_return_batch_sync tests/native/test_return_batch_sync.cpp)
  target_link_libraries(test_return_batch_sync PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_return_batch_sync COMMAND test_return_batch_sync)
  set_tests_properties(test_return_batch_sync PROPERTIES TIMEOUT 30)
  add_executable(test_return_ready_publication tests/native/test_return_ready_publication.cpp)
  target_link_libraries(test_return_ready_publication PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_return_ready_publication COMMAND test_return_ready_publication)
  add_test(NAME test_return_ready_publication_cuda COMMAND test_return_ready_publication --cuda)
  set_tests_properties(test_return_ready_publication test_return_ready_publication_cuda
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)
  target_link_libraries(test_return_notifications PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_return_notifications COMMAND test_return_notifications)
  set_tests_properties(test_return_notifications PROPERTIES TIMEOUT 30)
  target_link_libraries(test_return_copies PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_return_copies COMMAND test_return_copies)
  target_link_libraries(test_down_ready_trace PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_down_ready_trace COMMAND test_down_ready_trace)
  target_link_libraries(test_c_api PRIVATE rdma_cpu_proxy_shared Threads::Threads)
  add_test(NAME test_c_api COMMAND test_c_api ${CMAKE_CURRENT_SOURCE_DIR}/config/example_config.json)

  add_executable(test_forward_events tests/native/test_forward_events.cpp)
  target_link_libraries(test_forward_events PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_forward_events COMMAND test_forward_events)
  set_tests_properties(test_forward_events PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)

  add_executable(test_prepared_forwarding tests/native/test_prepared_forwarding.cpp)
  add_executable(test_dynamic_forwarding tests/native/test_dynamic_forwarding.cpp)
  target_link_libraries(test_dynamic_forwarding PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_dynamic_forwarding COMMAND test_dynamic_forwarding)
  set_tests_properties(test_dynamic_forwarding PROPERTIES TIMEOUT 60)
  target_link_libraries(test_prepared_forwarding PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_prepared_forwarding COMMAND test_prepared_forwarding)
  set_tests_properties(test_prepared_forwarding PROPERTIES TIMEOUT 120)
  add_test(NAME test_early_local_ready COMMAND test_prepared_forwarding --early-local-ready)
  add_test(NAME test_forward_round_integration COMMAND test_prepared_forwarding --complete-round)
  add_test(NAME test_forward_round_forwarding COMMAND test_prepared_forwarding --complete-round-forwarding)
  set_tests_properties(test_forward_round_forwarding PROPERTIES TIMEOUT 180)
  set_tests_properties(test_forward_round_integration PROPERTIES TIMEOUT 120)
  set_tests_properties(test_early_local_ready PROPERTIES TIMEOUT 120)
  add_test(NAME test_prepared_atomic_ring COMMAND test_prepared_forwarding --atomic-ring-only)
  add_test(NAME test_prepared_atomic_ring_depth4 COMMAND test_prepared_forwarding --atomic-ring-depth4)
  set_tests_properties(test_prepared_atomic_ring test_prepared_atomic_ring_depth4 PROPERTIES TIMEOUT 120)
  add_test(NAME test_stream_query COMMAND test_prepared_forwarding --stream-query)
  add_test(NAME test_stream_query_atomic COMMAND test_prepared_forwarding --stream-query-atomic)
  set_tests_properties(test_stream_query test_stream_query_atomic PROPERTIES TIMEOUT 120)
  add_test(NAME test_submit_epilogue COMMAND test_prepared_forwarding --submit-epilogue-only)
  set_tests_properties(test_submit_epilogue PROPERTIES TIMEOUT 120)
  add_test(NAME test_ping_pong2 COMMAND test_prepared_forwarding --ping-pong2-only)
  set_tests_properties(test_ping_pong2 PROPERTIES TIMEOUT 120)

  add_test(NAME test_unified_local COMMAND test_prepared_forwarding --unified-local)
  add_test(NAME test_dual_device_forwarding COMMAND test_prepared_forwarding --dual-device)
  add_test(NAME test_pooled_input_transport COMMAND test_prepared_forwarding --pooled-input)
  set_tests_properties(test_pooled_input_transport PROPERTIES TIMEOUT 30)
  set_tests_properties(test_dual_device_forwarding PROPERTIES TIMEOUT 120)
  add_test(NAME test_unified_local_atomic COMMAND test_prepared_forwarding --unified-local-atomic)
  add_test(NAME test_unified_local_query COMMAND test_prepared_forwarding --unified-local-query)
  set_tests_properties(test_unified_local test_unified_local_atomic test_unified_local_query
    PROPERTIES TIMEOUT 120)

  add_executable(test_ping_pong tests/native/test_ping_pong.cpp)
  target_link_libraries(test_ping_pong PRIVATE rdma_cpu_proxy_lib)
  add_test(NAME test_ping_pong COMMAND test_ping_pong)
  add_test(NAME test_forward_round_ping_pong COMMAND test_ping_pong --complete-round)
  set_tests_properties(test_forward_round_ping_pong PROPERTIES TIMEOUT 120)
  set_tests_properties(test_ping_pong PROPERTIES TIMEOUT 120)
endif()

message(STATUS "RDMA_PROXY_HAVE_CUDA=${RDMA_PROXY_HAVE_CUDA}")
message(STATUS "RDMA_PROXY_HAVE_VERBS=${RDMA_PROXY_HAVE_VERBS}")
