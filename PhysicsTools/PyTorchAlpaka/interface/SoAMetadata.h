#ifndef PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h
#define PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h

#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <any>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ATen/core/ScalarType.h>

#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/ROCmSerialSyncHandle.h"

namespace cms::torch::alpakatools {

  using namespace cms::soa;
  using namespace cms::alpakatools;

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  // Wrapper struct to merge info about scalar columns and multidimensional eigen columns
  struct Columns {
    std::vector<int> columns;

    // Empty constructor, to fill iteratively
    Columns() {}

    // Constructor for scalar columns
    Columns(int columns_) { columns.push_back(columns_); }

    // Constructor for multidimensional eigen columns
    Columns(const std::vector<int>& columns_) : columns(columns_) {}
    Columns(std::vector<int>&& columns_) : columns(std::move(columns_)) {}

    size_t size() const { return columns.size(); }
    int operator[](int i) const { return columns[i]; }
    void push(int i) { columns.push_back(i); }
  };

  // Block of SoA Columns with same type and element size.
  // Calculates size and stride and stores torch type.
  struct Block {
    // Constructor for columns and eigen columns
    Block(const int nElements,
          const size_t alignment,
          const void* ptr,
          const Columns& columns,
          const ::torch::ScalarType type,
          const size_t bytes)
        : ptr_(ptr), type_(type), bytes_(bytes), alignment_(alignment) {
      stride_ = create_stride(nElements, alignment, columns, bytes);
      size_ = create_size(nElements, columns);
    };

    // Constructor for scalar columns
    Block(const int nElements, const size_t alignment, const void* ptr, const ::torch::ScalarType type, const size_t bytes)
        : ptr_(ptr), type_(type), bytes_(bytes), alignment_(alignment) {
      stride_ = create_stride(nElements, alignment, 1, bytes, true);
      size_ = create_size(nElements, 1);
    };

    static int get_elems_per_column(const int nElements, const size_t alignment, const size_t bytes) {
      int per_bunch = alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);
      return bunches * per_bunch;
    }

    size_t alignment() const { return alignment_; }
    const void* ptr() const { return ptr_; }
    ::torch::ScalarType type() const { return type_; }
    size_t bytes() const { return bytes_; }
    const std::vector<long int>& size() const { return size_; }
    const std::vector<long int>& stride() const { return stride_; }


  private:
    static std::vector<long int> create_size(const int nElements, const Columns& columns) {
      std::vector<long int> size(columns.size() + 1);
      size[0] = nElements;
      std::copy(columns.columns.begin(), columns.columns.end(), size.begin() + 1);
      if (columns.size() > 1 && columns[0] == 1) {
        size.erase(size.begin()+1);
      }

      return size;
    }

    static std::vector<long int> create_stride(
        const int nElements, const size_t alignment, const Columns& columns, const size_t bytes, const bool is_scalar = false) {
      int N = columns.size() + 1;
      std::vector<long int> stride(N);

      int per_bunch = alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);

      if (!is_scalar)
        stride[0] = 1;
      else {
        // Jump no element per row, to fill with scalar value
        stride[0] = 0;
        bunches = 1;
      }
      stride[std::min(2, N - 1)] = bunches * per_bunch;

      // eigen are stored in column major, but still for every column.
      if (N > 2) {
        for (int i = 3; i < N; i++) {
          stride[i] = stride[i - 1] * columns[i - 2];
        }
        stride[1] = stride[N - 1] * columns[N - 2];
        if (columns[0] == 1) {
          stride.erase(stride.begin()+1);
        }
      }
      return stride;
    }

    std::vector<long int> stride_;
    std::vector<long int> size_;

    const void* ptr_;
    const ::torch::ScalarType type_;
    const size_t bytes_;
    const size_t alignment_;
  };

  // Metadata for SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  struct SoAMetadata {
  private:
    std::map<std::string, Block> blocks;
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    std::vector<std::shared_ptr<alpaka_rocm_async::torch::ROCmSerialSyncHandleBase>> rocm_serial_sync_handles_;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

    template <typename T>
    inline static ::torch::ScalarType get_type() {
      return ::torch::CppTypeToScalarType<T>();
    }

    inline static std::vector<int> standard_order(int size) {
      std::vector<int> order(size);
      for (int i = 0; i < size; i++) {
        order[i] = i;
      }
      return order;
    }

    template <typename T, typename... Others>
    bool check_location(int elements, const T* column, const T* other_column, Others... others) {
      return check_location(elements, other_column, others...) && (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, const T* column, const T* other_column) {
      return (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, const T* column) {
      return true;
    }

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nElements;
    int nBlocks;

    SoAMetadata(int nElements_) : nElements(nElements_), nBlocks(0) {}

    // Eigen columns
    template <typename SoALayout, typename T, typename... Others>
      requires(SameTypes<typename T::ValueType, typename Others::ValueType...> && T::columnType == SoAColumnType::eigen)
    void append_block(const std::string& name,
                      int nElements_,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      using ScalarType = typename T::ScalarType;
      auto [d_ptr, stride] = std::get<0>(column).tupleOrPointer();
      int elems = Block::get_elems_per_column(nElements, SoALayout::alignment, sizeof(ScalarType));
      assert(check_location(elems * T::ValueType::RowsAtCompileTime * T::ValueType::ColsAtCompileTime,
                            d_ptr,
                            std::get<0>(std::get<0>(others).tupleOrPointer())...));

      Columns col{{sizeof...(others) + 1, T::ValueType::RowsAtCompileTime}};
      if (T::ValueType::ColsAtCompileTime > 1) {
        col.push(T::ValueType::ColsAtCompileTime);
      }

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      auto rocm_serial_sync_handle = std::make_shared<alpaka_rocm_async::torch::ROCmSerialSyncHandle<ScalarType>>(
          d_ptr, 1 + sizeof...(Others), elems * T::ValueType::RowsAtCompileTime * T::ValueType::ColsAtCompileTime);
      rocm_serial_sync_handles_.push_back(std::move(rocm_serial_sync_handle));
      auto* ptr = rocm_serial_sync_handles_.back()->ptr();
#else
      auto* ptr = d_ptr;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

      blocks.try_emplace(name, nElements_, SoALayout::alignment, ptr, col, get_type<ScalarType>(), sizeof(ScalarType));
      order.push_back(name);
      nBlocks += 1;
    }

    // Append a block based on a typed pointer and a column object.
    template <typename SoALayout, typename T, typename... Others>
      requires(SameTypes<typename T::ScalarType, typename Others::ScalarType...> &&
               T::columnType == SoAColumnType::column)
    void append_block(const std::string& name,
                      int nElements_,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      using ScalarType = typename T::ScalarType;
      int elems = Block::get_elems_per_column(nElements_, SoALayout::alignment, sizeof(ScalarType));
      assert(check_location(elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...));

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      auto rocm_serial_sync_handle = std::make_shared<alpaka_rocm_async::torch::ROCmSerialSyncHandle<ScalarType>>(
          std::get<0>(column).tupleOrPointer(), 1 + sizeof...(Others), elems);
      rocm_serial_sync_handles_.push_back(std::move(rocm_serial_sync_handle));
      auto* ptr = rocm_serial_sync_handles_.back()->ptr();
#else
      auto* ptr = std::get<0>(column).tupleOrPointer();
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

      blocks.try_emplace(name,
                         nElements_,
                         SoALayout::alignment,
                         ptr,
                         sizeof...(others) + 1,
                         get_type<ScalarType>(),
                         sizeof(ScalarType));
      order.push_back(name);
      nBlocks += 1;
    }

    // Scalar columns are broadcasted
    template <typename SoALayout, SoAColumnType col_type, typename T>
      requires(std::is_arithmetic_v<T> && col_type == SoAColumnType::scalar)
    void append_block(const std::string& name,
                      int nElements_,
                      std::tuple<SoAParametersImpl<col_type, T>, cms::soa::size_type> column) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      auto rocm_serial_sync_handle = std::make_shared<alpaka_rocm_async::torch::ROCmSerialSyncHandle<T>>(
          std::get<0>(column).tupleOrPointer(), 1, 1);
      rocm_serial_sync_handles_.push_back(std::move(rocm_serial_sync_handle));
      auto* ptr = rocm_serial_sync_handles_.back()->ptr();
#else
      auto* ptr = std::get<0>(column).tupleOrPointer();
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

      blocks.try_emplace(name, nElements_, SoALayout::alignment, ptr, get_type<T>(), sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    // The order is defined by the order append_block is called.
    // It can be changed by passing a vector of the block names afterwards.
    // All blocks have to be mentioned.
    void change_order(const std::vector<std::string>& new_order) { order = new_order; }
    void change_order(std::vector<std::string>&& new_order) { order = std::move(new_order); }

    inline const Block& operator[](const std::string& key) const { return blocks.at(key); }

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    template <typename TQueue>
    void copyToHost(TQueue& queue) {
      for (int i = 0; i < nBlocks; i++)
        rocm_serial_sync_handles_[i]->copyToHost(queue);
    }

    template <typename TQueue>
    void copyToDevice(TQueue& queue) {
      for (int i = 0; i < nBlocks; i++)
        rocm_serial_sync_handles_[i]->copyToDevice(queue);
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
  };

  // Metadata to run model with input SOA and fill output SOA.
  class ModelMetadata {
  public:
    SoAMetadata input;
    SoAMetadata output;

    // Used in model class to correctly choose multi or single output conversion
    bool multi_head;

    ModelMetadata(const SoAMetadata& input_,
                  const SoAMetadata& output_,
                  bool multi_head_ = false)
        : input(input_), output(output_), multi_head(multi_head_) {}

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // For AMD CPU fallback both inputs and outputs are copied to host
    template <typename TQueue>
    void copyToHost(TQueue& queue) {
      input.copyToHost(queue);
      output.copyToHost(queue);
      // explicit synchronize to ensure data is in place before inference
      alpaka::wait(queue);
    }

    // For AMD CPU fallback only outputs are copied to device, no need to copy inputs back
    template <typename TQueue>
    void copyToDevice(TQueue& queue) {
      output.copyToDevice(queue);
      // no need to explicitly synchronize, rely on implicit synchronization mechanism in framework
      // alpaka::wait(queue);
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h
