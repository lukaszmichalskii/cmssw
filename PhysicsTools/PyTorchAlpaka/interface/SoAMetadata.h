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
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

namespace cms::torch::alpakatools {

  using namespace cms::soa;

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  // Wrapper struct to merge info about scalar columns and multidimensional eigen columns
  struct Columns {
    std::vector<int> columns;

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
  template <typename SOA_Layout>
  struct Block {
    std::vector<long int> stride;
    std::vector<long int> size;

    void* ptr;
    ::torch::ScalarType type;
    size_t bytes;
    bool is_scalar = false;

    Block() : ptr(nullptr) {}
    // Constructor for columns and eigen columns
    Block(int nElements, void* ptr_, const Columns& columns_, ::torch::ScalarType type_, size_t bytes_)
        : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = create_stride(nElements, columns_, bytes_);
      size = create_size(nElements, columns_);
    };

    // Constructor for scalar columns
    Block(int nElements, void* ptr_, ::torch::ScalarType type_, size_t bytes_) : ptr(ptr_), type(type_), bytes(bytes_) {
      stride = create_stride(nElements, 1, bytes_, true);
      size = create_size(nElements, 1);
    };

    static int get_elems_per_column(int nElements, size_t bytes) {
      int per_bunch = SOA_Layout::alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);
      return bunches * per_bunch;
    }

  private:
    static std::vector<long int> create_size(int nElements, const Columns& columns) {
      std::vector<long int> size(columns.size() + 1);
      size[0] = nElements;
      std::copy(columns.columns.begin(), columns.columns.end(), size.begin() + 1);

      return size;
    }

    static std::vector<long int> create_stride(int nElements,
                                               const Columns& columns,
                                               size_t bytes,
                                               bool is_scalar = false) {
      int N = columns.size() + 1;
      std::vector<long int> stride(N);

      int per_bunch = SOA_Layout::alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);

      if (!is_scalar)
        stride[0] = 1;
      else {
        // Jump no element per row, to fill with scalar value
        stride[0] = 0;
        bunches = 1 * nElements;
      }
      stride[std::min(2, N - 1)] = bunches * per_bunch;
      // eigen are stored in column major, but still for every column.
      if (N > 2) {
        for (int i = 3; i < N; i++) {
          stride[i] = stride[i - 1] * columns[i - 2];
        }
        stride[1] = stride[N - 1] * columns[N - 2];
      }
      return stride;
    }
  };

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  struct IAutoCopyableMemBuf {
    virtual ~IAutoCopyableMemBuf() = default;
    virtual void h2d(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) = 0;
  };

  template <typename T>
  struct AutoCopyableMemBuf : IAutoCopyableMemBuf {
    using HostView = cms::alpakatools::host_view<T[]>;
    using DeviceView = cms::alpakatools::device_view<ALPAKA_ACCELERATOR_NAMESPACE::Device, T[]>;

    HostView host_view_;
    DeviceView device_view_;

    AutoCopyableMemBuf(HostView host_view, DeviceView device_view)
       : host_view_(host_view), device_view_(device_view) {}

    ~AutoCopyableMemBuf() override {}

    void h2d(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) override {
      alpaka::memcpy(queue, device_view_, host_view_);
    }
  };
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  // Metadata for SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  template <typename SOA_Layout>
  struct SoAMetadata {
  private:
    std::map<std::string, Block<SOA_Layout>> blocks;

    #ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    std::vector<std::shared_ptr<IAutoCopyableMemBuf>> lifetime_policy_buffers_;  // debug
    std::vector<std::any> host_buffers_;  // RAII hack
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
    bool check_location(int elements, T* column, T* other_column, Others... others) {
      return check_location(elements, other_column, others...) && (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, T* column, T* other_column) {
      return (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, T* column) {
      return true;
    }

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nElements;
    int nBlocks;

    SoAMetadata(int nElements_) : nElements(nElements_), nBlocks(0) {}

    #ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    void h2d(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
      for (auto& autocpybuf : lifetime_policy_buffers_) {
        autocpybuf->h2d(queue);
      }
    }
    #endif  // ALPAKA_ACC_GPU_HIP_ENABLED

    template <typename T, typename... Others>
      requires(SameTypes<typename T::ValueType, typename Others::ValueType...> && T::columnType == SoAColumnType::eigen)
    void append_block(const std::string& name,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      const auto [ptr, stride] = std::get<0>(column).tupleOrPointer();

      int elems = Block<SOA_Layout>::get_elems_per_column(nElements, sizeof(typename T::ScalarType));
      assert(check_location(elems * T::ValueType::RowsAtCompileTime * T::ValueType::ColsAtCompileTime,
                            ptr,
                            std::get<0>(std::get<0>(others).tupleOrPointer())...));

      Columns col({sizeof...(others) + 1, T::ValueType::RowsAtCompileTime});
      if (T::ValueType::ColsAtCompileTime > 1)
        col.push(T::ValueType::ColsAtCompileTime);

      blocks.try_emplace(name, nElements, ptr, col, get_type<typename T::ScalarType>(), sizeof(typename T::ScalarType));
      order.push_back(name);
      nBlocks += 1;
    }

    // Append a block based on a typed pointer and a column object.
    // Can be normal column or eigen column.
    template <typename T, typename... Others>
      requires(SameTypes<typename T::ScalarType, typename Others::ScalarType...> &&
               T::columnType == SoAColumnType::column)
    void append_block(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue,
                      const std::string& name,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      #ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      using ScalarType = typename T::ScalarType;
      int nCols = 1 + sizeof...(Others);
      int elemsPerCol = Block<SOA_Layout>::get_elems_per_column(nElements, sizeof(ScalarType));
      size_t totalElems = elemsPerCol * nCols;

      // allocate one contiguous host buffer for all columns
      auto hostBuf = cms::alpakatools::make_host_buffer<ScalarType[]>(totalElems);
      alpaka::memset(queue, hostBuf, 0x00);
      ScalarType* hostPtr = alpaka::getPtrNative(hostBuf);

      // keep buffer alive by storing in std::any
      host_buffers_.push_back(hostBuf);

      // helper to copy each column into contiguous buffer
      auto copy_to_offset = [&](auto &col_tuple, int offset) {
        auto* dev_ptr = std::get<0>(col_tuple).tupleOrPointer();
        auto dev = alpaka::getDev(queue);
        auto dev_view = cms::alpakatools::make_device_view(dev, dev_ptr, nElements);

        std::span<ScalarType> target_span(hostPtr + offset * elemsPerCol, elemsPerCol);
        auto host_view = cms::alpakatools::make_host_view(target_span);
        alpaka::memcpy(queue, host_view, dev_view);
        lifetime_policy_buffers_.emplace_back(std::make_shared<AutoCopyableMemBuf<ScalarType>>(host_view, dev_view));
      };

      // copy columns to host buffer
      copy_to_offset(column, 0);
      int idx = 1;
      (copy_to_offset(others, idx++), ...);

      // store metadata block
      blocks.try_emplace(name,
                        nElements,
                        hostPtr,
                        sizeof...(others) + 1,
                        get_type<typename T::ScalarType>(),
                        sizeof(ScalarType));
      order.push_back(name);
      nBlocks += 1;
      #else
      int elems = Block<SOA_Layout>::get_elems_per_column(nElements, sizeof(typename T::ScalarType));
      assert(check_location(elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...));

      blocks.try_emplace(name,
                         nElements,
                         std::get<0>(column).tupleOrPointer(),
                         sizeof...(others) + 1,
                         get_type<typename T::ScalarType>(),
                         sizeof(typename T::ScalarType));
      order.push_back(name);
      nBlocks += 1;
      #endif  // ALPAKA_ACC_GPU_HIP_ENABLED

      // for (int col = 0; col < nCols; ++col) {
      //   int padding = elemsPerCol - nElements;

      //   std::cout << "Column " << col 
      //             << " (nElements = " << nElements
      //             << ", elemsPerCol = " << elemsPerCol
      //             << ", padding = " << padding << "):" << std::endl;

      //   // print actual data
      //   std::cout << "  data: ";
      //   for (int i = 0; i < nElements; ++i) {
      //     std::cout << hostPtr[col * elemsPerCol + i] << ", ";
      //   }
      //   std::cout << std::endl;

      //   // print padding/trailing zeros
      //   if (padding > 0) {
      //     std::cout << "  padding/trailing zeros: ";
      //     for (int i = nElements; i < elemsPerCol; ++i) {
      //         std::cout << hostPtr[col * elemsPerCol + i] << ", ";
      //     }
      //     std::cout << std::endl;
      //   }
      // }
    }

    template <SoAColumnType col_type, typename T>
      requires(std::is_arithmetic_v<T> && col_type == SoAColumnType::scalar)
    void append_block(const std::string& name, std::tuple<SoAParametersImpl<col_type, T>, cms::soa::size_type> column) {
      blocks.try_emplace(name, nElements, std::get<0>(column).tupleOrPointer(), get_type<T>(), sizeof(T));
      order.push_back(name);
      nBlocks += 1;
    }

    // The order is defined by the order append_block is called.
    // It can be changed by passing a vector of the block names afterwards.
    // All blocks have to be mentioned.
    void change_order(const std::vector<std::string>& new_order) { order = new_order; }
    void change_order(std::vector<std::string>&& new_order) { order = std::move(new_order); }

    inline Block<SOA_Layout> operator[](const std::string& key) const { return blocks.at(key); }
  };

  // Metadata to run model with input SOA and fill output SOA.
  template <typename SOA_Input, typename SOA_Output>
  class ModelMetadata {
  public:
    SoAMetadata<SOA_Input> input;
    SoAMetadata<SOA_Output> output;

    // Used in AOT model class to correctly choose multi or single output conversion
    // Default value true, as single value can be parsed with multi output
    bool multi_output;

    ModelMetadata(const SoAMetadata<SOA_Input>& input_,
                  const SoAMetadata<SOA_Output>& output_,
                  bool multi_output_ = true)
        : input(input_), output(output_), multi_output(multi_output_) {}

    void h2d(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) { output.h2d(queue); alpaka::wait(queue); }
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h