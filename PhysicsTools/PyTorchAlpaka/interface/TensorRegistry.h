#ifndef PhysicsTools_PyTorchAlpaka_interface_TensorRegistry_h
#define PhysicsTools_PyTorchAlpaka_interface_TensorRegistry_h

#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/core/ScalarType.h>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorView.h"

namespace cms::torch::alpakatools {

  using namespace cms::soa;
  using namespace cms::alpakatools;

  template <typename T>
  bool check_location(int elements, const T* column) {
    return true;
  }

  template <typename T>
  bool check_location(int elements, const T* column, const T* other_column) {
      return (column + elements) == other_column;
  }

  template <typename T, typename... Others>
  bool check_location(int elements, const T* column, const T* other_column, Others... others) {
      return (column + elements) == other_column && check_location(elements, other_column, others...);
  }

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  template <typename TSoAParamsImpl, typename... Others>
  concept SameValueType = SameTypes<typename TSoAParamsImpl::ValueType, typename Others::ValueType...>;

  template <typename TSoAParamsImpl, typename... Others>
  concept SameScalarType = SameTypes<typename TSoAParamsImpl::ScalarType, typename Others::ScalarType...>;

  template <typename T>
  ::torch::ScalarType get_type() {
    return ::torch::CppTypeToScalarType<T>();
  }

  class TensorRegistry {
  public:
    explicit TensorRegistry(int batch_size) : batch_size_(batch_size) {}

    // SOA_EIGEN_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameTypes<typename TSoAParamsImpl::ValueType, typename Others::ValueType...> && TSoAParamsImpl::columnType == SoAColumnType::eigen)
    void register_tensor(const std::string& name, int batch_size, std::tuple<TSoAParamsImpl, cms::soa::size_type> column, std::tuple<Others, cms::soa::size_type>... others) {
      using data_t = typename TSoAParamsImpl::ScalarType;
      auto [ptr, stride] = std::get<0>(column).tupleOrPointer();
      int n_elems = getElementsPerColumn(batch_size, SoALayout::alignment, sizeof(data_t));
      assert(check_location(
          n_elems * TSoAParamsImpl::ValueType::RowsAtCompileTime * TSoAParamsImpl::ValueType::ColsAtCompileTime, 
          ptr, std::get<0>(std::get<0>(others).tupleOrPointer())...));
        
      std::vector<int> tensor_dims;
      if constexpr (TSoAParamsImpl::ValueType::ColsAtCompileTime > 1)
        tensor_dims = {1 + sizeof...(Others), TSoAParamsImpl::ValueType::RowsAtCompileTime, TSoAParamsImpl::ValueType::ColsAtCompileTime};
      else 
        tensor_dims = {1 + sizeof...(Others), TSoAParamsImpl::ValueType::RowsAtCompileTime};
            
      emplace_tensor<data_t>(name, SoALayout::alignment, ptr, batch_size, tensor_dims);
    }

    // SOA_EIGEN_COLUMN with default batch size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameTypes<typename TSoAParamsImpl::ValueType, typename Others::ValueType...> && TSoAParamsImpl::columnType == SoAColumnType::eigen)
    void register_tensor(const std::string& name,
                      std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      register_tensor<SoALayout, TSoAParamsImpl, Others...>(name, batch_size_, column, others...);
    }

    // SOA_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameTypes<typename TSoAParamsImpl::ScalarType, typename Others::ScalarType...> && TSoAParamsImpl::columnType == SoAColumnType::column)
    void register_tensor(const std::string& name, int batch_size, std::tuple<TSoAParamsImpl, cms::soa::size_type> column, std::tuple<Others, cms::soa::size_type>... others) {
      using data_t = typename TSoAParamsImpl::ScalarType;
      int n_elems = getElementsPerColumn(batch_size, SoALayout::alignment, sizeof(data_t));
      assert(check_location(n_elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...));

      emplace_tensor<data_t>(name, SoALayout::alignment, std::get<0>(column).tupleOrPointer(), batch_size, std::vector<int>{1 + sizeof...(Others)});
    }

    // SOA_COLUMN with default batch size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameTypes<typename TSoAParamsImpl::ScalarType, typename Others::ScalarType...> && TSoAParamsImpl::columnType == SoAColumnType::column)
    void register_tensor(const std::string& name,
                      std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      register_tensor<SoALayout, TSoAParamsImpl, Others...>(name, batch_size_, column, others...);
    }

    // SOA_SCALAR
    template <typename SoALayout, SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == SoAColumnType::scalar)
    void register_tensor(const std::string& name, int batch_size, std::tuple<SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      const auto* ptr = std::get<0>(column).tupleOrPointer();
      emplace_tensor<T>(name, SoALayout::alignment, ptr, batch_size);
    }

    // SOA_SCALAR with default batch size
    template <typename SoALayout, SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == SoAColumnType::scalar)
    void register_tensor(const std::string& name, std::tuple<SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      register_tensor<SoALayout, column_t, T>(name, batch_size_, column);
    }

    size_t size() const { return registry_.size(); }
    const TensorView<>& operator[](const size_t index) const { return registry_.at(order_[index]); }

    // class iterator {
    // public:
    //   using iterator_category = std::forward_iterator_tag;
    //   using value_type = TensorView;
    //   using difference_type = std::ptrdiff_t;
    //   using pointer = const TensorView*;
    //   using reference = const TensorView&;

    //   iterator(std::vector<std::string>::const_iterator it,
    //           const std::map<std::string, TensorView>* registry)
    //       : it_(it), registry_(registry) {}

    //   reference operator*() const { return registry_->at(*it_); }
    //   pointer operator->() const { return &registry_->at(*it_); }

    //   iterator& operator++() { ++it_; return *this; }
    //   iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }

    //   bool operator==(const iterator& other) const = default;

    // private:
    //   std::vector<std::string>::const_iterator it_;
    //   const std::map<std::string, TensorView>* registry_;
    // };

    // iterator begin() const { return iterator(order_.cbegin(), &registry_); }
    // iterator end() const { return iterator(order_.cend(), &registry_); }

  private:
    template <typename T>
    void emplace_tensor(const std::string& name, size_t alignment, const void* ptr, int batch_size, std::vector<int> dims) {
      registry_.try_emplace(name, alignment, sizeof(T), ptr, get_type<T>(), batch_size, std::move(dims));
      order_.push_back(name);
    }

    int batch_size_;
    std::vector<std::string> order_;
    std::map<std::string, TensorView<>> registry_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorRegistry_h