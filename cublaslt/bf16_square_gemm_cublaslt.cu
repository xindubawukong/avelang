#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +              \
                               cudaGetErrorString(status));                    \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status = (expr);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(std::string("cuBLASLt error at ") + __FILE__ +  \
                               ":" + std::to_string(__LINE__) + ": status " + \
                               std::to_string(static_cast<int>(status)));      \
    }                                                                          \
  } while (0)

namespace {

struct CaseResult {
  int size;
  int repeats;
  float milliseconds;
  double tflops;
  size_t workspace_bytes;
};

__global__ void fill_bf16_kernel(__nv_bfloat16 *ptr, size_t count,
                                 float scale) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < count; i += stride) {
    float value = static_cast<float>((i * 17 + 13) & 0xff) * scale;
    ptr[i] = __float2bfloat16(value);
  }
}

void fill_bf16(__nv_bfloat16 *ptr, size_t count, float scale) {
  int block = 256;
  int grid = static_cast<int>(
      std::min<size_t>((count + block - 1) / block, 65535));
  fill_bf16_kernel<<<grid, block>>>(ptr, count, scale);
  CHECK_CUDA(cudaGetLastError());
}

int repeats_for_size(int size) {
  if (size <= 1024) {
    return 200;
  }
  if (size <= 2048) {
    return 100;
  }
  if (size <= 4096) {
    return 50;
  }
  if (size <= 8192) {
    return 20;
  }
  return 5;
}

int select_device(int device_count) {
  int best_device = 0;
  int best_major = -1;
  int best_minor = -1;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    if (prop.major > best_major ||
        (prop.major == best_major && prop.minor > best_minor)) {
      best_device = device;
      best_major = prop.major;
      best_minor = prop.minor;
    }
    if (prop.major >= 9) {
      return device;
    }
  }
  return best_device;
}

CaseResult benchmark_square(cublasLtHandle_t lt_handle, int size,
                            void *workspace, size_t workspace_bytes) {
  const int64_t m = size;
  const int64_t n = size;
  const int64_t k = size;
  const size_t elements = static_cast<size_t>(size) * size;
  const size_t bytes = elements * sizeof(__nv_bfloat16);

  __nv_bfloat16 *d_a = nullptr;
  __nv_bfloat16 *d_b = nullptr;
  __nv_bfloat16 *d_c = nullptr;
  __nv_bfloat16 *d_d = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));
  CHECK_CUDA(cudaMalloc(&d_d, bytes));

  fill_bf16(d_a, elements, 1.0f / 255.0f);
  fill_bf16(d_b, elements, 1.0f / 511.0f);
  fill_bf16(d_c, elements, 0.0f);
  CHECK_CUDA(cudaMemset(d_d, 0, bytes));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  CHECK_CUBLAS(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t trans = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, m, k, m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, k, n, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, m, n, m));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes)));

  constexpr int kMaxAlgorithms = 32;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristic(kMaxAlgorithms);
  int returned_results = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      lt_handle, op_desc, a_desc, b_desc, c_desc, d_desc, preference,
      kMaxAlgorithms, heuristic.data(), &returned_results));
  if (returned_results == 0) {
    throw std::runtime_error("cuBLASLt returned no valid algorithm");
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  auto run = [&](const cublasLtMatmulAlgo_t *algo) {
    CHECK_CUBLAS(cublasLtMatmul(lt_handle, op_desc, &alpha, d_a, a_desc, d_b,
                                b_desc, &beta, d_c, c_desc, d_d, d_desc, algo,
                                workspace, workspace_bytes, nullptr));
  };

  for (int i = 0; i < 3; ++i) {
    run(&heuristic[0].algo);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  int repeats = repeats_for_size(size);
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    run(&heuristic[0].algo);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / static_cast<float>(repeats);
  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                 static_cast<double>(k);
  double tflops = flops / (static_cast<double>(avg_ms) * 1.0e9);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));

  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));

  return {size, repeats, avg_ms, tflops, heuristic[0].workspaceSize};
}

} // namespace

int main() {
  try {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      std::cerr << "No CUDA devices found.\n";
      return 1;
    }

    int device = select_device(device_count);
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "# cuBLASLt BF16 square GEMM benchmark\n";
    std::cout << "selected_cuda_device," << device << "\n";
    std::cout << "device," << prop.name << "\n";
    std::cout << "compute_capability," << prop.major << "." << prop.minor
              << "\n";
    std::cout << "cuda_runtime_version," << CUDART_VERSION << "\n";
    std::cout << "cublas_version," << CUBLAS_VER_MAJOR << "."
              << CUBLAS_VER_MINOR << "." << CUBLAS_VER_PATCH << "\n";

    cublasLtHandle_t lt_handle = nullptr;
    CHECK_CUBLAS(cublasLtCreate(&lt_handle));

    size_t workspace_bytes = 256ULL * 1024ULL * 1024ULL;
    void *workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

    std::cout << "M,N,K,repeats,avg_ms,tflops,algo_workspace_bytes\n";
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    for (int size : sizes) {
      CaseResult result =
          benchmark_square(lt_handle, size, workspace, workspace_bytes);
      std::cout << result.size << "," << result.size << "," << result.size
                << "," << result.repeats << "," << std::fixed
                << std::setprecision(4) << result.milliseconds << ","
                << std::setprecision(2) << result.tflops << ","
                << result.workspace_bytes << "\n";
    }

    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtDestroy(lt_handle));
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
