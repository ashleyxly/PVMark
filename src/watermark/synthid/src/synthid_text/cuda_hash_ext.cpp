#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> mimc_batched_wet_update_cuda(
    torch::Tensor contexts,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor prime,
    torch::Tensor half_prime,
    torch::Tensor r2,
    torch::Tensor one,
    torch::Tensor round_keys,
    torch::Tensor keys_mont,
    torch::Tensor key_index_mont,
    int64_t num_steps,
    int64_t batch_size,
    int64_t context_history_size);

std::vector<torch::Tensor> mimc_batched_wet_debug_cuda(
    torch::Tensor contexts,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor prime,
    torch::Tensor half_prime,
    torch::Tensor r2,
    torch::Tensor one,
    torch::Tensor round_keys,
    torch::Tensor keys_mont,
    torch::Tensor key_index_mont,
    int64_t num_steps,
    int64_t batch_size,
    int64_t context_history_size);

torch::Tensor batched_score_update_cuda(
    torch::Tensor context_hashes,
    torch::Tensor g_values,
    torch::Tensor scores,
    int64_t num_steps,
    int64_t batch_size,
    int64_t context_history_size);

torch::Tensor mimc_online_wet_update_cuda(
    torch::Tensor contexts,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor context_history,
    torch::Tensor prime,
    torch::Tensor half_prime,
    torch::Tensor r2,
    torch::Tensor one,
    torch::Tensor round_keys,
    torch::Tensor keys_mont,
    torch::Tensor key_index_mont,
    int64_t write_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mimc_batched_wet_update",
      &mimc_batched_wet_update_cuda,
      "MiMC BN254 batched SynthID WET replay update (CUDA)");
  m.def(
      "mimc_batched_wet_debug",
      &mimc_batched_wet_debug_cuda,
      "MiMC BN254 batched SynthID WET replay debug tensors (CUDA)");
  m.def(
      "batched_score_update",
      &batched_score_update_cuda,
      "Batched SynthID repetition check and score update (CUDA)");
  m.def(
      "mimc_online_wet_update",
      &mimc_online_wet_update_cuda,
      "MiMC BN254 online SynthID WET hash/history/score update (CUDA)");
}
