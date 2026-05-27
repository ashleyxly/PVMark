#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int LIMBS = 8;
constexpr int MIMC_ROUNDS = 91;
constexpr int WORD_BITS = 32;
constexpr uint32_t WORD_MASK = 0xffffffffu;
constexpr uint32_t MONT_INV32 = 4026531839u;
constexpr int MAX_CANDIDATES = 256;
constexpr int MAX_KEYS = 64;

__device__ __forceinline__ void copy8(uint32_t* dst, const uint32_t* src) {
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    dst[i] = src[i];
  }
}

__device__ __forceinline__ void copy8_from_row(
    uint32_t* dst,
    const uint32_t* src,
    int64_t row) {
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    dst[i] = src[row * LIMBS + i];
  }
}

__device__ __forceinline__ void copy8_to_row(
    uint32_t* dst,
    int64_t row,
    const uint32_t* src) {
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    dst[row * LIMBS + i] = src[i];
  }
}

__device__ __forceinline__ void set_u64_limbs(uint32_t* out, uint64_t value) {
  out[0] = static_cast<uint32_t>(value & WORD_MASK);
  out[1] = static_cast<uint32_t>((value >> WORD_BITS) & WORD_MASK);
#pragma unroll
  for (int i = 2; i < LIMBS; ++i) {
    out[i] = 0u;
  }
}

__device__ __forceinline__ bool ge8(
    const uint32_t* left,
    const uint32_t* right) {
#pragma unroll
  for (int offset = 0; offset < LIMBS; ++offset) {
    const int index = LIMBS - 1 - offset;
    if (left[index] > right[index]) {
      return true;
    }
    if (left[index] < right[index]) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ void sub_prime_if_needed(
    uint32_t* out,
    const uint32_t* prime) {
  uint64_t borrow = 0;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    const uint64_t subtrahend = static_cast<uint64_t>(prime[i]) + borrow;
    const uint64_t current = static_cast<uint64_t>(out[i]);
    out[i] = static_cast<uint32_t>((current - subtrahend) & WORD_MASK);
    borrow = current < subtrahend ? 1 : 0;
  }
}

__device__ __forceinline__ void add_mod(
    uint32_t* out,
    const uint32_t* left,
    const uint32_t* right,
    const uint32_t* prime) {
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    const uint64_t total =
        static_cast<uint64_t>(left[i]) + static_cast<uint64_t>(right[i]) + carry;
    out[i] = static_cast<uint32_t>(total & WORD_MASK);
    carry = total >> WORD_BITS;
  }
  if (carry != 0 || ge8(out, prime)) {
    sub_prime_if_needed(out, prime);
  }
}

__device__ __forceinline__ void montgomery_mul(
    uint32_t* out,
    const uint32_t* left,
    const uint32_t* right,
    const uint32_t* prime) {
  uint32_t tmp[LIMBS + 1];
#pragma unroll
  for (int i = 0; i < LIMBS + 1; ++i) {
    tmp[i] = 0u;
  }

  for (int i = 0; i < LIMBS; ++i) {
    uint64_t carry = 0;
    const uint64_t right_limb = static_cast<uint64_t>(right[i]);
#pragma unroll
    for (int j = 0; j < LIMBS; ++j) {
      const uint64_t uv =
          static_cast<uint64_t>(tmp[j]) +
          static_cast<uint64_t>(left[j]) * right_limb +
          carry;
      tmp[j] = static_cast<uint32_t>(uv & WORD_MASK);
      carry = uv >> WORD_BITS;
    }
    tmp[LIMBS] = static_cast<uint32_t>(carry);

    const uint32_t factor = static_cast<uint32_t>(
        (static_cast<uint64_t>(tmp[0]) * static_cast<uint64_t>(MONT_INV32)) &
        WORD_MASK);
    carry = 0;
#pragma unroll
    for (int j = 0; j < LIMBS; ++j) {
      const uint64_t uv =
          static_cast<uint64_t>(tmp[j]) +
          static_cast<uint64_t>(factor) * static_cast<uint64_t>(prime[j]) +
          carry;
      const uint32_t low = static_cast<uint32_t>(uv & WORD_MASK);
      carry = uv >> WORD_BITS;
      if (j > 0) {
        tmp[j - 1] = low;
      }
    }
    const uint64_t uv = static_cast<uint64_t>(tmp[LIMBS]) + carry;
    tmp[LIMBS - 1] = static_cast<uint32_t>(uv & WORD_MASK);
    tmp[LIMBS] = static_cast<uint32_t>(uv >> WORD_BITS);
  }

#pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    out[i] = tmp[i];
  }
  if (ge8(out, prime)) {
    sub_prime_if_needed(out, prime);
  }
}

__device__ __forceinline__ void to_montgomery_u64(
    uint32_t* out,
    uint64_t value,
    const uint32_t* r2,
    const uint32_t* prime) {
  uint32_t canonical[LIMBS];
  set_u64_limbs(canonical, value);
  montgomery_mul(out, canonical, r2, prime);
}

__device__ __forceinline__ void from_montgomery(
    uint32_t* out,
    const uint32_t* value,
    const uint32_t* one,
    const uint32_t* prime) {
  montgomery_mul(out, value, one, prime);
}

__device__ __forceinline__ void pow7(
    uint32_t* out,
    const uint32_t* value,
    const uint32_t* prime) {
  uint32_t square[LIMBS];
  uint32_t fourth[LIMBS];
  uint32_t sixth[LIMBS];
  montgomery_mul(square, value, value, prime);
  montgomery_mul(fourth, square, square, prime);
  montgomery_mul(sixth, fourth, square, prime);
  montgomery_mul(out, sixth, value, prime);
}

__device__ __forceinline__ void mimc_non_feistel(
    uint32_t* out,
    const uint32_t* x,
    const uint32_t* k,
    const uint32_t* round_keys,
    const uint32_t* prime) {
  uint32_t r[LIMBS] = {0u};
  uint32_t tmp[LIMBS];
  uint32_t tmp2[LIMBS];
  uint32_t round_key[LIMBS];

  add_mod(tmp, k, x, prime);
  pow7(r, tmp, prime);

  for (int round_index = 1; round_index < MIMC_ROUNDS; ++round_index) {
    copy8_from_row(round_key, round_keys, round_index);
    add_mod(tmp, k, r, prime);
    add_mod(tmp2, tmp, round_key, prime);
    pow7(r, tmp2, prime);
  }

  add_mod(out, r, k, prime);
}

__device__ __forceinline__ void mimc_hash_pair(
    uint32_t* out,
    const uint32_t* input1,
    const uint32_t* input2,
    const uint32_t* round_keys,
    const uint32_t* prime) {
  uint32_t r[LIMBS] = {0u};
  uint32_t enc[LIMBS];
  uint32_t tmp[LIMBS];

  mimc_non_feistel(enc, input1, r, round_keys, prime);
  add_mod(tmp, input1, enc, prime);
  add_mod(r, r, tmp, prime);

  mimc_non_feistel(enc, input2, r, round_keys, prime);
  add_mod(tmp, input2, enc, prime);
  add_mod(out, r, tmp, prime);
}

__global__ void context_kernel(
    const int64_t* __restrict__ contexts,
    int32_t* __restrict__ context_out,
    int64_t num_rows,
    int sliding_window_size,
    const uint32_t* __restrict__ prime,
    const uint32_t* __restrict__ r2,
    const uint32_t* __restrict__ one,
    const uint32_t* __restrict__ round_keys) {
  const int64_t row = blockIdx.x;
  if (row >= num_rows || threadIdx.x != 0) {
    return;
  }

  uint32_t result[LIMBS];
  uint32_t token_mont[LIMBS];
  uint32_t next_hash[LIMBS];
  to_montgomery_u64(result, 1u, r2, prime);
  for (int offset = 0; offset < sliding_window_size; ++offset) {
    const uint64_t token = static_cast<uint64_t>(
        contexts[row * sliding_window_size + offset]);
    to_montgomery_u64(token_mont, token, r2, prime);
    mimc_hash_pair(next_hash, result, token_mont, round_keys, prime);
    copy8(result, next_hash);
  }
  from_montgomery(next_hash, result, one, prime);
  copy8_to_row(reinterpret_cast<uint32_t*>(context_out), row, next_hash);
}

__global__ void fused_candidate_score_kernel(
    const int32_t* __restrict__ context_hashes_i32,
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    float* __restrict__ output,
    int64_t num_rows,
    int64_t num_steps,
    int64_t batch_size,
    int context_history_size,
    int candidate_size,
    int num_keys,
    const uint32_t* __restrict__ prime,
    const uint32_t* __restrict__ half_prime,
    const uint32_t* __restrict__ r2,
    const uint32_t* __restrict__ one,
    const uint32_t* __restrict__ round_keys,
    const uint32_t* __restrict__ keys_mont,
    const uint32_t* __restrict__ key_index_mont) {
  const int64_t row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int tid = threadIdx.x;

  __shared__ uint8_t g_values[MAX_CANDIDATES * MAX_KEYS];
  __shared__ float probs[MAX_CANDIDATES];
  __shared__ uint8_t repeated_shared;
  __shared__ float g_mass_shared;

  const uint32_t* context_hashes =
      reinterpret_cast<const uint32_t*>(context_hashes_i32);

  if (tid == 0) {
    const int64_t step = row / batch_size;
    const int64_t batch = row - step * batch_size;
    bool repeated = false;
    bool current_zero = true;
#pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
      if (context_hashes[row * LIMBS + limb] != 0u) {
        current_zero = false;
        break;
      }
    }
    if (current_zero && step < context_history_size) {
      repeated = true;
    }
    if (!repeated) {
      int64_t first_step = step - context_history_size;
      if (first_step < 0) {
        first_step = 0;
      }
      for (int64_t previous_step = first_step; previous_step < step; ++previous_step) {
        const int64_t previous_row = previous_step * batch_size + batch;
        bool matches = true;
#pragma unroll
        for (int limb = 0; limb < LIMBS; ++limb) {
          if (context_hashes[previous_row * LIMBS + limb] !=
              context_hashes[row * LIMBS + limb]) {
            matches = false;
            break;
          }
        }
        if (matches) {
          repeated = true;
          break;
        }
      }
    }
    repeated_shared = repeated ? 1u : 0u;
  }

  __syncthreads();

  if (repeated_shared != 0u) {
    for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
      output[row * candidate_size + candidate] =
          scores[row * candidate_size + candidate];
    }
    return;
  }

  const int total = candidate_size * num_keys;
  for (int flat = tid; flat < total; flat += blockDim.x) {
    const int candidate_index = flat / num_keys;
    const int key_index = flat - candidate_index * num_keys;
    uint32_t context_mont[LIMBS];
    uint32_t token_mont[LIMBS];
    uint32_t candidate_hash[LIMBS];
    uint32_t key_hash[LIMBS];
    uint32_t g_hash[LIMBS];
    uint32_t g_canonical[LIMBS];
    uint32_t key_value[LIMBS];
    uint32_t key_index_value[LIMBS];
    uint32_t enc[LIMBS];
    uint32_t tmp[LIMBS];
    uint32_t zero_key[LIMBS] = {0u};
    uint32_t candidate_key_prefix[LIMBS];

    copy8_from_row(context_mont, context_hashes, row);
    montgomery_mul(context_mont, context_mont, r2, prime);
    const uint64_t token = static_cast<uint64_t>(
        indices[row * candidate_size + candidate_index]);
    to_montgomery_u64(token_mont, token, r2, prime);
    mimc_hash_pair(candidate_hash, context_mont, token_mont, round_keys, prime);

    mimc_non_feistel(enc, candidate_hash, zero_key, round_keys, prime);
    add_mod(tmp, candidate_hash, enc, prime);
    add_mod(candidate_key_prefix, zero_key, tmp, prime);
    copy8_from_row(key_value, keys_mont, key_index);
    copy8_from_row(key_index_value, key_index_mont, key_index);
    mimc_non_feistel(enc, key_value, candidate_key_prefix, round_keys, prime);
    add_mod(tmp, key_value, enc, prime);
    add_mod(key_hash, candidate_key_prefix, tmp, prime);
    mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime);
    from_montgomery(g_canonical, g_hash, one, prime);
    g_values[flat] =
        (ge8(g_canonical, half_prime) && !ge8(half_prime, g_canonical)) ? 1u : 0u;
  }

  __syncthreads();

  if (tid == 0) {
    float max_score = scores[row * candidate_size];
    for (int candidate = 1; candidate < candidate_size; ++candidate) {
      const float score = scores[row * candidate_size + candidate];
      if (score > max_score) {
        max_score = score;
      }
    }
    float normalizer = 0.0f;
    for (int candidate = 0; candidate < candidate_size; ++candidate) {
      const float value = expf(scores[row * candidate_size + candidate] - max_score);
      probs[candidate] = value;
      normalizer += value;
    }
    for (int candidate = 0; candidate < candidate_size; ++candidate) {
      probs[candidate] /= normalizer;
    }
  }

  __syncthreads();

  for (int key_index = 0; key_index < num_keys; ++key_index) {
    if (tid == 0) {
      float g_mass = 0.0f;
      for (int candidate = 0; candidate < candidate_size; ++candidate) {
        if (g_values[candidate * num_keys + key_index] != 0u) {
          g_mass += probs[candidate];
        }
      }
      g_mass_shared = g_mass;
    }

    __syncthreads();

    for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
      const float g = g_values[candidate * num_keys + key_index] != 0u ? 1.0f : 0.0f;
      probs[candidate] *= 1.0f + g - g_mass_shared;
    }

    __syncthreads();
  }

  for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
    const float prob = probs[candidate];
    output[row * candidate_size + candidate] =
        prob > 0.0f ? logf(prob) : -1.0e12f;
  }
}

__global__ void candidate_g_values_kernel(
    const int32_t* __restrict__ context_hashes_i32,
    const int64_t* __restrict__ indices,
    uint8_t* __restrict__ g_values_out,
    int64_t num_rows,
    int candidate_size,
    int num_keys,
    const uint32_t* __restrict__ prime,
    const uint32_t* __restrict__ half_prime,
    const uint32_t* __restrict__ r2,
    const uint32_t* __restrict__ one,
    const uint32_t* __restrict__ round_keys,
    const uint32_t* __restrict__ keys_mont,
    const uint32_t* __restrict__ key_index_mont) {
  const int64_t row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int tid = threadIdx.x;
  const uint32_t* context_hashes =
      reinterpret_cast<const uint32_t*>(context_hashes_i32);

  const int total = candidate_size * num_keys;
  for (int flat = tid; flat < total; flat += blockDim.x) {
    const int candidate_index = flat / num_keys;
    const int key_index = flat - candidate_index * num_keys;
    uint32_t context_mont[LIMBS];
    uint32_t token_mont[LIMBS];
    uint32_t candidate_hash[LIMBS];
    uint32_t key_hash[LIMBS];
    uint32_t g_hash[LIMBS];
    uint32_t g_canonical[LIMBS];
    uint32_t key_value[LIMBS];
    uint32_t key_index_value[LIMBS];
    uint32_t enc[LIMBS];
    uint32_t tmp[LIMBS];
    uint32_t zero_key[LIMBS] = {0u};
    uint32_t candidate_key_prefix[LIMBS];

    copy8_from_row(context_mont, context_hashes, row);
    montgomery_mul(context_mont, context_mont, r2, prime);
    const uint64_t token = static_cast<uint64_t>(
        indices[row * candidate_size + candidate_index]);
    to_montgomery_u64(token_mont, token, r2, prime);
    mimc_hash_pair(candidate_hash, context_mont, token_mont, round_keys, prime);
    mimc_non_feistel(enc, candidate_hash, zero_key, round_keys, prime);
    add_mod(tmp, candidate_hash, enc, prime);
    add_mod(candidate_key_prefix, zero_key, tmp, prime);
    copy8_from_row(key_value, keys_mont, key_index);
    copy8_from_row(key_index_value, key_index_mont, key_index);
    mimc_non_feistel(enc, key_value, candidate_key_prefix, round_keys, prime);
    add_mod(tmp, key_value, enc, prime);
    add_mod(key_hash, candidate_key_prefix, tmp, prime);
    mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime);
    from_montgomery(g_canonical, g_hash, one, prime);
    g_values_out[row * candidate_size * num_keys + flat] =
        (ge8(g_canonical, half_prime) && !ge8(half_prime, g_canonical)) ? 1u : 0u;
  }
}

__global__ void batched_repetition_kernel(
    const int32_t* __restrict__ context_hashes_i32,
    uint8_t* __restrict__ repeated_flags,
    int64_t num_rows,
    int64_t num_steps,
    int64_t batch_size,
    int context_history_size) {
  const int64_t row = blockIdx.x;
  if (row >= num_rows || threadIdx.x != 0) {
    return;
  }
  const uint32_t* context_hashes =
      reinterpret_cast<const uint32_t*>(context_hashes_i32);
  const int64_t step = row / batch_size;
  const int64_t batch = row - step * batch_size;
  bool repeated = false;
  bool current_zero = true;
#pragma unroll
  for (int limb = 0; limb < LIMBS; ++limb) {
    if (context_hashes[row * LIMBS + limb] != 0u) {
      current_zero = false;
      break;
    }
  }
  if (current_zero && step < context_history_size) {
    repeated = true;
  }
  if (!repeated) {
    int64_t first_step = step - context_history_size;
    if (first_step < 0) {
      first_step = 0;
    }
    for (int64_t previous_step = first_step; previous_step < step; ++previous_step) {
      const int64_t previous_row = previous_step * batch_size + batch;
      bool matches = true;
#pragma unroll
      for (int limb = 0; limb < LIMBS; ++limb) {
        if (context_hashes[previous_row * LIMBS + limb] !=
            context_hashes[row * LIMBS + limb]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        repeated = true;
        break;
      }
    }
  }
  repeated_flags[row] = repeated ? 1u : 0u;
}

__global__ void score_update_from_g_kernel(
    const int32_t* __restrict__ context_hashes_i32,
    const uint8_t* __restrict__ g_values,
    const float* __restrict__ scores,
    float* __restrict__ output,
    int64_t num_rows,
    int64_t num_steps,
    int64_t batch_size,
    int context_history_size,
    int candidate_size,
    int num_keys) {
  const int64_t row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int tid = threadIdx.x;
  const uint32_t* context_hashes =
      reinterpret_cast<const uint32_t*>(context_hashes_i32);

  __shared__ float probs[MAX_CANDIDATES];
  __shared__ uint8_t repeated_shared;
  __shared__ float g_mass_shared;

  if (tid == 0) {
    const int64_t step = row / batch_size;
    const int64_t batch = row - step * batch_size;
    bool repeated = false;
    bool current_zero = true;
#pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
      if (context_hashes[row * LIMBS + limb] != 0u) {
        current_zero = false;
        break;
      }
    }
    if (current_zero && step < context_history_size) {
      repeated = true;
    }
    if (!repeated) {
      int64_t first_step = step - context_history_size;
      if (first_step < 0) {
        first_step = 0;
      }
      for (int64_t previous_step = first_step; previous_step < step; ++previous_step) {
        const int64_t previous_row = previous_step * batch_size + batch;
        bool matches = true;
#pragma unroll
        for (int limb = 0; limb < LIMBS; ++limb) {
          if (context_hashes[previous_row * LIMBS + limb] !=
              context_hashes[row * LIMBS + limb]) {
            matches = false;
            break;
          }
        }
        if (matches) {
          repeated = true;
          break;
        }
      }
    }
    repeated_shared = repeated ? 1u : 0u;
  }

  __syncthreads();

  if (repeated_shared != 0u) {
    for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
      output[row * candidate_size + candidate] =
          scores[row * candidate_size + candidate];
    }
    return;
  }

  if (tid == 0) {
    float max_score = scores[row * candidate_size];
    for (int candidate = 1; candidate < candidate_size; ++candidate) {
      const float score = scores[row * candidate_size + candidate];
      if (score > max_score) {
        max_score = score;
      }
    }
    float normalizer = 0.0f;
    for (int candidate = 0; candidate < candidate_size; ++candidate) {
      const float value = expf(scores[row * candidate_size + candidate] - max_score);
      probs[candidate] = value;
      normalizer += value;
    }
    for (int candidate = 0; candidate < candidate_size; ++candidate) {
      probs[candidate] /= normalizer;
    }
  }

  __syncthreads();

  const int64_t row_g_offset = row * candidate_size * num_keys;
  for (int key_index = 0; key_index < num_keys; ++key_index) {
    if (tid == 0) {
      float g_mass = 0.0f;
      for (int candidate = 0; candidate < candidate_size; ++candidate) {
        if (g_values[row_g_offset + candidate * num_keys + key_index] != 0u) {
          g_mass += probs[candidate];
        }
      }
      g_mass_shared = g_mass;
    }

    __syncthreads();

    for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
      const float g =
          g_values[row_g_offset + candidate * num_keys + key_index] != 0u ? 1.0f : 0.0f;
      probs[candidate] *= 1.0f + g - g_mass_shared;
    }

    __syncthreads();
  }

  for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
    const float prob = probs[candidate];
    output[row * candidate_size + candidate] =
        prob > 0.0f ? logf(prob) : -1.0e12f;
  }
}

__global__ void online_fused_update_kernel(
    const int64_t* __restrict__ contexts,
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    float* __restrict__ output,
    int32_t* __restrict__ context_history_i32,
    int64_t batch_size,
    int sliding_window_size,
    int context_history_size,
    int candidate_size,
    int num_keys,
    int write_index,
    const uint32_t* __restrict__ prime,
    const uint32_t* __restrict__ half_prime,
    const uint32_t* __restrict__ r2,
    const uint32_t* __restrict__ one,
    const uint32_t* __restrict__ round_keys,
    const uint32_t* __restrict__ keys_mont,
    const uint32_t* __restrict__ key_index_mont) {
  const int64_t batch = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  const int tid = threadIdx.x;

  __shared__ uint32_t context_hash[LIMBS];
  __shared__ uint8_t g_values[MAX_CANDIDATES * MAX_KEYS];
  __shared__ float probs[MAX_CANDIDATES];
  __shared__ uint8_t repeated_shared;
  __shared__ float g_mass_shared;

  uint32_t* context_history =
      reinterpret_cast<uint32_t*>(context_history_i32);

  if (tid == 0) {
    uint32_t result[LIMBS];
    uint32_t token_mont[LIMBS];
    uint32_t next_hash[LIMBS];

    to_montgomery_u64(result, 1u, r2, prime);
    for (int offset = 0; offset < sliding_window_size; ++offset) {
      const uint64_t token = static_cast<uint64_t>(
          contexts[batch * sliding_window_size + offset]);
      to_montgomery_u64(token_mont, token, r2, prime);
      mimc_hash_pair(next_hash, result, token_mont, round_keys, prime);
      copy8(result, next_hash);
    }
    from_montgomery(next_hash, result, one, prime);
    copy8(context_hash, next_hash);

    bool repeated = false;
    for (int history_index = 0; history_index < context_history_size; ++history_index) {
      bool matches = true;
#pragma unroll
      for (int limb = 0; limb < LIMBS; ++limb) {
        const int64_t history_offset =
            (batch * context_history_size + history_index) * LIMBS + limb;
        if (context_history[history_offset] != context_hash[limb]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        repeated = true;
        break;
      }
    }
    repeated_shared = repeated ? 1u : 0u;

#pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
      const int64_t write_offset =
          (batch * context_history_size + write_index) * LIMBS + limb;
      context_history[write_offset] = context_hash[limb];
    }
  }

  __syncthreads();

  if (repeated_shared != 0u) {
    for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
      output[batch * candidate_size + candidate] =
          scores[batch * candidate_size + candidate];
    }
    return;
  }

  const int total = candidate_size * num_keys;
  for (int flat = tid; flat < total; flat += blockDim.x) {
    const int candidate_index = flat / num_keys;
    const int key_index = flat - candidate_index * num_keys;
    uint32_t context_mont[LIMBS];
    uint32_t token_mont[LIMBS];
    uint32_t candidate_hash[LIMBS];
    uint32_t key_hash[LIMBS];
    uint32_t g_hash[LIMBS];
    uint32_t g_canonical[LIMBS];
    uint32_t key_value[LIMBS];
    uint32_t key_index_value[LIMBS];
    uint32_t enc[LIMBS];
    uint32_t tmp[LIMBS];
    uint32_t zero_key[LIMBS] = {0u};
    uint32_t candidate_key_prefix[LIMBS];

    copy8(context_mont, context_hash);
    montgomery_mul(context_mont, context_mont, r2, prime);
    const uint64_t token = static_cast<uint64_t>(
        indices[batch * candidate_size + candidate_index]);
    to_montgomery_u64(token_mont, token, r2, prime);
    mimc_hash_pair(candidate_hash, context_mont, token_mont, round_keys, prime);

    mimc_non_feistel(enc, candidate_hash, zero_key, round_keys, prime);
    add_mod(tmp, candidate_hash, enc, prime);
    add_mod(candidate_key_prefix, zero_key, tmp, prime);
    copy8_from_row(key_value, keys_mont, key_index);
    copy8_from_row(key_index_value, key_index_mont, key_index);
    mimc_non_feistel(enc, key_value, candidate_key_prefix, round_keys, prime);
    add_mod(tmp, key_value, enc, prime);
    add_mod(key_hash, candidate_key_prefix, tmp, prime);
    mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime);
    from_montgomery(g_canonical, g_hash, one, prime);
    g_values[flat] =
        (ge8(g_canonical, half_prime) && !ge8(half_prime, g_canonical)) ? 1u : 0u;
  }

  __syncthreads();

  if (tid == 0) {
    float max_score = scores[batch * candidate_size];
    for (int candidate = 1; candidate < candidate_size; ++candidate) {
      const float score = scores[batch * candidate_size + candidate];
      if (score > max_score) {
        max_score = score;
      }
    }
    float normalizer = 0.0f;
    for (int candidate = 0; candidate < candidate_size; ++candidate) {
      const float value = expf(scores[batch * candidate_size + candidate] - max_score);
      probs[candidate] = value;
      normalizer += value;
    }
    for (int candidate = 0; candidate < candidate_size; ++candidate) {
      probs[candidate] /= normalizer;
    }
  }

  __syncthreads();

  for (int key_index = 0; key_index < num_keys; ++key_index) {
    if (tid == 0) {
      float g_mass = 0.0f;
      for (int candidate = 0; candidate < candidate_size; ++candidate) {
        if (g_values[candidate * num_keys + key_index] != 0u) {
          g_mass += probs[candidate];
        }
      }
      g_mass_shared = g_mass;
    }

    __syncthreads();

    for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
      const float g = g_values[candidate * num_keys + key_index] != 0u ? 1.0f : 0.0f;
      probs[candidate] *= 1.0f + g - g_mass_shared;
    }

    __syncthreads();
  }

  for (int candidate = tid; candidate < candidate_size; candidate += blockDim.x) {
    const float prob = probs[candidate];
    output[batch * candidate_size + candidate] =
        prob > 0.0f ? logf(prob) : -1.0e12f;
  }
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    torch::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

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
    int64_t context_history_size) {
  check_cuda_tensor(contexts, "contexts", torch::kInt64);
  check_cuda_tensor(indices, "indices", torch::kInt64);
  check_cuda_tensor(scores, "scores", torch::kFloat32);
  check_cuda_tensor(prime, "prime", torch::kUInt32);
  check_cuda_tensor(half_prime, "half_prime", torch::kUInt32);
  check_cuda_tensor(r2, "r2", torch::kUInt32);
  check_cuda_tensor(one, "one", torch::kUInt32);
  check_cuda_tensor(round_keys, "round_keys", torch::kUInt32);
  check_cuda_tensor(keys_mont, "keys_mont", torch::kUInt32);
  check_cuda_tensor(key_index_mont, "key_index_mont", torch::kUInt32);

  TORCH_CHECK(contexts.dim() == 2, "contexts must have shape [rows, context]");
  TORCH_CHECK(indices.dim() == 2, "indices must have shape [rows, candidates]");
  TORCH_CHECK(scores.dim() == 2, "scores must have shape [rows, candidates]");
  const int64_t num_rows = contexts.size(0);
  const int sliding_window_size = static_cast<int>(contexts.size(1));
  const int candidate_size = static_cast<int>(indices.size(1));
  const int num_keys = static_cast<int>(keys_mont.size(0));
  TORCH_CHECK(num_rows == num_steps * batch_size, "row count mismatch");
  TORCH_CHECK(scores.size(0) == num_rows, "scores row count mismatch");
  TORCH_CHECK(scores.size(1) == candidate_size, "scores candidate mismatch");
  TORCH_CHECK(candidate_size <= MAX_CANDIDATES, "too many candidates");
  TORCH_CHECK(num_keys <= MAX_KEYS, "too many keys");
  TORCH_CHECK(round_keys.size(0) == MIMC_ROUNDS, "unexpected MiMC round count");
  TORCH_CHECK(round_keys.size(1) == LIMBS, "round keys must be [rounds, limbs]");
  TORCH_CHECK(keys_mont.size(1) == LIMBS, "keys_mont must be [keys, limbs]");
  TORCH_CHECK(key_index_mont.size(1) == LIMBS, "key_index_mont must be [keys, limbs]");

  const c10::cuda::CUDAGuard device_guard(contexts.device());
  auto context_hashes = torch::empty(
      {num_rows, LIMBS},
      torch::TensorOptions().device(contexts.device()).dtype(torch::kInt32));
  auto output = torch::empty_like(scores);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  context_kernel<<<num_rows, 1, 0, stream>>>(
      contexts.data_ptr<int64_t>(),
      context_hashes.data_ptr<int32_t>(),
      num_rows,
      sliding_window_size,
      reinterpret_cast<const uint32_t*>(prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(r2.data_ptr()),
      reinterpret_cast<const uint32_t*>(one.data_ptr()),
      reinterpret_cast<const uint32_t*>(round_keys.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fused_candidate_score_kernel<<<num_rows, 32, 0, stream>>>(
      context_hashes.data_ptr<int32_t>(),
      indices.data_ptr<int64_t>(),
      scores.data_ptr<float>(),
      output.data_ptr<float>(),
      num_rows,
      num_steps,
      batch_size,
      static_cast<int>(context_history_size),
      candidate_size,
      num_keys,
      reinterpret_cast<const uint32_t*>(prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(half_prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(r2.data_ptr()),
      reinterpret_cast<const uint32_t*>(one.data_ptr()),
      reinterpret_cast<const uint32_t*>(round_keys.data_ptr()),
      reinterpret_cast<const uint32_t*>(keys_mont.data_ptr()),
      reinterpret_cast<const uint32_t*>(key_index_mont.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {output, context_hashes};
}

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
    int64_t context_history_size) {
  check_cuda_tensor(contexts, "contexts", torch::kInt64);
  check_cuda_tensor(indices, "indices", torch::kInt64);
  check_cuda_tensor(scores, "scores", torch::kFloat32);
  check_cuda_tensor(prime, "prime", torch::kUInt32);
  check_cuda_tensor(half_prime, "half_prime", torch::kUInt32);
  check_cuda_tensor(r2, "r2", torch::kUInt32);
  check_cuda_tensor(one, "one", torch::kUInt32);
  check_cuda_tensor(round_keys, "round_keys", torch::kUInt32);
  check_cuda_tensor(keys_mont, "keys_mont", torch::kUInt32);
  check_cuda_tensor(key_index_mont, "key_index_mont", torch::kUInt32);

  const int64_t num_rows = contexts.size(0);
  const int sliding_window_size = static_cast<int>(contexts.size(1));
  const int candidate_size = static_cast<int>(indices.size(1));
  const int num_keys = static_cast<int>(keys_mont.size(0));
  const c10::cuda::CUDAGuard device_guard(contexts.device());
  auto context_hashes = torch::empty(
      {num_rows, LIMBS},
      torch::TensorOptions().device(contexts.device()).dtype(torch::kInt32));
  auto g_values = torch::empty(
      {num_rows, candidate_size, num_keys},
      torch::TensorOptions().device(contexts.device()).dtype(torch::kUInt8));
  auto repeated_flags = torch::empty(
      {num_rows},
      torch::TensorOptions().device(contexts.device()).dtype(torch::kUInt8));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  context_kernel<<<num_rows, 1, 0, stream>>>(
      contexts.data_ptr<int64_t>(),
      context_hashes.data_ptr<int32_t>(),
      num_rows,
      sliding_window_size,
      reinterpret_cast<const uint32_t*>(prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(r2.data_ptr()),
      reinterpret_cast<const uint32_t*>(one.data_ptr()),
      reinterpret_cast<const uint32_t*>(round_keys.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  candidate_g_values_kernel<<<num_rows, 32, 0, stream>>>(
      context_hashes.data_ptr<int32_t>(),
      indices.data_ptr<int64_t>(),
      g_values.data_ptr<uint8_t>(),
      num_rows,
      candidate_size,
      num_keys,
      reinterpret_cast<const uint32_t*>(prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(half_prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(r2.data_ptr()),
      reinterpret_cast<const uint32_t*>(one.data_ptr()),
      reinterpret_cast<const uint32_t*>(round_keys.data_ptr()),
      reinterpret_cast<const uint32_t*>(keys_mont.data_ptr()),
      reinterpret_cast<const uint32_t*>(key_index_mont.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  batched_repetition_kernel<<<num_rows, 1, 0, stream>>>(
      context_hashes.data_ptr<int32_t>(),
      repeated_flags.data_ptr<uint8_t>(),
      num_rows,
      num_steps,
      batch_size,
      static_cast<int>(context_history_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {context_hashes, g_values, repeated_flags};
}

torch::Tensor batched_score_update_cuda(
    torch::Tensor context_hashes,
    torch::Tensor g_values,
    torch::Tensor scores,
    int64_t num_steps,
    int64_t batch_size,
    int64_t context_history_size) {
  check_cuda_tensor(context_hashes, "context_hashes", torch::kInt32);
  check_cuda_tensor(g_values, "g_values", torch::kUInt8);
  check_cuda_tensor(scores, "scores", torch::kFloat32);
  TORCH_CHECK(context_hashes.dim() == 2, "context_hashes must be [rows, limbs]");
  TORCH_CHECK(g_values.dim() == 3, "g_values must be [rows, candidates, keys]");
  TORCH_CHECK(scores.dim() == 2, "scores must be [rows, candidates]");
  const int64_t num_rows = context_hashes.size(0);
  const int candidate_size = static_cast<int>(scores.size(1));
  const int num_keys = static_cast<int>(g_values.size(2));
  TORCH_CHECK(num_rows == num_steps * batch_size, "row count mismatch");
  TORCH_CHECK(context_hashes.size(1) == LIMBS, "context hashes must have 8 limbs");
  TORCH_CHECK(g_values.size(0) == num_rows, "g_values row mismatch");
  TORCH_CHECK(g_values.size(1) == candidate_size, "g_values candidate mismatch");
  TORCH_CHECK(scores.size(0) == num_rows, "scores row mismatch");
  TORCH_CHECK(candidate_size <= MAX_CANDIDATES, "too many candidates");
  TORCH_CHECK(num_keys <= MAX_KEYS, "too many keys");

  const c10::cuda::CUDAGuard device_guard(context_hashes.device());
  auto output = torch::empty_like(scores);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  score_update_from_g_kernel<<<num_rows, 32, 0, stream>>>(
      context_hashes.data_ptr<int32_t>(),
      g_values.data_ptr<uint8_t>(),
      scores.data_ptr<float>(),
      output.data_ptr<float>(),
      num_rows,
      num_steps,
      batch_size,
      static_cast<int>(context_history_size),
      candidate_size,
      num_keys);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

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
    int64_t write_index) {
  check_cuda_tensor(contexts, "contexts", torch::kInt64);
  check_cuda_tensor(indices, "indices", torch::kInt64);
  check_cuda_tensor(scores, "scores", torch::kFloat32);
  check_cuda_tensor(context_history, "context_history", torch::kInt32);
  check_cuda_tensor(prime, "prime", torch::kUInt32);
  check_cuda_tensor(half_prime, "half_prime", torch::kUInt32);
  check_cuda_tensor(r2, "r2", torch::kUInt32);
  check_cuda_tensor(one, "one", torch::kUInt32);
  check_cuda_tensor(round_keys, "round_keys", torch::kUInt32);
  check_cuda_tensor(keys_mont, "keys_mont", torch::kUInt32);
  check_cuda_tensor(key_index_mont, "key_index_mont", torch::kUInt32);

  TORCH_CHECK(contexts.dim() == 2, "contexts must have shape [batch, context]");
  TORCH_CHECK(indices.dim() == 2, "indices must have shape [batch, candidates]");
  TORCH_CHECK(scores.dim() == 2, "scores must have shape [batch, candidates]");
  TORCH_CHECK(
      context_history.dim() == 3,
      "context_history must have shape [batch, history, limbs]");
  const int64_t batch_size = contexts.size(0);
  const int sliding_window_size = static_cast<int>(contexts.size(1));
  const int candidate_size = static_cast<int>(indices.size(1));
  const int context_history_size = static_cast<int>(context_history.size(1));
  const int num_keys = static_cast<int>(keys_mont.size(0));
  TORCH_CHECK(indices.size(0) == batch_size, "indices batch mismatch");
  TORCH_CHECK(scores.size(0) == batch_size, "scores batch mismatch");
  TORCH_CHECK(scores.size(1) == candidate_size, "scores candidate mismatch");
  TORCH_CHECK(context_history.size(0) == batch_size, "history batch mismatch");
  TORCH_CHECK(context_history.size(2) == LIMBS, "history must have 8 limbs");
  TORCH_CHECK(candidate_size <= MAX_CANDIDATES, "too many candidates");
  TORCH_CHECK(num_keys <= MAX_KEYS, "too many keys");
  TORCH_CHECK(write_index >= 0, "write_index must be non-negative");
  TORCH_CHECK(
      write_index < context_history_size,
      "write_index must be smaller than context history size");
  TORCH_CHECK(round_keys.size(0) == MIMC_ROUNDS, "unexpected MiMC round count");
  TORCH_CHECK(round_keys.size(1) == LIMBS, "round keys must be [rounds, limbs]");
  TORCH_CHECK(keys_mont.size(1) == LIMBS, "keys_mont must be [keys, limbs]");
  TORCH_CHECK(key_index_mont.size(1) == LIMBS, "key_index_mont must be [keys, limbs]");

  const c10::cuda::CUDAGuard device_guard(contexts.device());
  auto output = torch::empty_like(scores);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  online_fused_update_kernel<<<batch_size, 256, 0, stream>>>(
      contexts.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(),
      scores.data_ptr<float>(),
      output.data_ptr<float>(),
      context_history.data_ptr<int32_t>(),
      batch_size,
      sliding_window_size,
      context_history_size,
      candidate_size,
      num_keys,
      static_cast<int>(write_index),
      reinterpret_cast<const uint32_t*>(prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(half_prime.data_ptr()),
      reinterpret_cast<const uint32_t*>(r2.data_ptr()),
      reinterpret_cast<const uint32_t*>(one.data_ptr()),
      reinterpret_cast<const uint32_t*>(round_keys.data_ptr()),
      reinterpret_cast<const uint32_t*>(keys_mont.data_ptr()),
      reinterpret_cast<const uint32_t*>(key_index_mont.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
