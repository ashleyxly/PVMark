#include <cuda_runtime.h>
#include <stdint.h>

#define LIMBS 8

__device__ __forceinline__ void copy8(uint32_t dst[LIMBS], const uint32_t* src) {
  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) dst[i] = src[i];
}

__device__ __forceinline__ void init_zero8(uint32_t dst[LIMBS]) {
  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) dst[i] = 0U;
}

__device__ __forceinline__ void init_prime(uint32_t p[LIMBS]) {
  p[0] = 4026531841U;
  p[1] = 1138881939U;
  p[2] = 2042196113U;
  p[3] = 674490440U;
  p[4] = 2172737629U;
  p[5] = 3092268470U;
  p[6] = 3778125865U;
  p[7] = 811880050U;
}

__device__ __forceinline__ bool geq8(const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  #pragma unroll
  for (int rev = 0; rev < LIMBS; ++rev) {
    int i = LIMBS - 1 - rev;
    if (a[i] > b[i]) return true;
    if (a[i] < b[i]) return false;
  }
  return true;
}

__device__ __forceinline__ bool lt8(const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  #pragma unroll
  for (int rev = 0; rev < LIMBS; ++rev) {
    int i = LIMBS - 1 - rev;
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return false;
}

__device__ __forceinline__ void sub_assign(uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  uint64_t borrow = 0;
  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t ai = (uint64_t)a[i];
    uint64_t bi = (uint64_t)b[i] + borrow;
    if (ai >= bi) {
      a[i] = (uint32_t)(ai - bi);
      borrow = 0;
    } else {
      a[i] = (uint32_t)((1ULL << 32) + ai - bi);
      borrow = 1;
    }
  }
}

__device__ __forceinline__ void add_assign_mod(uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t p[LIMBS]) {
  uint64_t carry = 0;
  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t uv = (uint64_t)a[i] + (uint64_t)b[i] + carry;
    a[i] = (uint32_t)uv;
    carry = uv >> 32;
  }
  if (carry != 0 || geq8(a, p)) sub_assign(a, p);
}

__device__ __forceinline__ void add_const_assign_mod(uint32_t a[LIMBS], const uint32_t* constants, int offset, const uint32_t p[LIMBS]) {
  uint64_t carry = 0;
  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t uv = (uint64_t)a[i] + (uint64_t)constants[offset + i] + carry;
    a[i] = (uint32_t)uv;
    carry = uv >> 32;
  }
  if (carry != 0 || geq8(a, p)) sub_assign(a, p);
}

__device__ __forceinline__ void double_assign_mod(uint32_t a[LIMBS], const uint32_t p[LIMBS]) {
  uint64_t carry = 0;
  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t uv = (uint64_t)a[i] + (uint64_t)a[i] + carry;
    a[i] = (uint32_t)uv;
    carry = uv >> 32;
  }
  if (carry != 0 || geq8(a, p)) sub_assign(a, p);
}

__device__ __forceinline__ void mont_mul(uint32_t out[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t p[LIMBS], uint32_t n0) {
  uint64_t tmp[18];
  #pragma unroll
  for (int i = 0; i < 18; ++i) tmp[i] = 0ULL;

  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint64_t carry = 0;
    uint64_t ai = (uint64_t)a[i];
    #pragma unroll
    for (int j = 0; j < LIMBS; ++j) {
      uint64_t uv = tmp[i + j] + ai * (uint64_t)b[j] + carry;
      tmp[i + j] = (uint32_t)uv;
      carry = uv >> 32;
    }
    int k = i + LIMBS;
    while (carry != 0) {
      uint64_t uv = tmp[k] + carry;
      tmp[k] = (uint32_t)uv;
      carry = uv >> 32;
      ++k;
    }
  }

  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) {
    uint32_t m = (uint32_t)(tmp[i] * (uint64_t)n0);
    uint64_t carry = 0;
    #pragma unroll
    for (int j = 0; j < LIMBS; ++j) {
      uint64_t uv = tmp[i + j] + (uint64_t)m * (uint64_t)p[j] + carry;
      tmp[i + j] = (uint32_t)uv;
      carry = uv >> 32;
    }
    int k = i + LIMBS;
    while (carry != 0) {
      uint64_t uv = tmp[k] + carry;
      tmp[k] = (uint32_t)uv;
      carry = uv >> 32;
      ++k;
    }
  }

  #pragma unroll
  for (int i = 0; i < LIMBS; ++i) out[i] = (uint32_t)tmp[i + LIMBS];
  if (geq8(out, p)) sub_assign(out, p);
}

__device__ __forceinline__ void u32_to_mont(uint32_t out[LIMBS], uint32_t value, const uint32_t* r2, const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t canonical[LIMBS];
  init_zero8(canonical);
  canonical[0] = value;
  mont_mul(out, canonical, r2, p, n0);
}

__device__ __forceinline__ void pow5_inplace(uint32_t a[LIMBS], const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t x2[LIMBS];
  uint32_t x4[LIMBS];
  mont_mul(x2, a, a, p, n0);
  mont_mul(x4, x2, x2, p, n0);
  mont_mul(a, x4, a, p, n0);
}

__device__ __forceinline__ void pow7_inplace(uint32_t a[LIMBS], const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t x2[LIMBS];
  uint32_t x4[LIMBS];
  uint32_t x6[LIMBS];
  mont_mul(x2, a, a, p, n0);
  mont_mul(x4, x2, x2, p, n0);
  mont_mul(x6, x4, x2, p, n0);
  mont_mul(a, x6, a, p, n0);
}

__device__ __forceinline__ void mat_external2(uint32_t s0[LIMBS], uint32_t s1[LIMBS], const uint32_t p[LIMBS]) {
  uint32_t total[LIMBS];
  copy8(total, s0);
  add_assign_mod(total, s1, p);
  add_assign_mod(s0, total, p);
  add_assign_mod(s1, total, p);
}

__device__ __forceinline__ void mat_internal2(uint32_t s0[LIMBS], uint32_t s1[LIMBS], const uint32_t p[LIMBS]) {
  uint32_t total[LIMBS];
  copy8(total, s0);
  add_assign_mod(total, s1, p);
  add_assign_mod(s0, total, p);
  double_assign_mod(s1, p);
  add_assign_mod(s1, total, p);
}

__device__ __forceinline__ void mat_external3(uint32_t s0[LIMBS], uint32_t s1[LIMBS], uint32_t s2[LIMBS], const uint32_t p[LIMBS]) {
  uint32_t total[LIMBS];
  copy8(total, s0);
  add_assign_mod(total, s1, p);
  add_assign_mod(total, s2, p);
  add_assign_mod(s0, total, p);
  add_assign_mod(s1, total, p);
  add_assign_mod(s2, total, p);
}

__device__ __forceinline__ void mat_internal3(uint32_t s0[LIMBS], uint32_t s1[LIMBS], uint32_t s2[LIMBS], const uint32_t p[LIMBS]) {
  uint32_t total[LIMBS];
  copy8(total, s0);
  add_assign_mod(total, s1, p);
  add_assign_mod(total, s2, p);
  add_assign_mod(s0, total, p);
  add_assign_mod(s1, total, p);
  double_assign_mod(s2, p);
  add_assign_mod(s2, total, p);
}

__global__ void poseidon2_t2_bias_kernel(
    uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3,
    uint32_t seed4, uint32_t seed5, uint32_t seed6, uint32_t seed7,
    int vocab_size, const uint32_t* threshold, const uint32_t* r2, const uint32_t* one,
    const uint32_t* rc, uint32_t n0, float delta, float* scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size) return;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  s0[0] = seed0; s0[1] = seed1; s0[2] = seed2; s0[3] = seed3;
  s0[4] = seed4; s0[5] = seed5; s0[6] = seed6; s0[7] = seed7;
  u32_to_mont(s1, (uint32_t)idx, r2, p, n0);

  mat_external2(s0, s1, p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 2 + 1) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    mat_external2(s0, s1, p);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    mat_internal2(s0, s1, p);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 2 + 1) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    mat_external2(s0, s1, p);
  }
  mont_mul(result, s0, one, p, n0);
  if (lt8(result, threshold)) scores[idx] += delta;
}

__global__ void poseidon2_t2_bias_precomputed_kernel(
    uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3,
    uint32_t seed4, uint32_t seed5, uint32_t seed6, uint32_t seed7,
    int vocab_size, const uint32_t* threshold, const uint32_t* token_mont,
    const uint32_t* one, const uint32_t* rc, uint32_t n0, float delta, float* scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size) return;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  s0[0] = seed0; s0[1] = seed1; s0[2] = seed2; s0[3] = seed3;
  s0[4] = seed4; s0[5] = seed5; s0[6] = seed6; s0[7] = seed7;
  copy8(s1, token_mont + idx * LIMBS);

  mat_external2(s0, s1, p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 2 + 1) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    mat_external2(s0, s1, p);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    mat_internal2(s0, s1, p);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 2 + 1) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    mat_external2(s0, s1, p);
  }
  mont_mul(result, s0, one, p, n0);
  if (lt8(result, threshold)) scores[idx] += delta;
}

__global__ void poseidon2_t3_bias_kernel(
    uint32_t secret_key, uint32_t previous_token, int vocab_size, const uint32_t* threshold,
    const uint32_t* r2, const uint32_t* one, const uint32_t* rc, uint32_t n0, float delta, float* scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size) return;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t s2[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  u32_to_mont(s0, secret_key, r2, p, n0);
  u32_to_mont(s1, previous_token, r2, p, n0);
  u32_to_mont(s2, (uint32_t)idx, r2, p, n0);

  mat_external3(s0, s1, s2, p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    mat_external3(s0, s1, s2, p);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    mat_internal3(s0, s1, s2, p);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    mat_external3(s0, s1, s2, p);
  }
  mont_mul(result, s0, one, p, n0);
  if (lt8(result, threshold)) scores[idx] += delta;
}

__global__ void poseidon2_t3_bias_precomputed_kernel(
    uint32_t secret0, uint32_t secret1, uint32_t secret2, uint32_t secret3,
    uint32_t secret4, uint32_t secret5, uint32_t secret6, uint32_t secret7,
    uint32_t prev0, uint32_t prev1, uint32_t prev2, uint32_t prev3,
    uint32_t prev4, uint32_t prev5, uint32_t prev6, uint32_t prev7,
    int vocab_size, const uint32_t* threshold, const uint32_t* token_mont,
    const uint32_t* one, const uint32_t* rc, uint32_t n0, float delta, float* scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size) return;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t s2[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  s0[0] = secret0; s0[1] = secret1; s0[2] = secret2; s0[3] = secret3;
  s0[4] = secret4; s0[5] = secret5; s0[6] = secret6; s0[7] = secret7;
  s1[0] = prev0; s1[1] = prev1; s1[2] = prev2; s1[3] = prev3;
  s1[4] = prev4; s1[5] = prev5; s1[6] = prev6; s1[7] = prev7;
  copy8(s2, token_mont + idx * LIMBS);

  mat_external3(s0, s1, s2, p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    mat_external3(s0, s1, s2, p);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    mat_internal3(s0, s1, s2, p);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    mat_external3(s0, s1, s2, p);
  }
  mont_mul(result, s0, one, p, n0);
  if (lt8(result, threshold)) scores[idx] += delta;
}

__global__ void poseidon2_t2_masks_precomputed_kernel(
    int prefix_count, int vocab_size, const uint32_t* seeds, const uint32_t* threshold,
    const uint32_t* token_mont, const uint32_t* one, const uint32_t* rc, uint32_t n0,
    uint8_t* masks) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = prefix_count * vocab_size;
  if (linear >= total) return;
  int prefix_idx = linear / vocab_size;
  int token_idx = linear - prefix_idx * vocab_size;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  copy8(s0, seeds + prefix_idx * LIMBS);
  copy8(s1, token_mont + token_idx * LIMBS);

  mat_external2(s0, s1, p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 2 + 1) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    mat_external2(s0, s1, p);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    mat_internal2(s0, s1, p);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 2) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 2 + 1) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    mat_external2(s0, s1, p);
  }
  mont_mul(result, s0, one, p, n0);
  masks[linear] = lt8(result, threshold) ? 1 : 0;
}

__global__ void poseidon2_t3_masks_precomputed_kernel(
    int prefix_count, int vocab_size, const uint32_t* secret, const uint32_t* previous,
    const uint32_t* threshold, const uint32_t* token_mont, const uint32_t* one,
    const uint32_t* rc, uint32_t n0, uint8_t* masks) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = prefix_count * vocab_size;
  if (linear >= total) return;
  int prefix_idx = linear / vocab_size;
  int token_idx = linear - prefix_idx * vocab_size;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t s2[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  copy8(s0, secret);
  copy8(s1, previous + prefix_idx * LIMBS);
  copy8(s2, token_mont + token_idx * LIMBS);

  mat_external3(s0, s1, s2, p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    mat_external3(s0, s1, s2, p);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    mat_internal3(s0, s1, s2, p);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 3) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    mat_external3(s0, s1, s2, p);
  }
  mont_mul(result, s0, one, p, n0);
  masks[linear] = lt8(result, threshold) ? 1 : 0;
}

__device__ __forceinline__ void mimc_non_feistel(
    uint32_t out[LIMBS], const uint32_t x[LIMBS], const uint32_t k[LIMBS],
    const uint32_t* round_keys, const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t r[LIMBS];
  uint32_t t[LIMBS];
  init_zero8(r);

  copy8(t, k);
  add_assign_mod(t, x, p);
  pow7_inplace(t, p, n0);
  copy8(r, t);

  #pragma unroll
  for (int i = 1; i < 91; ++i) {
    copy8(t, k);
    add_assign_mod(t, r, p);
    add_assign_mod(t, round_keys + i * LIMBS, p);
    pow7_inplace(t, p, n0);
    copy8(r, t);
  }

  add_assign_mod(r, k, p);
  copy8(out, r);
}

__device__ __forceinline__ void mimc_absorb(
    uint32_t state[LIMBS], const uint32_t input[LIMBS], const uint32_t* round_keys,
    const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t nf[LIMBS];
  mimc_non_feistel(nf, input, state, round_keys, p, n0);
  add_assign_mod(state, input, p);
  add_assign_mod(state, nf, p);
}

__device__ __forceinline__ void mimc_hash2(
    uint32_t out[LIMBS], const uint32_t input0[LIMBS], const uint32_t input1[LIMBS],
    const uint32_t* round_keys, const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t state[LIMBS];
  init_zero8(state);
  mimc_absorb(state, input0, round_keys, p, n0);
  mimc_absorb(state, input1, round_keys, p, n0);
  copy8(out, state);
}

__device__ __forceinline__ void mimc_hash3(
    uint32_t out[LIMBS], const uint32_t input0[LIMBS], const uint32_t input1[LIMBS],
    const uint32_t input2[LIMBS], const uint32_t* round_keys, const uint32_t p[LIMBS],
    uint32_t n0) {
  uint32_t state[LIMBS];
  init_zero8(state);
  mimc_absorb(state, input0, round_keys, p, n0);
  mimc_absorb(state, input1, round_keys, p, n0);
  mimc_absorb(state, input2, round_keys, p, n0);
  copy8(out, state);
}

__global__ void mimc_t2_masks_precomputed_kernel(
    int prefix_count, int vocab_size, const uint32_t* seeds, const uint32_t* threshold,
    const uint32_t* token_mont, const uint32_t* one, const uint32_t* round_keys,
    uint32_t n0, uint8_t* masks) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = prefix_count * vocab_size;
  if (linear >= total) return;
  int prefix_idx = linear / vocab_size;
  int token_idx = linear - prefix_idx * vocab_size;

  uint32_t p[LIMBS];
  uint32_t seed[LIMBS];
  uint32_t token[LIMBS];
  uint32_t result_mont[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  copy8(seed, seeds + prefix_idx * LIMBS);
  copy8(token, token_mont + token_idx * LIMBS);
  mimc_hash2(result_mont, seed, token, round_keys, p, n0);
  mont_mul(result, result_mont, one, p, n0);
  masks[linear] = lt8(result, threshold) ? 1 : 0;
}

__global__ void mimc_t3_masks_precomputed_kernel(
    int prefix_count, int vocab_size, const uint32_t* secret, const uint32_t* previous,
    const uint32_t* threshold, const uint32_t* token_mont, const uint32_t* one,
    const uint32_t* round_keys, uint32_t n0, uint8_t* masks) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = prefix_count * vocab_size;
  if (linear >= total) return;
  int prefix_idx = linear / vocab_size;
  int token_idx = linear - prefix_idx * vocab_size;

  uint32_t p[LIMBS];
  uint32_t sec[LIMBS];
  uint32_t prev[LIMBS];
  uint32_t token[LIMBS];
  uint32_t result_mont[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  copy8(sec, secret);
  copy8(prev, previous + prefix_idx * LIMBS);
  copy8(token, token_mont + token_idx * LIMBS);
  mimc_hash3(result_mont, sec, prev, token, round_keys, p, n0);
  mont_mul(result, result_mont, one, p, n0);
  masks[linear] = lt8(result, threshold) ? 1 : 0;
}

__device__ __forceinline__ void poseidon_fast_mds3(
    uint32_t s0[LIMBS], uint32_t s1[LIMBS], uint32_t s2[LIMBS],
    const uint32_t* mds, const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t old0[LIMBS];
  uint32_t old1[LIMBS];
  uint32_t old2[LIMBS];
  uint32_t acc0[LIMBS];
  uint32_t acc1[LIMBS];
  uint32_t acc2[LIMBS];
  uint32_t prod[LIMBS];
  copy8(old0, s0);
  copy8(old1, s1);
  copy8(old2, s2);
  init_zero8(acc0);
  init_zero8(acc1);
  init_zero8(acc2);

  mont_mul(prod, mds + (0 * 3 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc0, prod, p);
  mont_mul(prod, mds + (0 * 3 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc0, prod, p);
  mont_mul(prod, mds + (0 * 3 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc0, prod, p);

  mont_mul(prod, mds + (1 * 3 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc1, prod, p);
  mont_mul(prod, mds + (1 * 3 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc1, prod, p);
  mont_mul(prod, mds + (1 * 3 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc1, prod, p);

  mont_mul(prod, mds + (2 * 3 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc2, prod, p);
  mont_mul(prod, mds + (2 * 3 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc2, prod, p);
  mont_mul(prod, mds + (2 * 3 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc2, prod, p);

  copy8(s0, acc0);
  copy8(s1, acc1);
  copy8(s2, acc2);
}

__device__ __forceinline__ void poseidon_fast_mds4(
    uint32_t s0[LIMBS], uint32_t s1[LIMBS], uint32_t s2[LIMBS], uint32_t s3[LIMBS],
    const uint32_t* mds, const uint32_t p[LIMBS], uint32_t n0) {
  uint32_t old0[LIMBS];
  uint32_t old1[LIMBS];
  uint32_t old2[LIMBS];
  uint32_t old3[LIMBS];
  uint32_t acc0[LIMBS];
  uint32_t acc1[LIMBS];
  uint32_t acc2[LIMBS];
  uint32_t acc3[LIMBS];
  uint32_t prod[LIMBS];
  copy8(old0, s0);
  copy8(old1, s1);
  copy8(old2, s2);
  copy8(old3, s3);
  init_zero8(acc0);
  init_zero8(acc1);
  init_zero8(acc2);
  init_zero8(acc3);

  mont_mul(prod, mds + (0 * 4 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc0, prod, p);
  mont_mul(prod, mds + (0 * 4 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc0, prod, p);
  mont_mul(prod, mds + (0 * 4 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc0, prod, p);
  mont_mul(prod, mds + (0 * 4 + 3) * LIMBS, old3, p, n0);
  add_assign_mod(acc0, prod, p);

  mont_mul(prod, mds + (1 * 4 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc1, prod, p);
  mont_mul(prod, mds + (1 * 4 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc1, prod, p);
  mont_mul(prod, mds + (1 * 4 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc1, prod, p);
  mont_mul(prod, mds + (1 * 4 + 3) * LIMBS, old3, p, n0);
  add_assign_mod(acc1, prod, p);

  mont_mul(prod, mds + (2 * 4 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc2, prod, p);
  mont_mul(prod, mds + (2 * 4 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc2, prod, p);
  mont_mul(prod, mds + (2 * 4 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc2, prod, p);
  mont_mul(prod, mds + (2 * 4 + 3) * LIMBS, old3, p, n0);
  add_assign_mod(acc2, prod, p);

  mont_mul(prod, mds + (3 * 4 + 0) * LIMBS, old0, p, n0);
  add_assign_mod(acc3, prod, p);
  mont_mul(prod, mds + (3 * 4 + 1) * LIMBS, old1, p, n0);
  add_assign_mod(acc3, prod, p);
  mont_mul(prod, mds + (3 * 4 + 2) * LIMBS, old2, p, n0);
  add_assign_mod(acc3, prod, p);
  mont_mul(prod, mds + (3 * 4 + 3) * LIMBS, old3, p, n0);
  add_assign_mod(acc3, prod, p);

  copy8(s0, acc0);
  copy8(s1, acc1);
  copy8(s2, acc2);
  copy8(s3, acc3);
}

__device__ __forceinline__ void poseidon_fast_permute3(
    uint32_t s0[LIMBS], uint32_t s1[LIMBS], uint32_t s2[LIMBS],
    const uint32_t* rc, const uint32_t* mds, const uint32_t p[LIMBS], uint32_t n0) {
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 3 + 0) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    poseidon_fast_mds3(s0, s1, s2, mds, p, n0);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 3 + 0) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    poseidon_fast_mds3(s0, s1, s2, mds, p, n0);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 3 + 0) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 3 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 3 + 2) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    poseidon_fast_mds3(s0, s1, s2, mds, p, n0);
  }
}

__device__ __forceinline__ void poseidon_fast_permute4(
    uint32_t s0[LIMBS], uint32_t s1[LIMBS], uint32_t s2[LIMBS], uint32_t s3[LIMBS],
    const uint32_t* rc, const uint32_t* mds, const uint32_t p[LIMBS], uint32_t n0) {
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    add_const_assign_mod(s0, rc, (r * 4 + 0) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 4 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 4 + 2) * LIMBS, p);
    add_const_assign_mod(s3, rc, (r * 4 + 3) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    pow5_inplace(s3, p, n0);
    poseidon_fast_mds4(s0, s1, s2, s3, mds, p, n0);
  }
  for (int r = 4; r < 60; ++r) {
    add_const_assign_mod(s0, rc, (r * 4 + 0) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 4 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 4 + 2) * LIMBS, p);
    add_const_assign_mod(s3, rc, (r * 4 + 3) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    poseidon_fast_mds4(s0, s1, s2, s3, mds, p, n0);
  }
  #pragma unroll
  for (int r = 60; r < 64; ++r) {
    add_const_assign_mod(s0, rc, (r * 4 + 0) * LIMBS, p);
    add_const_assign_mod(s1, rc, (r * 4 + 1) * LIMBS, p);
    add_const_assign_mod(s2, rc, (r * 4 + 2) * LIMBS, p);
    add_const_assign_mod(s3, rc, (r * 4 + 3) * LIMBS, p);
    pow5_inplace(s0, p, n0);
    pow5_inplace(s1, p, n0);
    pow5_inplace(s2, p, n0);
    pow5_inplace(s3, p, n0);
    poseidon_fast_mds4(s0, s1, s2, s3, mds, p, n0);
  }
}

__global__ void poseidon_fast_w3_masks_precomputed_kernel(
    int prefix_count, int vocab_size, const uint32_t* seeds, const uint32_t* threshold,
    const uint32_t* token_mont, const uint32_t* domain, const uint32_t* one,
    const uint32_t* rc, const uint32_t* mds, uint32_t n0, uint8_t* masks) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = prefix_count * vocab_size;
  if (linear >= total) return;
  int prefix_idx = linear / vocab_size;
  int token_idx = linear - prefix_idx * vocab_size;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t s2[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  copy8(s0, seeds + prefix_idx * LIMBS);
  copy8(s1, token_mont + token_idx * LIMBS);
  copy8(s2, domain);

  poseidon_fast_permute3(s0, s1, s2, rc, mds, p, n0);
  mont_mul(result, s0, one, p, n0);
  masks[linear] = lt8(result, threshold) ? 1 : 0;
}

__global__ void poseidon_fast_w4_masks_precomputed_kernel(
    int prefix_count, int vocab_size, const uint32_t* secret, const uint32_t* previous,
    const uint32_t* threshold, const uint32_t* token_mont, const uint32_t* domain,
    const uint32_t* one, const uint32_t* rc, const uint32_t* mds, uint32_t n0,
    uint8_t* masks) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = prefix_count * vocab_size;
  if (linear >= total) return;
  int prefix_idx = linear / vocab_size;
  int token_idx = linear - prefix_idx * vocab_size;

  uint32_t p[LIMBS];
  uint32_t s0[LIMBS];
  uint32_t s1[LIMBS];
  uint32_t s2[LIMBS];
  uint32_t s3[LIMBS];
  uint32_t result[LIMBS];
  init_prime(p);
  copy8(s0, secret);
  copy8(s1, previous + prefix_idx * LIMBS);
  copy8(s2, token_mont + token_idx * LIMBS);
  copy8(s3, domain);

  poseidon_fast_permute4(s0, s1, s2, s3, rc, mds, p, n0);
  mont_mul(result, s0, one, p, n0);
  masks[linear] = lt8(result, threshold) ? 1 : 0;
}

extern "C" int poseidon2_t2_bias(
    uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3,
    uint32_t seed4, uint32_t seed5, uint32_t seed6, uint32_t seed7,
    int vocab_size, const uint32_t* threshold, const uint32_t* r2, const uint32_t* one,
    const uint32_t* rc, uint32_t n0, float delta, float* scores, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int blocks = (vocab_size + threads - 1) / threads;
  poseidon2_t2_bias_kernel<<<blocks, threads, 0, stream>>>(
      seed0, seed1, seed2, seed3, seed4, seed5, seed6, seed7,
      vocab_size, threshold, r2, one, rc, n0, delta, scores);
  return (int)cudaGetLastError();
}

extern "C" int poseidon2_t2_bias_precomputed(
    uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3,
    uint32_t seed4, uint32_t seed5, uint32_t seed6, uint32_t seed7,
    int vocab_size, const uint32_t* threshold, const uint32_t* token_mont,
    const uint32_t* one, const uint32_t* rc, uint32_t n0,
    float delta, float* scores, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int blocks = (vocab_size + threads - 1) / threads;
  poseidon2_t2_bias_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      seed0, seed1, seed2, seed3, seed4, seed5, seed6, seed7,
      vocab_size, threshold, token_mont, one, rc, n0, delta, scores);
  return (int)cudaGetLastError();
}

extern "C" int poseidon2_t3_bias(
    uint32_t secret_key, uint32_t previous_token, int vocab_size, const uint32_t* threshold,
    const uint32_t* r2, const uint32_t* one, const uint32_t* rc, uint32_t n0,
    float delta, float* scores, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int blocks = (vocab_size + threads - 1) / threads;
  poseidon2_t3_bias_kernel<<<blocks, threads, 0, stream>>>(
      secret_key, previous_token, vocab_size, threshold, r2, one, rc, n0, delta, scores);
  return (int)cudaGetLastError();
}

extern "C" int poseidon2_t3_bias_precomputed(
    uint32_t secret0, uint32_t secret1, uint32_t secret2, uint32_t secret3,
    uint32_t secret4, uint32_t secret5, uint32_t secret6, uint32_t secret7,
    uint32_t prev0, uint32_t prev1, uint32_t prev2, uint32_t prev3,
    uint32_t prev4, uint32_t prev5, uint32_t prev6, uint32_t prev7,
    int vocab_size, const uint32_t* threshold, const uint32_t* token_mont,
    const uint32_t* one, const uint32_t* rc, uint32_t n0,
    float delta, float* scores, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int blocks = (vocab_size + threads - 1) / threads;
  poseidon2_t3_bias_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      secret0, secret1, secret2, secret3, secret4, secret5, secret6, secret7,
      prev0, prev1, prev2, prev3, prev4, prev5, prev6, prev7,
      vocab_size, threshold, token_mont, one, rc, n0, delta, scores);
  return (int)cudaGetLastError();
}

extern "C" int poseidon2_t2_masks_precomputed(
    int prefix_count, int vocab_size, const uint32_t* seeds, const uint32_t* threshold,
    const uint32_t* token_mont, const uint32_t* one, const uint32_t* rc, uint32_t n0,
    uint8_t* masks, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int total = prefix_count * vocab_size;
  int blocks = (total + threads - 1) / threads;
  poseidon2_t2_masks_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      prefix_count, vocab_size, seeds, threshold, token_mont, one, rc, n0, masks);
  return (int)cudaGetLastError();
}

extern "C" int poseidon2_t3_masks_precomputed(
    int prefix_count, int vocab_size, const uint32_t* secret, const uint32_t* previous,
    const uint32_t* threshold, const uint32_t* token_mont, const uint32_t* one,
    const uint32_t* rc, uint32_t n0, uint8_t* masks, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int total = prefix_count * vocab_size;
  int blocks = (total + threads - 1) / threads;
  poseidon2_t3_masks_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      prefix_count, vocab_size, secret, previous, threshold, token_mont, one, rc, n0, masks);
  return (int)cudaGetLastError();
}

extern "C" int poseidon_fast_w3_masks_precomputed(
    int prefix_count, int vocab_size, const uint32_t* seeds, const uint32_t* threshold,
    const uint32_t* token_mont, const uint32_t* domain, const uint32_t* one,
    const uint32_t* rc, const uint32_t* mds, uint32_t n0, uint8_t* masks,
    void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int total = prefix_count * vocab_size;
  int blocks = (total + threads - 1) / threads;
  poseidon_fast_w3_masks_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      prefix_count, vocab_size, seeds, threshold, token_mont, domain, one, rc, mds, n0, masks);
  return (int)cudaGetLastError();
}

extern "C" int poseidon_fast_w4_masks_precomputed(
    int prefix_count, int vocab_size, const uint32_t* secret, const uint32_t* previous,
    const uint32_t* threshold, const uint32_t* token_mont, const uint32_t* domain,
    const uint32_t* one, const uint32_t* rc, const uint32_t* mds, uint32_t n0,
    uint8_t* masks, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int total = prefix_count * vocab_size;
  int blocks = (total + threads - 1) / threads;
  poseidon_fast_w4_masks_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      prefix_count, vocab_size, secret, previous, threshold, token_mont, domain, one, rc, mds, n0, masks);
  return (int)cudaGetLastError();
}

extern "C" int mimc_t2_masks_precomputed(
    int prefix_count, int vocab_size, const uint32_t* seeds, const uint32_t* threshold,
    const uint32_t* token_mont, const uint32_t* one, const uint32_t* round_keys,
    uint32_t n0, uint8_t* masks, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int total = prefix_count * vocab_size;
  int blocks = (total + threads - 1) / threads;
  mimc_t2_masks_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      prefix_count, vocab_size, seeds, threshold, token_mont, one, round_keys, n0, masks);
  return (int)cudaGetLastError();
}

extern "C" int mimc_t3_masks_precomputed(
    int prefix_count, int vocab_size, const uint32_t* secret, const uint32_t* previous,
    const uint32_t* threshold, const uint32_t* token_mont, const uint32_t* one,
    const uint32_t* round_keys, uint32_t n0, uint8_t* masks, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int threads = 128;
  int total = prefix_count * vocab_size;
  int blocks = (total + threads - 1) / threads;
  mimc_t3_masks_precomputed_kernel<<<blocks, threads, 0, stream>>>(
      prefix_count, vocab_size, secret, previous, threshold, token_mont, one, round_keys, n0, masks);
  return (int)cudaGetLastError();
}
