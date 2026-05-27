use ff::PrimeField;
use halo2_proofs::circuit::Value;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::vec;
use core::hash;
use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};
use std::slice::RSplit;

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;
use rayon::range;


use std::time::{Duration, Instant};
use indicatif::{ProgressBar, ProgressStyle};
use std::thread;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::{Mutex, Arc};


use halo2_gadgets::poseidon::{primitives::*, Hash, Pow5Chip, Pow5Config};
use halo2_proofs::halo2curves::bn256::Fr as Fp;

use rug::{Integer, ops::Pow};
use num_bigint::{BigUint, ToBigUint};
use std::str::FromStr;
use num_traits::Num;
use pyo3::exceptions::PyValueError;

//Poseidon Field
use poseidon::hash::Poseidon;
use ark_ff::PrimeField as Poseidon_PrimeField;
use ark_bn254::Fr as Poseidon_Fr;

//Fast Poseidon
use crate::poseidon_fast::spec_width_3::PoseidonSpec as PoseidonSpec3;
use crate::poseidon_fast::spec_width_4::PoseidonSpec as PoseidonSpec4;
use crate::poseidon_fast::poseidon_params_width_3;
use crate::poseidon_fast::poseidon_params_width_4;


//Poseidon2 Field
extern crate zkhash;
use zkhash::fields::bn256::FpBN256 as Poseidon2_Fr;
use zkhash::poseidon2::{poseidon2::Poseidon2, poseidon2_instance_bn256_t_2::POSEIDON_2_BN256_PARAMS_T_2};
use zkhash::poseidon2::poseidon2_instance_bn256::POSEIDON2_BN256_PARAMS;

//MiMC Field
extern crate mimc_rs;
use mimc_rs::Fr as MiMC_Fr;
use mimc_rs::Mimc7;
extern crate ff_2;
use ff_2::*;
use ff_2::to_hex;
use ff_2::PrimeField as MiMC_PrimeField;

//Another MiMC
use arkworks_mimc::utils::mimc_hash_non_feistel;

//Blake2b Field
extern crate crypto;
use self::crypto::digest::Digest;
use self::crypto::blake2b::Blake2b;
use halo2_proofs::halo2curves::bn256::Fr as Fr_bn256;
use ark_bn254::Fr as Blake2b_Fr;
use ark_ff::BigInteger;

//Keccak256 Field
use self::crypto::sha3::Sha3;
use ark_bn254::Fr as Keccak256_Fr;


//SHA256 Field
use self::crypto::sha2::Sha256;
use ark_bn254::Fr as Sha256_Fr;


pub mod utils;
pub mod poseidon;
pub mod pedersen;
pub mod poseidon_fast;

use crate::utils::gamma_to_u64;
use crate::utils::from_str_to_fr;
use crate::utils::little_endian_u8_array_to_string;
use crate::utils::little_endian_u8_array_to_decimal_string;

// Define hash_type
#[derive(Debug, Clone, Copy)]
pub enum HashType {
    SHA256,
    BLAKE2b,
    KECCAK256,
    // PEDERSEN,
    POSEIDON,
    POSEIDON2,
    MIMC,
}


// Pyfunction invoked by Python -- Define hash computation for two inputs    input1 input2 decimal string   output decimal string
#[pyfunction]
pub fn single_two_inputs_hash_computation_decimal(input1: String, input2: String, hash_type: i32) -> PyResult<String> {
    let mut final_result = String::new();
    match hash_type {
        0 => {
            let mut hasher = Sha256::new();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);

            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            // let hex = format!("{:x}", result2);
            final_result = result2.to_string();

        }
        1 => {
            let mut hasher = Blake2b::new(32);
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            // let hex = format!("{:x}", result2);
            final_result = result2.to_string();
            
        }
        2 => {
            let mut hasher = Sha3::keccak256();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            // let hex = format!("{:x}", result2);
            final_result = result2.to_string();
        }

        3 => {
            let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
            let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
                .hash(input);
            // let res_string = little_endian_u8_array_to_string(&result.to_bytes());
            let res_string = little_endian_u8_array_to_decimal_string(&result.to_bytes());
            final_result = res_string.clone();
        }
        4 => {
            let poseidon = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
            let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
            let input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from_str(&input2).unwrap()];
            let perm = poseidon.permutation(&input);
            let result = perm[0];
            let result_bigint = result.into_bigint();
            let result_bigint_string = result_bigint.to_string();
            let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            // let hex_string = format!("{:x}", decimal_bigint);
            final_result = decimal_bigint.to_string();
        }
        5 => {
            let res = mimc_hash_non_feistel(input1, input2, None);
            let decimal_number = BigUint::from_str_radix(res.as_str(), 16).unwrap();
            final_result = decimal_number.to_string();
        }
        i32::MIN..=-1_i32 | 6_i32..=i32::MAX => {
            final_result = "Not implemented yet".to_string();
        }
    }
    Ok(final_result)
}

// Define hash computation for two inputs    input1 input2 decimal string   output decimal string
pub fn two_inputs_hash_computation_decimal(input1: String, input2: String, hash_type: HashType) -> String {
    let mut final_result = String::new();
    match hash_type {
        HashType::SHA256 => {
            let mut hasher = Sha256::new();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);

            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            // let hex = format!("{:x}", result2);
            final_result = result2.to_string();

        }
        HashType::BLAKE2b => {
            let mut hasher = Blake2b::new(32);
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            // let hex = format!("{:x}", result2);
            final_result = result2.to_string();
            
        }
        HashType::KECCAK256 => {
            let mut hasher = Sha3::keccak256();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            // let hex = format!("{:x}", result2);
            final_result = result2.to_string();
        }

        HashType::POSEIDON => {
            let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
            let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
                .hash(input);
            // let res_string = little_endian_u8_array_to_string(&result.to_bytes());
            let res_string = little_endian_u8_array_to_decimal_string(&result.to_bytes());
            final_result = res_string.clone();
        }
        HashType::POSEIDON2 => {
            let poseidon = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
            let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
            let input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from_str(&input2).unwrap()];
            let perm = poseidon.permutation(&input);
            let result = perm[0];
            let result_bigint = result.into_bigint();
            let result_bigint_string = result_bigint.to_string();
            let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            // let hex_string = format!("{:x}", decimal_bigint);
            final_result = decimal_bigint.to_string();
        }
        HashType::MIMC => {
            let res = mimc_hash_non_feistel(input1, input2, None);
            let decimal_number = BigUint::from_str_radix(res.as_str(), 16).unwrap();
            final_result = decimal_number.to_string();
        }
    }
    final_result
}

// Define hash computation for two inputs    input1 input2 decimal string   output decimal string
pub fn multi_two_inputs_hash_computation_decimal(current_hash: String, data: Vec<String>, hash_type: HashType) -> String {
    let mut result = current_hash.clone();

    for i in 0..data.len() {
        result = two_inputs_hash_computation_decimal(result, data[i].clone(), hash_type);
    }
    
    result
}

#[pyfunction]
fn _compute_keys_use_hash(ngrams: Vec<Vec<u64>>, indices: Vec<Vec<u64>>, keys: Vec<u64>, hash_type: i32) -> PyResult<(Vec<Vec<Vec<String>>>, Vec<String>)>
{
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    // println!("ngrams.size: {:?}, indices.size: {:?}, keys.size: {:?}", ngrams.len(), indices.len(), keys.len());
    // println!("ngrams[0].size: {:?}, indices[0].size: {:?}", ngrams[0].len(), indices[0].len());
    let batch_size = ngrams.len();
    let sliding_window_size = ngrams[0].len();
    let candidate_token_size = indices[0].len();
    let num_keys = keys.len();
    // println!("batch_size: {}, sliding_window_size: {}, candidate_token_size: {}, num_keys: {}", batch_size, sliding_window_size, candidate_token_size, num_keys);

    if sliding_window_size < 2 {
        println!("The sliding window size must be greater than 1");
        return Ok((vec![], vec![]));
    }

    let hash_result_with_just_context: Vec<String> = (0..batch_size).into_par_iter().map(|batch| {
        let mut hash_result_this_batch_setup = "1".to_string();
        let hash_result_with_just_context_this_batch = multi_two_inputs_hash_computation_decimal(
            hash_result_this_batch_setup.clone(),
            ngrams[batch].clone().iter().map(|x| x.to_string()).collect(),
            hash_type_enum,
        );
        hash_result_with_just_context_this_batch
    }).collect();
    // println!("hash_result_with_just_context: {:?}", hash_result_with_just_context);

    let hash_result: Vec<Vec<Vec<String>>> = (0..batch_size)
        .into_par_iter()  // 外层循环并行化
        .map(|batch| {
            // 并行化内部循环处理
            let hash_result_this_batch: Vec<Vec<String>> = (0..candidate_token_size)
                .into_par_iter()  // 内层循环并行化
                .map(|index| {
                    let hash_result_temp_inner = multi_two_inputs_hash_computation_decimal(
                        hash_result_with_just_context[batch].clone(),
                        vec![indices[batch][index].to_string()],
                        hash_type_enum,
                    );

                    // 对每个 key 并行化处理
                    let hash_result_temp_vec: Vec<String> = (0..num_keys)
                        .into_par_iter()  // 内层循环并行化
                        .map(|key_index| {
                            let hash_result_temp = multi_two_inputs_hash_computation_decimal(
                                hash_result_temp_inner.clone(),
                                vec![keys[key_index].to_string()],
                                hash_type_enum,
                            );
                            hash_result_temp
                        })
                        .collect();

                    hash_result_temp_vec
                })
                .collect();  // 将所有结果收集到 Vec<Vec<String>> 中

            hash_result_this_batch
        })
        .collect();  // 将每个 batch 的结果收集到最终的 hash_result 中
    // println!("hash_result: {:?}", hash_result);

    return Ok((hash_result, hash_result_with_just_context));
}

#[pyfunction]
fn _sample_g_values_use_hash(ngrams_keys: Vec<Vec<Vec<String>>>, field_prime: String, hash_type: i32) -> PyResult<(Vec<Vec<Vec<u64>>>)> {
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    
    let batch_size = ngrams_keys.len();
    let candidate_token_size = ngrams_keys[0].len();
    let num_keys = ngrams_keys[0][0].len();
    let big_prime_int = match BigUint::from_str_radix(field_prime.as_str(), 16) {
        Ok(v) => v,
        Err(_e) => {
            println!("Failed to parse field_prime to BigUint");
            return Err(PyValueError::new_err("Failed to parse field_prime to BigUint"));
        }
    };
    let threshold = big_prime_int / BigUint::from(2u64);

    let g_value: Vec<Vec<Vec<u64>>> = (0..batch_size)
        .into_par_iter()
        .map(|batch| {
            let g_value_this_batch: Vec<Vec<u64>> = (0..candidate_token_size)
                .into_par_iter()
                .map(|index| {
                    let g_value_temp_inner: Vec<u64> = (0..num_keys)
                        .into_par_iter()
                        .map(|key_index| {
                            let g_value_temp = multi_two_inputs_hash_computation_decimal(
                                ngrams_keys[batch][index][key_index].clone(),
                                vec![key_index.to_string()],
                                hash_type_enum,
                            );
                            let g_value_temp_int = match BigUint::from_str_radix(g_value_temp.as_str(), 10) {
                                Ok(v) => v,
                                Err(_e) => {
                                    println!("Failed to parse g_value_temp to BigUint");
                                    return 0;
                                }
                            };
                            if g_value_temp_int > threshold {
                                1 as u64
                            }
                            else {
                                0 as u64
                            }
                        })
                        .collect();

                    g_value_temp_inner
                })
                .collect();

            g_value_this_batch
        })
        .collect();


    return Ok(g_value);
}


#[pyfunction]
fn _compute_ngram_keys_use_hash(ngrams: Vec<Vec<Vec<u64>>>, keys: Vec<u64>, hash_type: i32) -> PyResult<Vec<Vec<Vec<String>>>>
{
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    let batch_size = ngrams.len();
    let num_ngram = ngrams[0].len();
    let _ngram_len = ngrams[0][0].len();
    let num_keys = keys.len();

    let hash_result_with_just_context: Vec<Vec<String>> = (0..batch_size).into_par_iter().map(|batch| {
        let mut hash_result_this_batch_setup = "1".to_string();

        let hash_result_inner: Vec<String> = (0..num_ngram).into_par_iter().map(|ngram_index| {
            let hash_result_with_just_context_this_batch = multi_two_inputs_hash_computation_decimal(
                hash_result_this_batch_setup.clone(),
                ngrams[batch][ngram_index].clone().iter().map(|x| x.to_string()).collect(),
                hash_type_enum,
            );
            hash_result_with_just_context_this_batch
        }).collect();
        hash_result_inner
    }).collect();
    // println!("hash_result_with_just_context: {:?}", hash_result_with_just_context);

    let hash_result: Vec<Vec<Vec<String>>> = (0..batch_size)
        .into_par_iter()  // 外层循环并行化
        .map(|batch| {
            // 并行化内部循环处理
            let hash_result_this_batch: Vec<Vec<String>> = (0..num_ngram)
                .into_par_iter()  // 内层循环并行化
                .map(|ngram_index| {
                    // 对每个 key 并行化处理
                    let hash_result_temp_vec: Vec<String> = (0..num_keys)
                        .into_par_iter()  // 内层循环并行化
                        .map(|key_index| {
                            let hash_result_temp = multi_two_inputs_hash_computation_decimal(
                                hash_result_with_just_context[batch][ngram_index].clone(),
                                vec![keys[key_index].to_string()],
                                hash_type_enum,
                            );
                            hash_result_temp
                        })
                        .collect();

                    hash_result_temp_vec
                })
                .collect();  // 将所有结果收集到 Vec<Vec<String>> 中

            hash_result_this_batch
        })
        .collect();  // 将每个 batch 的结果收集到最终的 hash_result 中
    // println!("hash_result: {:?}", hash_result);

    return Ok(hash_result);
}


// ****************************************************************** //
// Under BN254 Field
pub fn compute_LCG_random(current_hash: String, data: Vec<u64>, multiplier: Option<u64>, increment: Option<u64>) -> String {
    let multiplier = multiplier.unwrap_or(6364136223846793005);
    let increment = increment.unwrap_or(1);
    
    // let current_hash_fr = from_str_to_fr(&current_hash).unwrap();
    let multiplier_fr = Fp::from(multiplier);
    let increment_fr = Fp::from(increment);

    let mut result = from_str_to_fr(&current_hash).unwrap();

    for i in 0..data.len() {
        let data_fr = Fp::from(data[i]);
        result = result + data_fr;
        result = result * multiplier_fr;
        result = result + increment_fr;
    }

    little_endian_u8_array_to_decimal_string(&result.to_bytes())
}

#[pyfunction]
pub fn compute_LCG_random_use_rust(current_hash: String, data: Vec<Vec<u64>>, multiplier: Option<u64>, increment: Option<u64>) -> PyResult<Vec<String>> {
    let batch_size = data.len();

    let multiplier = multiplier.unwrap_or(6364136223846793005);
    let increment = increment.unwrap_or(1);
    
    // let current_hash_fr = from_str_to_fr(&current_hash).unwrap();
    let multiplier_fr = Fp::from(multiplier);
    let increment_fr = Fp::from(increment);

    // let mut result = from_str_to_fr(&current_hash).unwrap();

    let hash_result: Vec<String> = (0..batch_size).into_par_iter().map(|batch| {
        let data_this_batch = data[batch].clone();
        let mut result = from_str_to_fr(&current_hash).unwrap();
        for i in 0..data_this_batch.len() {
            let data_fr = Fp::from(data_this_batch[i]);
            result = result + data_fr;
            result = result * multiplier_fr;
            result = result + increment_fr;
        }
        little_endian_u8_array_to_decimal_string(&result.to_bytes())
    }).collect();


    Ok(hash_result)
}


#[pyfunction]
fn _compute_keys_use_LCG(ngrams: Vec<Vec<u64>>, indices: Vec<Vec<u64>>, keys: Vec<u64>) -> PyResult<(Vec<Vec<Vec<String>>>, Vec<String>)>
{
    // println!("ngrams.size: {:?}, indices.size: {:?}, keys.size: {:?}", ngrams.len(), indices.len(), keys.len());
    // println!("ngrams[0].size: {:?}, indices[0].size: {:?}", ngrams[0].len(), indices[0].len());
    let batch_size = ngrams.len();
    let sliding_window_size = ngrams[0].len();
    let candidate_token_size = indices[0].len();
    let num_keys = keys.len();
    // println!("batch_size: {}, sliding_window_size: {}, candidate_token_size: {}, num_keys: {}", batch_size, sliding_window_size, candidate_token_size, num_keys);

    if sliding_window_size < 2 {
        println!("The sliding window size must be greater than 1");
        return Ok((vec![], vec![]));
    }

    let hash_result_with_just_context: Vec<String> = (0..batch_size).into_par_iter().map(|batch| {
        let mut hash_result_this_batch_setup = "1".to_string();
        let hash_result_with_just_context_this_batch = compute_LCG_random(
            hash_result_this_batch_setup.clone(),
            ngrams[batch].clone(),
            None,
            None,
        );
        hash_result_with_just_context_this_batch
    }).collect();
    // println!("hash_result_with_just_context: {:?}", hash_result_with_just_context);

    let hash_result: Vec<Vec<Vec<String>>> = (0..batch_size)
        .into_par_iter()  // 外层循环并行化
        .map(|batch| {
            // 并行化内部循环处理
            let hash_result_this_batch: Vec<Vec<String>> = (0..candidate_token_size)
                .into_par_iter()  // 内层循环并行化
                .map(|index| {
                    let hash_result_temp_inner = compute_LCG_random(
                        hash_result_with_just_context[batch].clone(),
                        vec![indices[batch][index]],
                        None,
                        None,
                    );

                    // 对每个 key 并行化处理
                    let hash_result_temp_vec: Vec<String> = (0..num_keys)
                        .into_par_iter()  // 内层循环并行化
                        .map(|key_index| {
                            let hash_result_temp = compute_LCG_random(
                                hash_result_temp_inner.clone(),
                                vec![keys[key_index]],
                                None,
                                None,
                            );
                            hash_result_temp
                        })
                        .collect();

                    hash_result_temp_vec
                })
                .collect();  // 将所有结果收集到 Vec<Vec<String>> 中

            hash_result_this_batch
        })
        .collect();  // 将每个 batch 的结果收集到最终的 hash_result 中
    // println!("hash_result: {:?}", hash_result);

    return Ok((hash_result, hash_result_with_just_context));
}

#[pyfunction]
fn _sample_g_values_use_LCG(ngrams_keys: Vec<Vec<Vec<String>>>, field_prime: String) -> PyResult<(Vec<Vec<Vec<u64>>>)> {
    let batch_size = ngrams_keys.len();
    let candidate_token_size = ngrams_keys[0].len();
    let num_keys = ngrams_keys[0][0].len();
    let big_prime_int = match BigUint::from_str_radix(field_prime.as_str(), 16) {
        Ok(v) => v,
        Err(_e) => {
            println!("Failed to parse field_prime to BigUint");
            return Err(PyValueError::new_err("Failed to parse field_prime to BigUint"));
        }
    };
    let threshold = big_prime_int / BigUint::from(2u64);

    let g_value: Vec<Vec<Vec<u64>>> = (0..batch_size)
        .into_par_iter()
        .map(|batch| {
            let g_value_this_batch: Vec<Vec<u64>> = (0..candidate_token_size)
                .into_par_iter()
                .map(|index| {
                    let g_value_temp_inner: Vec<u64> = (0..num_keys)
                        .into_par_iter()
                        .map(|key_index| {
                            let g_value_temp = compute_LCG_random(
                                ngrams_keys[batch][index][key_index].clone(),
                                vec![key_index as u64],
                                None,
                                None,
                            );
                            let g_value_temp_int = match BigUint::from_str_radix(g_value_temp.as_str(), 10) {
                                Ok(v) => v,
                                Err(_e) => {
                                    println!("Failed to parse g_value_temp to BigUint");
                                    return 0;
                                }
                            };
                            if g_value_temp_int > threshold {
                                1 as u64
                            }
                            else {
                                0 as u64
                            }
                        })
                        .collect();

                    g_value_temp_inner
                })
                .collect();

            g_value_this_batch
        })
        .collect();


    return Ok(g_value);
}


#[pyfunction]
fn _compute_ngram_keys_use_LCG(ngrams: Vec<Vec<Vec<u64>>>, keys: Vec<u64>) -> PyResult<Vec<Vec<Vec<String>>>>
{
    // println!("ngrams.size: {:?}, indices.size: {:?}, keys.size: {:?}", ngrams.len(), indices.len(), keys.len());
    // println!("ngrams[0].size: {:?}, indices[0].size: {:?}", ngrams[0].len(), indices[0].len());
    let batch_size = ngrams.len();
    let num_ngram = ngrams[0].len();
    let ngram_len = ngrams[0][0].len();
    let num_keys = keys.len();
    // println!("batch_size: {}, sliding_window_size: {}, candidate_token_size: {}, num_keys: {}", batch_size, sliding_window_size, candidate_token_size, num_keys);

    // if ngram_len < 2 {
    //     println!("The sliding window size must be greater than 1");
    //     return Ok((vec![], vec![]));
    // }

    let hash_result_with_just_context: Vec<Vec<String>> = (0..batch_size).into_par_iter().map(|batch| {
        let mut hash_result_this_batch_setup = "1".to_string();

        let hash_result_inner: Vec<String> = (0..num_ngram).into_par_iter().map(|ngram_index| {
            let hash_result_with_just_context_this_batch = compute_LCG_random(
                hash_result_this_batch_setup.clone(),
                ngrams[batch][ngram_index].clone(),
                None,
                None,
            );
            hash_result_with_just_context_this_batch
        }).collect();
        hash_result_inner
    }).collect();
    // println!("hash_result_with_just_context: {:?}", hash_result_with_just_context);

    let hash_result: Vec<Vec<Vec<String>>> = (0..batch_size)
        .into_par_iter()  // 外层循环并行化
        .map(|batch| {
            // 并行化内部循环处理
            let hash_result_this_batch: Vec<Vec<String>> = (0..num_ngram)
                .into_par_iter()  // 内层循环并行化
                .map(|ngram_index| {
                    // 对每个 key 并行化处理
                    let hash_result_temp_vec: Vec<String> = (0..num_keys)
                        .into_par_iter()  // 内层循环并行化
                        .map(|key_index| {
                            let hash_result_temp = compute_LCG_random(
                                hash_result_with_just_context[batch][ngram_index].clone(),
                                vec![keys[key_index]],
                                None,
                                None,
                            );
                            hash_result_temp
                        })
                        .collect();

                    hash_result_temp_vec
                })
                .collect();  // 将所有结果收集到 Vec<Vec<String>> 中

            hash_result_this_batch
        })
        .collect();  // 将每个 batch 的结果收集到最终的 hash_result 中
    // println!("hash_result: {:?}", hash_result);

    return Ok(hash_result);
}



// Define hash computation for two inputs    input1 input2 decimal string
pub fn two_inputs_hash_computation_used_only_in_sac_test(input1: String, input2: String, hash_type: HashType) -> String {
    let mut final_result = String::new();
    match hash_type {
        HashType::SHA256 => {
            let mut hasher = Sha256::new();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            // let hex = hex::encode(out);
            // result = hex;
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            // let result2 = decimal_number % modulo;
            let hex = format!("{:x}", decimal_number);
            final_result = hex.clone();

        }
        HashType::BLAKE2b => {
            let mut hasher = Blake2b::new(32);
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            // let result2 = decimal_number % modulo;
            let hex = format!("{:x}", decimal_number);
            final_result = hex.clone();
            
        }
        HashType::KECCAK256 => {
            let mut hasher = Sha3::keccak256();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            // let result2 = decimal_number % modulo;
            let hex = format!("{:x}", decimal_number);
            final_result = hex.clone();
        }
        // HashType::PEDERSEN => {
        //     let mut hasher = Pedersen::new();
        //     hasher.input_str(&input1);
        //     hasher.input_str(&input2);
        //     let hex = hasher.result_str();
        //     hex
        // }
        HashType::POSEIDON => {
            // let poseidon = Poseidon::new();
            // let mut sk = Poseidon_Fr::from_str(&input1).unwrap();
            // let input: Vec<Poseidon_Fr> = vec![sk.clone(), Poseidon_Fr::from_str(&input2).unwrap()];
            // let result = poseidon.hash(input.clone()).unwrap();
            // let result_bigint = result.into_bigint();
            // let result_bigint_string = result_bigint.to_string();
            // let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            // let hex_string = format!("{:x}", decimal_bigint);
            // final_result = hex_string.clone();
            let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
            let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
                .hash(input);
            let res_string = little_endian_u8_array_to_string(&result.to_bytes());
            final_result = res_string.clone();
        }
        HashType::POSEIDON2 => {
            let poseidon = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
            let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
            let input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from_str(&input2).unwrap()];
            let perm = poseidon.permutation(&input);
            let result = perm[0];
            let result_bigint = result.into_bigint();
            let result_bigint_string = result_bigint.to_string();
            let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            let hex_string = format!("{:x}", decimal_bigint);
            final_result = hex_string.clone();
        }
        HashType::MIMC => {
            // let mut hasher = Mimc7::new(91);
            // let input1_fr = MiMC_Fr::from_str(&input1).unwrap();
            // let input2_fr = MiMC_Fr::from_str(&input2).unwrap();
            // let input: Vec<MiMC_Fr> = vec![input1_fr.clone(), input2_fr.clone()];
            // let result = hasher.multi_hash(input.clone(), &MiMC_Fr::zero());
            // let hex_string = to_hex(&result);
            // final_result = hex_string.clone();
            let res = mimc_hash_non_feistel(input1, input2, None);
            final_result = res.clone();
        }
    }
    final_result
}

// Define hash computation for two inputs    input1 input2 decimal string
pub fn two_inputs_hash_computation(input1: String, input2: String, hash_type: HashType) -> String {
    let mut final_result = String::new();
    match hash_type {
        HashType::SHA256 => {
            let mut hasher = Sha256::new();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            // let hex = hex::encode(out);
            // result = hex;
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();

        }
        HashType::BLAKE2b => {
            let mut hasher = Blake2b::new(32);
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();
            
        }
        HashType::KECCAK256 => {
            let mut hasher = Sha3::keccak256();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();
        }
        // HashType::PEDERSEN => {
        //     let mut hasher = Pedersen::new();
        //     hasher.input_str(&input1);
        //     hasher.input_str(&input2);
        //     let hex = hasher.result_str();
        //     hex
        // }
        HashType::POSEIDON => {
            // let poseidon = Poseidon::new();
            // let mut sk = Poseidon_Fr::from_str(&input1).unwrap();
            // let input: Vec<Poseidon_Fr> = vec![sk.clone(), Poseidon_Fr::from_str(&input2).unwrap()];
            // let result = poseidon.hash(input.clone()).unwrap();
            // let result_bigint = result.into_bigint();
            // let result_bigint_string = result_bigint.to_string();
            // let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            // let hex_string = format!("{:x}", decimal_bigint);
            // final_result = hex_string.clone();
            let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
            let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
                .hash(input);
            let res_string = little_endian_u8_array_to_string(&result.to_bytes());
            final_result = res_string.clone();
        }
        HashType::POSEIDON2 => {
            let poseidon = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
            let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
            let input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from_str(&input2).unwrap()];
            let perm = poseidon.permutation(&input);
            let result = perm[0];
            let result_bigint = result.into_bigint();
            let result_bigint_string = result_bigint.to_string();
            let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            let hex_string = format!("{:x}", decimal_bigint);
            final_result = hex_string.clone();
        }
        HashType::MIMC => {
            // let mut hasher = Mimc7::new(91);
            // let input1_fr = MiMC_Fr::from_str(&input1).unwrap();
            // let input2_fr = MiMC_Fr::from_str(&input2).unwrap();
            // let input: Vec<MiMC_Fr> = vec![input1_fr.clone(), input2_fr.clone()];
            // let result = hasher.multi_hash(input.clone(), &MiMC_Fr::zero());
            // let hex_string = to_hex(&result);
            // final_result = hex_string.clone();
            let res = mimc_hash_non_feistel(input1, input2, None);
            final_result = res.clone();
        }
    }
    final_result
}


fn multiple_two_inputs_hash_computation(seed: String, vocab_size: i32, hash_type: HashType) -> Vec<String> {
    let mut this_round_token_hash: Vec<String> = Vec::new();
    match hash_type {
        HashType::POSEIDON => {
            // let poseidon = Poseidon::new();
            // let mut sk = Poseidon_Fr::from_str(&seed).unwrap();
            // // let len = vocab_size;
            // let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
            //     let mut input: Vec<Poseidon_Fr> = vec![sk.clone(), Poseidon_Fr::from(i as u64)];
            //     let result = poseidon.hash(input.clone()).unwrap();
            //     let result_bigint = result.into_bigint();
            //     let result_bigint_string = result_bigint.to_string();
            //     let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            //     let hex_string = format!("{:x}", decimal_bigint);
            //     hex_string
            // }).collect();
            let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
                let mut hash_inputs = [from_str_to_fr(&seed).unwrap(), Fp::from(i as u64)];
                let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
                    .hash(hash_inputs);

                let res_string = little_endian_u8_array_to_string(&result.to_bytes());
                res_string
            }).collect();
            this_round_token_hash.extend_from_slice(&this_round_token_hash_temp);
        }
        HashType::POSEIDON2 => {
            let poseidon2 = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
            let mut sk = Poseidon2_Fr::from_str(&seed).unwrap();
            let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
                let mut input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from(i as u64)];
                let perm = poseidon2.permutation(&input);
                let result = perm[0];
                let result_bigint = result.into_bigint();
                let result_bigint_string = result_bigint.to_string();
                let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
                let hex_string = format!("{:x}", decimal_bigint);
                hex_string
            }).collect();
            this_round_token_hash.extend_from_slice(&this_round_token_hash_temp);

        }
        HashType::MIMC => {
            // let mimc7 = Mimc7::new(91);
            // let mut sk = MiMC_Fr::from_str(&seed).unwrap();
            // let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
            //     let mut input: Vec<MiMC_Fr> = vec![sk.clone(), MiMC_Fr::from_str(i.to_string().as_str()).unwrap()];
            //     let result = mimc7.multi_hash(input.clone(), &MiMC_Fr::zero());
            //     // let result_bigint = result.into_repr();
            //     let hex_string = to_hex(&result);
            //     hex_string
            // }).collect();
            let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
                let res = mimc_hash_non_feistel(seed.clone(), i.to_string(), None);
                res
            }).collect();
            this_round_token_hash.extend_from_slice(&this_round_token_hash_temp);

        }
        // HashType::PEDERSEN => {

        // }
        HashType::SHA256 => {
            let sk = Sha256_Fr::from_str(&seed).unwrap();
            let sk_bigint = sk.into_bigint();
            let sk_bytes = sk_bigint.to_bytes_be();
            let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
                let mut inputs: Vec<u8> = vec![];
                inputs.append(&mut sk_bytes.clone());
                let i_fr = Sha256_Fr::from(i as u64);
                let i_bigint = i_fr.into_bigint();
                let i_bytes = i_bigint.to_bytes_be();
                inputs.append(&mut i_bytes.clone());
                let mut hasher = Sha256::new();
                hasher.input(&inputs);
                // let mut out = [0u8; 32];
                // hasher.result(&mut out);
                // let hex = hex::encode(out);
                let result = hasher.result_str();
                let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
                let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
                let result2 = decimal_number % modulo;
                let hex = format!("{:x}", result2);
                hex
            }).collect();
            this_round_token_hash.extend_from_slice(&this_round_token_hash_temp);


        }
        HashType::BLAKE2b => {
            let sk = Blake2b_Fr::from_str(&seed).unwrap();
            let sk_bigint = sk.into_bigint();
            let sk_bytes = sk_bigint.to_bytes_be();
            let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
                let mut inputs: Vec<u8> = vec![];
                inputs.append(&mut sk_bytes.clone());
                let i_fr = Blake2b_Fr::from(i as u64);
                let i_bigint = i_fr.into_bigint();
                let i_bytes = i_bigint.to_bytes_be();
                inputs.append(&mut i_bytes.clone());
                let mut hasher = Blake2b::new(32);
                hasher.input(&inputs);
                // let mut out = [0u8; 32];
                // hasher.result(&mut out);
                // let mut out_u64 = [0u64; 4];
                // for i in 0..4 {
                //     out_u64[i] = u64::from_le_bytes([
                //         out[i * 8],
                //         out[i * 8 + 1],
                //         out[i * 8 + 2],
                //         out[i * 8 + 3],
                //         out[i * 8 + 4],
                //         out[i * 8 + 5],
                //         out[i * 8 + 6],
                //         out[i * 8 + 7],
                //     ]);
                // }
                // let hex_list: Vec<String> = out_u64.iter().map(|&x| format!("{:x}", x)).collect();
                // let mut u64_array: [u64; 4] = [0; 4];
                // for (i, hex_str) in hex_list.iter().enumerate() {
                //     u64_array[i] = u64::from_str_radix(hex_str, 16).expect("Failed to parse hex string to u64");
                // }
                // let result = Fr_bn256::from_raw(u64_array.clone());
                // let data = result.to_bytes();
                // let hex_string: String = data.iter().map(|b| format!("{:x}", b)).collect();
                // hex_string
                let result = hasher.result_str();
                let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
                let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
                let result2 = decimal_number % modulo;
                let hex = format!("{:x}", result2);
                hex
            }).collect();
            this_round_token_hash.extend_from_slice(&this_round_token_hash_temp);
        }
        HashType::KECCAK256 => {
            let sk = Keccak256_Fr::from_str(&seed).unwrap();
            let sk_bigint = sk.into_bigint();
            let sk_bytes = sk_bigint.to_bytes_be();
            let this_round_token_hash_temp: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
                let mut inputs: Vec<u8> = vec![];
                inputs.append(&mut sk_bytes.clone());
                let i_fr = Keccak256_Fr::from(i as u64);
                let i_bigint = i_fr.into_bigint();
                let i_bytes = i_bigint.to_bytes_be();
                inputs.append(&mut i_bytes.clone());
                let mut hasher = Sha3::keccak256();
                hasher.input(&inputs);
                // let mut out = [0u8; 32];
                // hasher.result(&mut out);
                // let hex = hex::encode(out);
                let result = hasher.result_str();
                let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
                let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
                let result2 = decimal_number % modulo;
                let hex = format!("{:x}", result2);
                hex
            }).collect();
            this_round_token_hash.extend_from_slice(&this_round_token_hash_temp);
        }

        
    }
    this_round_token_hash

}


// Define hash computation for three inputs    input1 input2 input3 decimal string
fn three_inputs_hash_computation(input1: String, input2: String, input3: String, hash_type: HashType) -> String {
    let mut final_result = String::new();
    match hash_type {
        HashType::SHA256 => {
            let mut hasher = Sha256::new();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            let input3_bytes = input3.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            inputs.append(&mut input3_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            // let hex = hex::encode(out);
            // result = hex;
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();

        }
        HashType::BLAKE2b => {
            let mut hasher = Blake2b::new(32);
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            let input3_bytes = input3.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            inputs.append(&mut input3_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();
            
        }
        HashType::KECCAK256 => {
            let mut hasher = Sha3::keccak256();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            let input3_bytes = input3.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            inputs.append(&mut input3_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();
        }
        // HashType::PEDERSEN => {
        //     let mut hasher = Pedersen::new();
        //     hasher.input_str(&input1);
        //     hasher.input_str(&input2);
        //     let hex = hasher.result_str();
        //     hex
        // }
        HashType::POSEIDON => {
            // let poseidon = Poseidon::new();
            // let input: Vec<Poseidon_Fr> = vec![Poseidon_Fr::from_str(&input1).unwrap(), Poseidon_Fr::from_str(&input2).unwrap(), Poseidon_Fr::from_str(&input3).unwrap()];
            // let result = poseidon.hash(input.clone()).unwrap();
            // let result_bigint = result.into_bigint();
            // let result_bigint_string = result_bigint.to_string();
            // let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            // let hex_string = format!("{:x}", decimal_bigint);
            // final_result = hex_string.clone();
            let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap(), from_str_to_fr(&input3).unwrap()];
            let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec4, ConstantLength<3>, 4, 3>::init()
                .hash(input);
            let res_string = little_endian_u8_array_to_string(&result.to_bytes());
            final_result = res_string.clone();
        }
        HashType::POSEIDON2 => {
            let poseidon = Poseidon2::new(&POSEIDON2_BN256_PARAMS);
            // let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
            let input: Vec<Poseidon2_Fr> = vec![Poseidon2_Fr::from_str(&input1).unwrap(), Poseidon2_Fr::from_str(&input2).unwrap(), Poseidon2_Fr::from_str(&input3).unwrap()];
            let perm = poseidon.permutation(&input);
            let result = perm[0];
            let result_bigint = result.into_bigint();
            let result_bigint_string = result_bigint.to_string();
            let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            let hex_string = format!("{:x}", decimal_bigint);
            final_result = hex_string.clone();
        }
        HashType::MIMC => {
            // let mut hasher = Mimc7::new(91);
            // let input1_fr = MiMC_Fr::from_str(&input1).unwrap();
            // let input2_fr = MiMC_Fr::from_str(&input2).unwrap();
            // let input3_fr = MiMC_Fr::from_str(&input3).unwrap();
            // let input: Vec<MiMC_Fr> = vec![input1_fr.clone(), input2_fr.clone(), input3_fr.clone()];
            // let result = hasher.multi_hash(input.clone(), &MiMC_Fr::zero());
            // let hex_string = to_hex(&result);
            // final_result = hex_string.clone();
            let res = mimc_hash_non_feistel(input1, input2, Some(input3));
            final_result = res.clone();
        }
    }
    final_result
}


#[pyfunction]
fn single_two_inputs_hash_computation(input1: String, input2: String, hash_type: i32) -> String {
    let mut final_result = String::new();
    match hash_type {
        0 => {
            let mut hasher = Sha256::new();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            // let hex = hex::encode(out);
            // result = hex;
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();

        }
        1 => {
            let mut hasher = Blake2b::new(32);
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();
            
        }
        2 => {
            let mut hasher = Sha3::keccak256();
            let mut inputs: Vec<u8> = vec![];
            let input1_bytes = input1.as_bytes();
            let input2_bytes = input2.as_bytes();
            inputs.append(&mut input1_bytes.to_vec());
            inputs.append(&mut input2_bytes.to_vec());
            hasher.input(&inputs);
            // let mut out = [0u8; 32];
            // hasher.result(&mut out);
            let result = hasher.result_str();
            let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
            let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
            let result2 = decimal_number % modulo;
            let hex = format!("{:x}", result2);
            final_result = hex.clone();
        }
        // HashType::PEDERSEN => {
        //     let mut hasher = Pedersen::new();
        //     hasher.input_str(&input1);
        //     hasher.input_str(&input2);
        //     let hex = hasher.result_str();
        //     hex
        // }
        3 => {
            // let poseidon = Poseidon::new();
            // let mut sk = Poseidon_Fr::from_str(&input1).unwrap();
            // let input: Vec<Poseidon_Fr> = vec![sk.clone(), Poseidon_Fr::from_str(&input2).unwrap()];
            // let result = poseidon.hash(input.clone()).unwrap();
            // let result_bigint = result.into_bigint();
            // let result_bigint_string = result_bigint.to_string();
            // let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            // let hex_string = format!("{:x}", decimal_bigint);
            // final_result = hex_string.clone();
            let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
            let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
                .hash(input);
            let res_string = little_endian_u8_array_to_string(&result.to_bytes());
            final_result = res_string.clone();
        }
        4 => {
            let poseidon = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
            let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
            let input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from_str(&input2).unwrap()];
            let perm = poseidon.permutation(&input);
            let result = perm[0];
            let result_bigint = result.into_bigint();
            let result_bigint_string = result_bigint.to_string();
            let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
            let hex_string = format!("{:x}", decimal_bigint);
            final_result = hex_string.clone();
        }
        5 => {
            // let mut hasher = Mimc7::new(91);
            // let input1_fr = MiMC_Fr::from_str(&input1).unwrap();
            // let input2_fr = MiMC_Fr::from_str(&input2).unwrap();
            // let input: Vec<MiMC_Fr> = vec![input1_fr.clone(), input2_fr.clone()];
            // let result = hasher.multi_hash(input.clone(), &MiMC_Fr::zero());
            // let hex_string = to_hex(&result);
            // final_result = hex_string.clone();
            let res = mimc_hash_non_feistel(input1, input2, None);
            final_result = res.clone();
        }
        i32::MIN..=-1_i32 | 6_i32..=i32::MAX => {
            final_result = "Not implemented yet".to_string();
        }
    }
    final_result
}

// two-to-one hash  get threshold by sorting
#[pyfunction]
fn rayon_get_greenlist_id_and_threshold_use_multi_two_inputs_hash(seed: String, vocab_size: i32, green_list_size: i32, hash_type: i32) -> PyResult<(Vec<usize>, String)> {
    // println!("------seed: {:?}", seed);
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    let this_round_token_hash = multiple_two_inputs_hash_computation(seed.clone(), vocab_size, hash_type_enum.clone());
    // println!("------this_round_token_hash: {:?}", this_round_token_hash.len());
    let mut this_round_token_index: Vec<usize> = (0..vocab_size as usize).collect();
    this_round_token_index.par_sort_by(|&a, &b| this_round_token_hash[a].cmp(&this_round_token_hash[b]));
    let green_list_id = this_round_token_index[0..green_list_size as usize].to_vec();
    let threshold_index_string = this_round_token_index[green_list_size as usize - 1].to_string();
    // let threshold_index_16 = format!("{:x}", this_round_token_index[green_list_size as usize - 1]);
    // Note threshold_index_16!!!
    // println!("------green_list_id: {:?}", green_list_id.len());
    let this_round_threshold = two_inputs_hash_computation(seed, threshold_index_string, hash_type_enum);
    // println!("------this_round_threshold: {:?}", this_round_threshold.len());
    Ok((green_list_id, this_round_threshold))

}

#[pyfunction]
fn rayon_get_greenlist_id_and_fixed_threshold_use_multi_two_inputs_hash(seed: String, vocab_size: i32, green_list_size: i32, gamma: f64, big_prime: String, hash_type: i32) -> PyResult<(Vec<usize>, String)> {
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    let gamma_u64 = match gamma_to_u64(gamma) {
        Some(value) => value,
        None => return Err(PyValueError::new_err("Invalid gamma value")),
    };
    let big_prime_int = match BigUint::from_str_radix(&big_prime, 16) {
        Ok(val) => val,
        // Err(_) => return Err(PyValueError::new_err("Invalid big_prime value")),
        Err(_) => {
            println!("Error parsing big_prime");
            return Err(PyValueError::new_err("Invalid big_prime value"));
        }
    };
    // println!("big_prime_int: {:?}", big_prime_int);
    let fixed_threshold = big_prime_int / gamma_u64;
    let this_round_token_hash = multiple_two_inputs_hash_computation(seed.clone(), vocab_size, hash_type_enum.clone());
    let green_list_id = this_round_token_hash
        .into_par_iter()
        .enumerate()
        .filter_map(|(index, token_hash)| {
            // 对每个元素执行过滤和映射操作
            // if let Ok(value) = token_hash.parse::<BigUint>() {
            //     if value < fixed_threshold {
            //         Some(index) // 小于阈值的索引
            //     } else {
            //         None
            //     }
            // } else {
            //     None
            // }
            let res_int = match BigUint::from_str_radix(&token_hash, 16) {
                Ok(val) => val,
                Err(_) => return None,
            };
            if res_int < fixed_threshold {
                Some(index as usize) // 如果小于fixed_threshold，返回token序号
            } else {
                None // 否则，过滤掉这个token
            }
        })
        .collect();
    let threshold_index_string = fixed_threshold.to_str_radix(16);
    Ok((green_list_id, threshold_index_string))
}


#[pyfunction]
fn rayon_get_greenlist_id_and_threshold_use_multi_three_inputs_hash(secret_key: i64, pre_token_index: i64, vocab_size: i32, green_list_size: i32, hash_type: i32) -> PyResult<(Vec<usize>, String)> {
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    let secret_key_str = secret_key.to_string();
    let pre_token_index_str = pre_token_index.to_string();

    let this_round_token_hash: Vec<String> = (0..vocab_size).into_par_iter().map(|i| {
        let i_str = i.to_string();
        let res = three_inputs_hash_computation(secret_key_str.clone(), pre_token_index_str.clone(), i_str, hash_type_enum.clone());
        res
    }).collect();
    let mut this_round_token_index: Vec<usize> = (0..vocab_size as usize).collect();
    this_round_token_index.par_sort_by(|&a, &b| this_round_token_hash[a].cmp(&this_round_token_hash[b]));
    let green_list_id = this_round_token_index[0..green_list_size as usize].to_vec();
    let threshold_index_string = this_round_token_index[green_list_size as usize - 1].to_string();

    let this_round_threshold = three_inputs_hash_computation(secret_key.to_string(), pre_token_index.to_string(), threshold_index_string, hash_type_enum);
    
    Ok((green_list_id, this_round_threshold))

}

// #[pyfunction]
// fn rayon_get_greenlist_id_and_threshold_use_multi_three_inputs_hash(secret_key: i64, pre_token_index: i64, vocab_size: i32, green_list_size: i32, hash_type: i32) -> PyResult<(Vec<usize>, String)> {
//     let hash_type_enum = match hash_type {
//         0 => HashType::SHA256,
//         1 => HashType::BLAKE2b,
//         2 => HashType::KECCAK256,
//         3 => HashType::POSEIDON,
//         4 => HashType::POSEIDON2,
//         5 => HashType::MIMC,
//         // 6 => HashType::PEDERSEN,
//         _ => todo!("Not implemented yet"),
//     };
//     let secret_key_str = secret_key.to_string();
//     let pre_token_index_str = pre_token_index.to_string();

//     let this_round_token_hash: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::with_capacity(vocab_size as usize)));
//     (0..vocab_size).into_par_iter().for_each(|i| {
//         let i_str = i.to_string();
//         let res = three_inputs_hash_computation(secret_key_str.clone(), pre_token_index_str.clone(), i_str, hash_type_enum.clone());
//         this_round_token_hash.lock().unwrap().push(res);
//     });

//     let mut this_round_token_index: Vec<usize> = (0..vocab_size as usize).collect();
//     this_round_token_index.par_sort_by_key(|&i| {
//         let hash_vec = this_round_token_hash.lock().unwrap();
//         hash_vec[i].clone()
//     });

//     let green_list_id = this_round_token_index[0..green_list_size as usize].to_vec();
//     let threshold_index_string = this_round_token_index[green_list_size as usize - 1].to_string();

//     let this_round_threshold = three_inputs_hash_computation(secret_key.to_string(), pre_token_index.to_string(), threshold_index_string, hash_type_enum);
    
//     Ok((green_list_id, this_round_threshold))
// }




#[pyfunction]
fn rayon_get_greenlist_id_and_fixed_threshold_use_multi_three_inputs_hash(secret_key: i64, pre_token_index: i64, vocab_size: i32, green_list_size: i32, gamma: f64, big_prime: String, hash_type: i32) -> PyResult<(Vec<usize>, String)> {
    let hash_type_enum = match hash_type {
        0 => HashType::SHA256,
        1 => HashType::BLAKE2b,
        2 => HashType::KECCAK256,
        3 => HashType::POSEIDON,
        4 => HashType::POSEIDON2,
        5 => HashType::MIMC,
        // 6 => HashType::PEDERSEN,
        _ => todo!("Not implemented yet"),
    };
    let gamma_u64 = match gamma_to_u64(gamma) {
        Some(value) => value,
        None => return Err(PyValueError::new_err("Invalid gamma value")),
    };
    let big_prime_int = match BigUint::from_str_radix(&big_prime, 16) {
        Ok(val) => val,
        // Err(_) => return Err(PyValueError::new_err("Invalid big_prime value")),
        Err(_) => {
            println!("Error parsing big_prime");
            return Err(PyValueError::new_err("Invalid big_prime value"));
        }
    };
    // println!("big_prime_int: {:?}", big_prime_int);
    let fixed_threshold = big_prime_int / gamma_u64;
    let secret_key_str = secret_key.to_string();
    let pre_token_index_str = pre_token_index.to_string();

    let this_round_token_hash: Vec<String> = (0..vocab_size).into_par_iter().filter_map(|i| {
        let i_str = i.to_string();
        let res = three_inputs_hash_computation(secret_key_str.clone(), pre_token_index_str.clone(), i_str, hash_type_enum.clone());
        Some(res as String)
    }).collect();
    let green_list_id = this_round_token_hash
        .into_par_iter()
        .enumerate()
        .filter_map(|(index, token_hash)| {
            let res_int = match BigUint::from_str_radix(&token_hash, 16) {
                Ok(val) => val,
                Err(_) => return None,
            };
            if res_int < fixed_threshold {
                Some(index as usize) // 如果小于fixed_threshold，返回token序号
            } else {
                None // 否则，过滤掉这个token
            }
        })
        .collect();
    let threshold_index_string = fixed_threshold.to_str_radix(16);
    Ok((green_list_id, threshold_index_string))
}

#[pymodule]
fn hash_rustlib(_py: Python, m: &PyModule) -> PyResult<()>
{
    m.add_function(wrap_pyfunction!(rayon_get_greenlist_id_and_threshold_use_multi_two_inputs_hash, m)?)?;
    m.add_function(wrap_pyfunction!(single_two_inputs_hash_computation, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_get_greenlist_id_and_fixed_threshold_use_multi_two_inputs_hash, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_get_greenlist_id_and_threshold_use_multi_three_inputs_hash, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_get_greenlist_id_and_fixed_threshold_use_multi_three_inputs_hash, m)?)?;
    m.add_function(wrap_pyfunction!(_compute_keys_use_LCG, m)?)?;
    m.add_function(wrap_pyfunction!(_sample_g_values_use_LCG, m)?)?;
    m.add_function(wrap_pyfunction!(_compute_ngram_keys_use_LCG, m)?)?;
    m.add_function(wrap_pyfunction!(compute_LCG_random_use_rust, m)?)?;
    m.add_function(wrap_pyfunction!(single_two_inputs_hash_computation_decimal, m)?)?;
    m.add_function(wrap_pyfunction!(_compute_keys_use_hash, m)?)?;
    m.add_function(wrap_pyfunction!(_sample_g_values_use_hash, m)?)?;
    m.add_function(wrap_pyfunction!(_compute_ngram_keys_use_hash, m)?)?;
    
    Ok(())
}