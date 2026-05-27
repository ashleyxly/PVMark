use std::ffi::NulError;
use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;

use std::time::{Duration, Instant};

// use halo2_proofs::pasta::{Eq, EqAffine};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, BatchVerifier, Circuit,
    Column, ConstraintSystem, Fixed, TableColumn,
};
// use halo2_proofs::poly::commitment::{Guard, MSM};
// use halo2_proofs::poly::{commitment::Params, Rotation};
// use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge};
use halo2_proofs::dev::VerifyFailure;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
use halo2_gadgets::poseidon::{primitives::*, Hash, Pow5Chip, Pow5Config};
use halo2_proofs::arithmetic::Field;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::halo2curves;
use halo2_proofs::{circuit::{*, self}, plonk::*};
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSlice;

use std::marker::PhantomData;

use halo2_proofs::poly::Rotation;
use halo2_proofs::dev::MockProver;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, TranscriptReadBuffer,
    TranscriptWriterBuffer,
};
use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};

use halo2_proofs::poly::commitment::CommitmentScheme;

use std::error::Error;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;

use num_bigint::BigUint;

pub fn gamma_to_u64(gamma: f64) -> Option<u64> {
    if gamma <= 0.0 {
        // gamma 必须为正数
        return None;
    }

    let inverse_gamma = 1.0 / gamma;

    // 检查是否在 u64 的表示范围内
    if inverse_gamma >= u64::MAX as f64 || inverse_gamma <= 0.0 {
        return None;
    }

    Some(inverse_gamma as u64) // 转换为 u64，小数部分将被截断
}

pub fn dec_num_fp(org_value: Fp, mod_index: usize, dec_num: usize) -> (Fp, [Fp; 4]) {
    let mut value_bytes = org_value.to_bytes();
    let mut vec_dec_inputs: Vec<Fp> = Vec::new();
    for i in 0..dec_num {
        vec_dec_inputs.push(Fp::from(u64::from_le_bytes(value_bytes[i*8..(i+1)*8].try_into().unwrap())));
    }
    (vec_dec_inputs[mod_index].clone(), [vec_dec_inputs[0], vec_dec_inputs[1], vec_dec_inputs[2], vec_dec_inputs[3]])
    // (Fp::from(u64::from_le_bytes(value_bytes[0..8].try_into().unwrap())), Fp::from(u64::from_le_bytes(value_bytes[8..16].try_into().unwrap())), Fp::from(u64::from_le_bytes(value_bytes[16..24].try_into().unwrap())), Fp::from(u64::from_le_bytes(value_bytes[24..32].try_into().unwrap())))
    // [
    //     Fp::from(u64::from_le_bytes(value_bytes[0..8].try_into().unwrap())),
    //     Fp::from(u64::from_le_bytes(value_bytes[8..16].try_into().unwrap())),
    //     Fp::from(u64::from_le_bytes(value_bytes[16..24].try_into().unwrap())),
    //     Fp::from(u64::from_le_bytes(value_bytes[24..32].try_into().unwrap())),
    // ]
}


pub fn process_hex_string(hex_string: &str) -> [u64; 4] {
    // 步骤 1: 将hex_string分割为4个长度为16的string
    let chunks: Vec<&str> = hex_string.as_bytes().chunks(16).map(|chunk| std::str::from_utf8(chunk).unwrap()).collect();

    // 步骤 2: 将每个长度为16的string转化为u64
    let u64_values: Vec<u64> = chunks.into_iter().map(|s| u64::from_str_radix(s, 16).unwrap()).collect();

    // 步骤 3: 将所有的string按照倒序存到[u64;4]
    let mut result_array: [u64; 4] = Default::default();
    result_array.copy_from_slice(&u64_values[..4]);
    result_array.reverse();
    
    result_array
    
}

pub fn process_hex_string_2(hex_string: &str) -> [u8; 32] {
    // 步骤 1: 将hex_string分割为4个长度为16的string
    let chunks: Vec<&str> = hex_string.as_bytes().chunks(8).map(|chunk| std::str::from_utf8(chunk).unwrap()).collect();

    // 步骤 2: 将每个长度为16的string转化为u64
    let u8_values: Vec<u8> = chunks.into_iter().map(|s| u8::from_str_radix(s, 8).unwrap()).collect();

    // 步骤 3: 将所有的string按照倒序存到[u64;4]
    let mut result_array: [u8; 32] = Default::default();
    result_array.copy_from_slice(&u8_values[..32]);
    result_array.reverse();
    
    result_array
    
}

pub fn pad_string(input: &str, target_length: usize) -> String {
    let current_length = input.len();
    
    if current_length >= target_length {
        return input.to_string();
    }

    let padding_length = target_length - current_length;
    let padding = "0".repeat(padding_length);

    let padded_string = format!("{}{}", padding, input);

    padded_string
}

pub fn pow_of_two(by: usize) -> Fp {
    let pow_value = BigUint::from(2u64).pow(by as u32);
    let hex_string = format!("{:X}", pow_value);
    let padded_hex_string = pad_string(&hex_string, 64);
    let result_array = process_hex_string(&padded_hex_string);
    Fp::from_raw(result_array)
}



pub fn expr_from_bytes(bytes: &[Expression<Fp>]) -> Expression<Fp> {
    let mut value = Expression::Constant(Fp::ZERO);
    let mut multiplier = Fp::ONE;
    for byte in bytes.iter() {
        value = value + byte.clone() * Expression::Constant(multiplier);
        multiplier *= Fp::from(256);
    }
    value
}

pub fn bool_check(value: Expression<Fp>) -> Expression<Fp> {
    range_check(value, 2)
}

/// Restrict an expression such that 0 <= word < range.
pub fn range_check(word: Expression<Fp>, range: usize) -> Expression<Fp> {
    (1..range).fold(word.clone(), |acc, i| {
        acc * (Expression::Constant(Fp::from(i as u64)) - word.clone())
    })
}


pub fn gen_srs<Scheme: CommitmentScheme>(k: u32) -> Scheme::ParamsProver {
    Scheme::ParamsProver::new(k)
}

pub fn load_srs<Scheme: CommitmentScheme>(
    path: PathBuf,
) -> Result<Scheme::ParamsVerifier, Box<dyn Error>> {
    println!("loading srs from {:?}", path);
    let f = File::open(path.clone())
        .map_err(|_| format!("failed to load srs at {}", path.display()))?;
    let mut reader = BufReader::new(f);
    Params::<'_, Scheme::Curve>::read(&mut reader).map_err(Box::<dyn Error>::from)
}

/// helper function for load_params
pub fn load_params_cmd(
    srs_path: PathBuf,
    logrows: u32,
// ) -> ParamsKZG<Bn256> {
) -> Result<ParamsKZG<Bn256>, Box<dyn Error>> {
    let mut params: ParamsKZG<Bn256> = load_srs::<KZGCommitmentScheme<Bn256>>(srs_path)?;
    println!("downsizing params to {} logrows", logrows);
    if logrows < params.k() {
        params.downsize(logrows);
    }
    Ok(params)
    // params
}

pub fn select_suitable_k_value(total_row: i32) -> u32 {
    let mut new_k: u32 = 1;
    let mut pow_new_k = 2;
    while pow_new_k < total_row {
        pow_new_k <<= 1;
        new_k += 1;
    }

    new_k
}

