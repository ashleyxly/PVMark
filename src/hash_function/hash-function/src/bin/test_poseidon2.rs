extern crate zkhash;

use rayon::vec;
use zkhash::{
    fields::bn256::FpBN256, gmimc::{gmimc::Gmimc, gmimc_instance_bn256::GMIMC_BN_3_PARAMS}, poseidon::{poseidon::Poseidon, poseidon_instance_bn256::POSEIDON_BN_PARAMS}, poseidon2::{self, poseidon2::Poseidon2, poseidon2_instance_bn256::POSEIDON2_BN256_PARAMS, poseidon2_instance_bn256_t_2::POSEIDON_2_BN256_PARAMS_T_2}
};
type Scalar = FpBN256;
use zkhash::fields::utils::from_hex;
use std::io::Write;
use crate::zkhash::ark_ff::PrimeField;
use num_bigint::BigInt;

fn test() {
    let poseidon2 = Poseidon2::new(&POSEIDON2_BN256_PARAMS);
    let mut input: Vec<Scalar> = vec![];
    let t = poseidon2.get_t();
    for i in 0..t {
        input.push(Scalar::from(i as u64));
    }
    let perm = poseidon2.permutation(&input);
    println!("perm: {:?}", perm);
    println!("perm[0]: {:?}", perm[0]);
    // assert_eq!(perm[0], from_hex("0x0bb61d24daca55eebcb1929a82650f328134334da98ea4f847f760054f4a3033"));
    // assert_eq!(perm[1], from_hex("0x303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570"));
    // assert_eq!(perm[2], from_hex("0x1ed25194542b12eef8617361c3ba7c52e660b145994427cc86296242cf766ec8"));

}

fn test2() {
    let poseidon2 = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
    let mut input: Vec<Scalar> = vec![Scalar::from(0u64), Scalar::from(1u64)];
    let perm = poseidon2.permutation(&input);
    println!("perm: {:?}", perm);
    println!("perm[0]: {:?}", perm[0]);

}

fn generate_multi_two_to_one_hash() {
    let poseidon2 = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
    let mut sk = Scalar::from(20242024u64);
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let mut input: Vec<Scalar> = vec![sk, Scalar::from(i as u64)];
        let perm = poseidon2.permutation(&input);
        let result = perm[0];
        results.push(result);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/poseidon2/results.txt").unwrap();
    for i in 0..len {
        let result = results[i];
        // let temp = result.into_bigint();
        // let result_str = temp.to_string_radix(16); // 转换为十六进制字符串
        let result_str = result.to_string();
        // println!("result_str: {:?}", result_str);
        let decimal_bigint = num_bigint::BigUint::parse_bytes(result_str.as_bytes(), 10).unwrap();
        // println!("decimal_bigint: {:?}", decimal_bigint);
        let hex_string_hex = format!("{:x}", decimal_bigint);
        // println!("hex_string_hex: {:?}", hex_string_hex);
        // file.write_all(decimal_bigint.as_bytes()).unwrap();
        file.write_all(hex_string_hex.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }

}

fn generate_multi_three_to_one_hash() {
    let poseidon2 = Poseidon2::new(&POSEIDON2_BN256_PARAMS);
    let len = 51000;
    let pre_token_index = 16732;
    let mut results = vec![];
    for i in 0..len {
        let mut input: Vec<Scalar> = vec![Scalar::from(20242024u64), Scalar::from(pre_token_index as u64), Scalar::from(i as u64)];
        let perm = poseidon2.permutation(&input);
        let result = perm[0];
        results.push(result);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/poseidon2/results_2.txt").unwrap();
    for i in 0..len {
        let result = results[i];
        let result_str = result.to_string();
        let decimal_bigint = num_bigint::BigUint::parse_bytes(result_str.as_bytes(), 10).unwrap();
        let hex_string_hex = format!("{:x}", decimal_bigint);
        file.write_all(hex_string_hex.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}

fn main() {
    // test();
    // println!("-----------------");
    // test2();
    // generate_multi_two_to_one_hash();
    generate_multi_three_to_one_hash();
}