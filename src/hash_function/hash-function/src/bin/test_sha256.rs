extern crate crypto;
use self::crypto::digest::Digest;
use self::crypto::sha2::Sha256;
use ark_bn254::Fr;
use std::str::FromStr;
use ark_ff::PrimeField;
use num_bigint::BigUint;
use ark_ff::BigInteger;
use num_traits::Num;
use std::io::Write;

fn test() {
    // create a Sha256 object
    let mut hasher = Sha256::new();

    // write input message
    hasher.input_str("hello world");

    // read hash digest
    let hex = hasher.result_str();
    println!("hex: {:?}", hex);

    assert_eq!(hex,
            concat!("b94d27b9934d3e08a52e52d7da7dabfa",
                    "c484efe37a5380ee9088f7ace2efcde9"));

}

fn generate_multi_two_to_one_hash() {
    let sk = Fr::from_str("20242024").unwrap();
    let sk_bigint = sk.into_bigint();
    let sk_bytes = sk_bigint.to_bytes_be();
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let mut hasher = Sha256::new();
        let mut inputs: Vec<u8> = vec![];
        inputs.append(&mut sk_bytes.clone());
        let i_fr = Fr::from(i as u64);
        let i_bigint = i_fr.into_bigint();
        let i_bytes = i_bigint.to_bytes_be();
        inputs.append(&mut i_bytes.clone());
        hasher.input(&inputs);
        let hex = hasher.result_str();
        let decimal_number = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap();
        // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        // let result = decimal_number % modulo;
        results.push(decimal_number);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/sha256/results_2.txt").unwrap();
    for i in 0..len {
        let result = results[i].clone();
        let hex_string = format!("{:x}", result);
        file.write_all(hex_string.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}

fn generate_multi_three_to_one_hash() {
    let sk = Fr::from_str("20242024").unwrap();
    let sk_bigint = sk.into_bigint();
    let sk_bytes = sk_bigint.to_bytes_be();
    let pre_token_index: i32 = 1648;
    let pre_token_index_fr = Fr::from(pre_token_index as u64);
    let pre_token_index_bigint = pre_token_index_fr.into_bigint();
    let pre_token_index_bytes = pre_token_index_bigint.to_bytes_be();
    let len = 51000;
    let mut results = vec![];
    for i in 0..len {
        let mut hasher = Sha256::new();
        let mut inputs: Vec<u8> = vec![];
        inputs.append(&mut sk_bytes.clone());
        inputs.append(&mut pre_token_index_bytes.clone());
        let i_fr = Fr::from(i as u64);
        let i_bigint = i_fr.into_bigint();
        let i_bytes = i_bigint.to_bytes_be();
        inputs.append(&mut i_bytes.clone());
        hasher.input(&inputs);
        let hex = hasher.result_str();
        let decimal_number = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap();
        // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        // let result = decimal_number % modulo;
        results.push(decimal_number);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/sha256/results_3.txt").unwrap();
    for i in 0..len {
        let result = results[i].clone();
        let hex_string = format!("{:x}", result);
        file.write_all(hex_string.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }

}

fn main() {
    // test();
    // generate_multi_two_to_one_hash();
    generate_multi_three_to_one_hash();
}